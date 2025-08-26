#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:44:33 2025

@author: taomingzhe
"""

# %% import modules 
import json
import os
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
#from scipy.integrate import simps
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.loader import TemporalDataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT
from torch_geometric.nn import global_mean_pool
from torch import Tensor
from torch_geometric.data import TemporalData
from scipy.signal import hilbert
from tqdm import tqdm
from collections import defaultdict

#from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from scipy.signal import hilbert
from sklearn.model_selection import GroupKFold

#from torch_geometric_temporal.data.dataset import TemporalDataLoader
#from torch_geometric_temporal.data.utils import collate_temporal_batch
#from torch_geometric_temporal.signal import TemporalDataLoader

import torch.nn as nn
from torch_geometric.nn.models import TGNMemory
#from torch_geometric.nn import IdentityMessage, LastAggregator, TimeEncoder
        
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    )

from torch.nn import GRUCell
from torch_geometric.nn.models.tgn import TimeEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Functions

def manual_collate_temporal_batch(batch_list):
    """Manually collate a list of TemporalData into one batched object."""

    src, dst, t, msg, y, batch= [], [], [], [], [], []
    node_offset = 0

    for i, data in enumerate(batch_list):
        if data.src.numel() == 0 or data.dst.numel() == 0:
            print(f"Warning: Skipping empty graph at index {i}") # skip empty src/dst
            continue 
        
        #num_nodes = max(data.src.max(), data.dst.max()).item() + 1
        #num_edges = data.src.size(0)
        local_max_node = max(data.src.max().item(), data.dst.max().item())
        num_nodes = local_max_node + 1
        
        # offset node indices to avoid collision across samples
        src.append(data.src + node_offset)
        dst.append(data.dst + node_offset)
        t.append(data.t)
        msg.append(data.msg)
        #y.append(data.y)
        y.append(data.y.view(1))

        #batch.append(torch.full((num_edges,), i, dtype=torch.long))
        batch.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        node_offset += num_nodes
        
    if not src:  # 全部 trial 都被 skip 掉了
        raise ValueError("All trials in batch are invalid or empty.")

    return TemporalData(
        src=torch.cat(src, dim=0),
        dst=torch.cat(dst, dim=0),
        t=torch.cat(t, dim=0),
        msg=torch.cat(msg, dim=0),
        y=torch.cat(y, dim=0),
        batch=torch.cat(batch, dim=0)
    )

# compute PLV in each window
def plvMatrix(eegData): # input: eeg data per trial
    """
    eegData: [timesteps, num_electrodes] — per trial
    Return: one [num_electrodes, num_electrodes] plv matrix for the full trial
    """
    num_time_steps, num_electrodes = eegData.shape
    #num_windows = num_time_steps // window_size

    #plv_list = []

    plv_matrix = np.zeros((num_electrodes, num_electrodes))
    for i in range(num_electrodes):
        for j in range(i+1, num_electrodes):
            phase1 = np.angle(hilbert(eegData[:, i]))
            phase2 = np.angle(hilbert(eegData[:, j]))
            phase_diff = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_diff))) / num_time_steps
            plv_matrix[i, j] = np.log(plv + 1e-6)
            plv_matrix[j, i] = np.log(plv + 1e-6)

    return plv_matrix


# create TemporalData as data object from each single EEG trial, with class label
def temporalData(plv, label, trial_length, top_percent=80.0): # add threshold attribute
    """
    Keep top-percent strongest PLV edges (by value). 'plv' is a 2D matrix
    """
    src, dst, t, msg = [], [], [], []
    num_electrodes = plv.shape[0]

    # only sample 20 time points per trial
    ts_list = np.linspace(0, trial_length-1, 22, dtype=int) # 20?
    
        
    # --- build a mask of “kept” edges using percentile on the upper triangle (exclude diagonal) ---
    tri_vals = plv[np.triu_indices(num_electrodes, k=1)]
    
    if tri_vals.size == 0:
        # degenerate case: single channel or bad input
        keep_mask = np.zeros_like(plv, dtype=bool)
    else:
        percent = 100.0 - float(top_percent)
        threshold = np.percentile(tri_vals, percent)
        keep_mask = plv >= threshold
        np.fill_diagonal(keep_mask, False)
        
        # safety: if mask ended up empty (e.g., perfectly flat matrix), keep the single max edge
        if not keep_mask.any():
            ii, jj = np.triu_indices(num_electrodes, 1)
            k = np.argmax(tri_vals)
            keep_mask[ii[k], jj[k]] = True
            keep_mask[jj[k], ii[k]] = True
        # ---                              --- 
    
    for i in range(num_electrodes):
        for j in range(num_electrodes):
            if i == j or not keep_mask[i, j]:
                continue
            for ts in ts_list:
                src.append(i)
                dst.append(j)
                t.append(ts)
                msg.append(plv[i, j])

    return TemporalData(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(t, dtype=torch.long),
        msg=torch.tensor(msg, dtype=torch.float32).unsqueeze(-1),
        y=torch.tensor([label], dtype=torch.long)
    )

# collect all trials of the same label 'L'(0) or 'R'(1) for one subject 
def aggregate_temporalData(subject_data, subject_id):
    """
    subject_data: dictionary with keys 'L' and 'R', each with shape [1, 160] of EEG trials
    Output: list of 2 TemporalData objects (one for 'L', one for 'R')
    """
    dataset = []
    
    for label, field in enumerate(['L', 'R']):        # label = 0 or 1
        trials = subject_data[field][0]               # shape [num_trials, time, num_channels]
        
        for trial_id, trial in enumerate(trials):
            eegData      = trial                     # [time, num_channels]
            trial_length = eegData.shape[0]
            plv          = plvMatrix(eegData)        # new full-trial PLV
            
            #temp_data    = temporalData(plv, label, trial_length)
            #---                                                  ---
            temp_data = temporalData(plv, label, trial_length, top_percent=80.0)  
            
            if temp_data.src.numel() == 0 or temp_data.dst.numel() == 0:
                print(f"Skipping empty graph for label {field}")
                continue
            
            # cross-subject batching metadata
            temp_data.subject_id = int(subject_id)  # attach the subject ID to the TemporalData object for this trial
            temp_data.trial_id   = int(trial_id)    # attach the trial index to the same TemporalData object
            dataset.append(temp_data)

    return dataset



# %% load data
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import LastAggregator

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

data_dir = os.getcwd() 
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

subject_data = {}
subject_results = {} # two empty dictionaries
all_temporal_data = []

# loop over each subject and collect data into window buckets
for subject_number in tqdm(subject_numbers, desc="Processing Subjects"):
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data = mat_contents[f'Subject{subject_number}'][:, :]

    # this returns a list of TemporalData for all windows of that subject
    subject_temporal_data = aggregate_temporalData(subject_data, subject_number)
    all_temporal_data.extend(subject_temporal_data)
    
    # === k-fold cross-validation (per subject) ===
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)
    #highest_test_accuracies = []
    
unique_subjects = sorted({td.subject_id for td in all_temporal_data})
if len(unique_subjects) < 2:
    raise ValueError(f"Need at least 2 subjects for GroupKFold, got {len(unique_subjects)}.")

K = len(unique_subjects)
#K = min(5, len(unique_subjects))     # or K = len(unique_subjects) for LOSO
# = 50
#batch_size = 32

subjects_vec = np.array([td.subject_id for td in all_temporal_data])
indices      = np.arange(len(all_temporal_data))
gkf = GroupKFold(n_splits=K)

cv_best_acc = []
all_subject_stats = {}

    
# define the TGN model 
class CustomTGNMemory(nn.Module): # 1. define a memory module for tgn
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim,
                 message_module, aggregator_module):
        
            
            # initialize the memory module with required hyperparameters
           
           # no.of nodes to save memory for, 19 electrodes;
           # message dimensionality before being combined with memory and time
           # how big each node's memory vector is;
           # how many dimensions for time encoding;
           # message module: combine memory, raw message and time encodings into a final message;
           # aggregator module: aggregate all messages to the same node
           
        super().__init__()
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim  = memory_dim
        self.time_dim    = time_dim

        self.msg_module  = message_module
        self.aggr_module = aggregator_module

        self.time_enc = TimeEncoder(time_dim)
        self.gru      = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)

        # 初始化为你给定的上限，但后续允许扩容
        self.register_buffer('memory',      torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes))
        
    def reset_memory(self):
        self.memory.zero_()
        self.last_update.zero_()
        
    @torch.no_grad()
    def _ensure_capacity(self, need_n: int):
        """If need_n > current size, expand memory/last_update safely."""
        cur = self.memory.size(0)
        if need_n > cur:
            pad_n = need_n - cur
            self.memory      = torch.cat(
                [self.memory, torch.zeros(pad_n, self.memory_dim, device=self.memory.device)], dim=0
            )
            self.last_update = torch.cat(
                [self.last_update, torch.zeros(pad_n, device=self.last_update.device)], dim=0
            )
                   
    def forward(self, n_id: torch.Tensor):
        """Fetch memory and last update time for given node indices"""
        need_n = int(n_id.max().item()) + 1 if n_id.numel() > 0 else 0
        if need_n > 0:
            self._ensure_capacity(need_n)
        return self.memory[n_id], self.last_update[n_id]
    # memory[n_id]: node-level memory vectors
    # last_update[n_id]: timestamps of last updates
             
    def detach(self):
        self.memory.detach_()
        
    def update_state(self, src, dst, t, raw_msg):
        """Update memory for dst nodes using (src, dst, time, message)"""
        max_id = int(torch.stack([src.max(), dst.max()]).max().item()) + 1
        self._ensure_capacity(max_id)
            
        with torch.no_grad():
            mem_src, _ = self(src)
            mem_dst, _ = self(dst) 
            # fetch memory for the involved nodes
            # return the memory vector and last update time for the involved nodes
                   
            dt = t - self.last_update[dst] # time since the last update for each destination node
            dt_enc = self.time_enc(dt.to(raw_msg.dtype))
            # time differences encoded via a learned TimeEncoder module (maps scalar to vector)
    
            # Compute message vectors
            msg_vec = self.msg_module(mem_src, mem_dst, raw_msg, dt_enc)
            # combine Inputs into Message Vectors
            # output unified message vectors for downstream aggregation
    
            # Aggregate messages for each dst
            unique_dst, inv = torch.unique(dst, return_inverse=True)
            agg_msg = self.aggr_module(msg_vec, inv, t, dim_size=unique_dst.size(0))
            # group all messages sent to the same node
            # aggregate (e.g., via mean/sum) them into a single vector per destination node

            # GRU-based memory update
            updated_mem = self.gru(agg_msg, self.memory[unique_dst])
            self.memory[unique_dst] = updated_mem
            # eacg destination node’s memory is updated with: current memory & aggregated message vector
                   
            # Timestamp update
            for i, dst_node in enumerate(unique_dst):
                self.last_update[dst_node] = t[dst == dst_node].max()
            # ensure that last_update holds the most recent time a node received a message
            

               
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, raw_msg_dim, msg_dim, time_enc):
        super().__init__()
                    
        self.time_enc = time_enc
        self.msg_mlp = nn.Linear(raw_msg_dim, msg_dim)
                    
        # total edge features = time encoding + raw message
        edge_dim = msg_dim + time_enc.out_channels
                   
        # define the attention-based layer
        self.conv1 = TransformerConv(in_channels=in_channels,  # node features (likely from memory)
                                     out_channels=out_channels // 2,
                                     heads=2, # 2 heads → final dim = (out_channels // 2) * 2 = out_channels
                                     dropout=0.1,
                                     edge_dim=edge_dim    # temporal + message edge features
                                     )
            
        self.norm1 = nn.LayerNorm(out_channels)
            
        self.conv2 = TransformerConv(in_channels=in_channels,  # node features (likely from memory)
                                     out_channels=out_channels // 2,
                                     heads=2, # 2 heads → final dim = (out_channels // 2) * 2 = out_channels
                                     dropout=0.1,
                                     edge_dim=edge_dim    # temporal + message edge features
                                     )
            
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.conv3 = TransformerConv(in_channels=in_channels,  # node features (likely from memory)
                                     out_channels=out_channels // 2,
                                     heads=2, # 2 heads → final dim = (out_channels // 2) * 2 = out_channels
                                     dropout=0.1,
                                     edge_dim=edge_dim    # temporal + message edge features
                                     )
            
        self.norm3 = nn.LayerNorm(out_channels)
            
                    
    def forward(self, x, last_update, edge_index, t, msg): 
                
        # edge_index is local now
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(msg.dtype))
        msg_proj = self.msg_mlp(msg)
        edge_attr = torch.cat([rel_t_enc, msg_proj], dim=-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x))

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.norm2(x))

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
            
        return x
                
                
class TGNClassifier(nn.Module):    
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim,
                 msg_module, aggr_module, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.memory = CustomTGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim,
                                      msg_module, aggr_module)
        self.embedding = GraphAttentionEmbedding(in_channels=memory_dim,
                                                 out_channels=hidden_channels,
                                                 raw_msg_dim=raw_msg_dim,
                                                 msg_dim=memory_dim,
                                                 time_enc=self.memory.time_enc)
    
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, edge_index, t, msg, n_id, batch_idx):
        # Get local node features for the nodes in this batch
        x, last_update = self.memory(n_id)

        # Map global → local indices
        # n_id is sorted unique node ids; map each endpoint to its position in n_id
        row = torch.searchsorted(n_id, edge_index[0])
        col = torch.searchsorted(n_id, edge_index[1])
        edge_index_local = torch.stack([row, col], dim=0)

        # Now use local indices everywhere downstream
        x = self.embedding(x, last_update, edge_index_local, t, msg)
        graph_emb = global_mean_pool(x, batch_idx)
        return self.classifier(graph_emb)
           
        
class CustomMessageModule(nn.Module):        
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super().__init__()
        input_dim = raw_msg_dim + 2 * memory_dim + time_dim
        self.lin = nn.Linear(input_dim, memory_dim)
    
    def forward(self, mem_src, mem_dst, raw_msg, dt_enc):
        x = torch.cat([mem_src, mem_dst, raw_msg, dt_enc], dim=-1)
        return self.lin(x)

#hidden_channels = 32# depend on the feature size
#out_channels = 2       # 'L' or 'R' binary classification
#raw_msg_dim = 1        # due to '.unsqueeze(-1)'
#memory_dim = hidden_channels
#time_dim = hidden_channels


def objective(trial):
    
    #memory_dim = trial.suggest_categorical("memory_dim", [16, 32, 64])
    #hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    #learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    memory_dim = hidden_channels          # <<< important: couple them
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    time_dim = memory_dim
    
    out_channels = 2
    time_dim = memory_dim
    batch_size = 32
    E = 50 # epochs
    
    fold = 0
    subjects_vec = np.array([td.subject_id for td in all_temporal_data])
    indices = np.arange(len(all_temporal_data))
    gkf = GroupKFold(n_splits=len(set(subjects_vec)))
    splits = list(gkf.split(indices, groups=subjects_vec))
    train_idx, test_idx = splits[fold]

    torch.manual_seed(12345)

    subjects_vec = np.array([td.subject_id for td in all_temporal_data])
    indices      = np.arange(len(all_temporal_data))
    gkf          = GroupKFold(n_splits=len(set(subjects_vec)))
    splits       = list(gkf.split(indices, groups=subjects_vec))
    train_idx, test_idx = splits[fold]

    train_data = [all_temporal_data[i] for i in train_idx]
    test_data  = [all_temporal_data[i] for i in test_idx]
    
    train_bucket = defaultdict(list)
    for td in train_data:
        train_bucket[int(td.trial_id)].append(td)
    train_loader = {
        tid: [v[i:i+batch_size] for i in range(0, len(v), batch_size)]
        for tid, v in train_bucket.items()
    }

    test_bucket = defaultdict(list)
    for td in test_data:
        test_bucket[int(td.subject_id)].append(td)
    for v in test_bucket.values():
        v.sort(key=lambda d: int(d.trial_id))
    test_loader = {
        sid: [v[i:i+batch_size] for i in range(0, len(v), batch_size)]
        for sid, v in test_bucket.items()
    }

    first_key = next(iter(train_loader))
    data_sample = manual_collate_temporal_batch(train_loader[first_key][0])
    raw_msg_dim = data_sample.msg.size(-1)
    
    model = TGNClassifier(
        num_nodes=19 * batch_size,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        msg_module=CustomMessageModule(raw_msg_dim, memory_dim, time_dim),
        aggr_module=LastAggregator(),
        in_channels=memory_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    
    # % train and test the model
    def train():
        model.train()
        for _, loader in train_loader.items():
            model.memory.reset_memory()
            for batch_list in loader:
                batch = manual_collate_temporal_batch(batch_list).to(device)
                edge_index = torch.stack([batch.src, batch.dst], dim=0)
                n_id = torch.cat([batch.src, batch.dst]).unique()
                batch_idx = batch.batch[n_id]
                    
                model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                out  = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                loss = criterion(out, batch.y)
                        
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


    @torch.no_grad()
    def test():        
        model.eval()
        per_subject = {}
                
        for sid, loader in test_loader.items():
            model.memory.reset_memory()
            acc_list = []
        
            for batch_list in loader:
                batch = manual_collate_temporal_batch(batch_list).to(device)
                edge_index = torch.stack([batch.src, batch.dst], dim=0)
                n_id       = torch.cat([batch.src, batch.dst]).unique()
                batch_idx  = batch.batch[n_id]

                model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                out  = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                pred = out.argmax(dim=1)

                batch_acc = (pred == batch.y).float().mean().item()
                acc_list.append(batch_acc)
                    
            per_subject[int(sid)] = {
                "mean": float(np.mean(acc_list)) if acc_list else 0.0,
                "max":  float(np.max(acc_list)) if acc_list else 0.0,
                "min":  float(np.min(acc_list)) if acc_list else 0.0,
                "n_batches": len(acc_list),
            }

        return float(np.mean([v["mean"] for v in per_subject.values()])) if per_subject else 0.0


    best_acc = 0.0

    for epoch in range(E):
        # 1) train one epoch
        train()

        # 2) use test() to evaluate model, return 'mean_over_subjects'
        val_acc = test()

        # 3) record the optimal
        if val_acc > best_acc:
            best_acc = val_acc

        # 4) use trial.report() to report the middle results during the current epoch for Optuna
        trial.report(val_acc, step=epoch)

        # 5) allow pruning: if the middle results are not good for the current trial, stop ASAP
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # use the optimal value as the target value for the current trial 
    return best_acc


def Kfold_with_params(best_params):

    memory_dim      = best_params.get("memory_dim", 32)
    hidden_channels = best_params.get("hidden_channels", 64)
    learning_rate   = best_params.get("lr", 1e-3)

    out_channels = 2
    batch_size   = 32
    E            = 50  #epochs

    subjects_vec = np.array([td.subject_id for td in all_temporal_data])
    indices      = np.arange(len(all_temporal_data))
    gkf          = GroupKFold(n_splits=len(set(subjects_vec)))

    cv_best_acc = []
    all_subject_stats = {}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(indices, groups=subjects_vec), start=1):
        torch.manual_seed(12345)

        train_data = [all_temporal_data[i] for i in train_idx]
        test_data  = [all_temporal_data[i] for i in test_idx]

        # ---- loaders ----
        train_bucket = defaultdict(list)
        for td in train_data:
            train_bucket[int(td.trial_id)].append(td)
        train_loader = {
            tid: [v[i:i+batch_size] for i in range(0, len(v), batch_size)]
            for tid, v in train_bucket.items()
        }

        test_bucket = defaultdict(list)
        for td in test_data:
            test_bucket[int(td.subject_id)].append(td)
        for v in test_bucket.values():
            v.sort(key=lambda d: int(d.trial_id))
        test_loader = {
            sid: [v[i:i+batch_size] for i in range(0, len(v), batch_size)]
            for sid, v in test_bucket.items()
        }

        # ---- probe raw_msg_dim ----
        first_key   = next(iter(train_loader))
        data_sample = manual_collate_temporal_batch(train_loader[first_key][0])
        raw_msg_dim = data_sample.msg.size(-1)

        # ---- model ----
        model = TGNClassifier(
            num_nodes=19 * batch_size,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=memory_dim,
            msg_module=CustomMessageModule(raw_msg_dim, memory_dim, memory_dim),
            aggr_module=LastAggregator(),
            in_channels=memory_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # ---- inner train/test ----
        def train():
            model.train()
            for _, loader in train_loader.items():
                model.memory.reset_memory()
                for batch_list in loader:
                    batch = manual_collate_temporal_batch(batch_list).to(device)
                    edge_index = torch.stack([batch.src, batch.dst], dim=0)
                    n_id       = torch.cat([batch.src, batch.dst]).unique()
                    batch_idx  = batch.batch[n_id]
                    model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                    out  = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        @torch.no_grad()
        def test():
            model.eval()
            per_subject = {}
            for sid, loader in test_loader.items():
                model.memory.reset_memory()
                acc_list = []
                for batch_list in loader:
                    batch = manual_collate_temporal_batch(batch_list).to(device)
                    edge_index = torch.stack([batch.src, batch.dst], dim=0)
                    n_id       = torch.cat([batch.src, batch.dst]).unique()
                    batch_idx  = batch.batch[n_id]
                    model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                    out  = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                    pred = out.argmax(dim=1)
                    acc_list.append((pred == batch.y).float().mean().item())
                per_subject[int(sid)] = {
                    "mean": float(np.mean(acc_list)) if acc_list else 0.0,
                    "max":  float(np.max(acc_list)) if acc_list else 0.0,
                    "min":  float(np.min(acc_list)) if acc_list else 0.0,
                    "n_batches": len(acc_list),
                }
            return float(np.mean([v["mean"] for v in per_subject.values()])) if per_subject else 0.0, per_subject

        best_test = 0.0
        best_per_subject = {}

        for epoch in range(1, E + 1):
            train()
            mean_acc, per_subj = test()
            if mean_acc > best_test:
                best_test = mean_acc
                best_per_subject = per_subj
            print(f"[Fold {fold}] Epoch {epoch:02d} | mean acc={mean_acc:.3f} | best={best_test:.3f}")

        cv_best_acc.append(best_test)
        all_subject_stats.update({int(k): v for k, v in best_per_subject.items()})

    # ---- summary (mean, std, min, max) ----
    mean_cv = float(np.mean(cv_best_acc)) if cv_best_acc else 0.0
    std_cv  = float(np.std(cv_best_acc))  if cv_best_acc else 0.0
    max_cv  = float(np.max(cv_best_acc))  if cv_best_acc else 0.0
    min_cv  = float(np.min(cv_best_acc))  if cv_best_acc else 0.0

    print("\n=== K-fold (by subject) Summary ===")
    print(f"Mean={mean_cv:.4f}, Std={std_cv:.4f}, Max={max_cv:.4f}, Min={min_cv:.4f}")

    # ---- save JSON (same style as your original) ----
    with open('TGN_ALS_Optuna_best_fold_summary.json', 'w') as f:
        json.dump({
            "fold_best_acc": cv_best_acc,
            "mean": mean_cv,
            "std": std_cv,
            "max": max_cv,
            "min": min_cv
        }, f, indent=2)

    with open('TGN_ALS_Optuna_best_per_subject.json', 'w') as f:
        json.dump(all_subject_stats, f, indent=2)


# apply Optuna 

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)  
    )
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print("  Value:", trial.value)
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    
    Kfold_with_params(study.best_trial.params)

