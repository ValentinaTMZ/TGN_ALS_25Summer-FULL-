#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:44:33 2025

@author: taomingzhe
"""

# %% import modules 
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

#from torch_geometric_temporal.data.dataset import TemporalDataLoader
#from torch_geometric_temporal.data.utils import collate_temporal_batch
#from torch_geometric_temporal.signal import TemporalDataLoader


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
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix


# create TemporalData as data object from each single EEG trial, with class label
def temporalData(plv, label, trial_length):
    src, dst, t, msg = [], [], [], []
    num_electrodes = plv.shape[0]

    # only sample 20 time points per trial
    ts_list = np.linspace(0, trial_length-1, 22, dtype=int) # 20?

    for i in range(num_electrodes):
        for j in range(num_electrodes):
            if i == j:
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
def aggregate_temporalData(subject_data):
    """
    subject_data: dictionary with keys 'L' and 'R', each with shape [1, 160] of EEG trials
    Output: list of 2 TemporalData objects (one for 'L', one for 'R')
    """
    dataset = []
    
    for label, field in enumerate(['L', 'R']):        # label = 0 or 1
        trials = subject_data[field][0]               # shape [num_trials, time, num_channels]
        for trial in trials:
            eegData      = trial                     # [time, num_channels]
            trial_length = eegData.shape[0]
            plv          = plvMatrix(eegData)        # your new full-trial PLV
            temp_data    = temporalData(plv, label, trial_length)
            if temp_data.src.numel() == 0 or temp_data.dst.numel() == 0:
                print(f"Skipping empty graph for label {field}")
                continue
            dataset.append(temp_data)

    return dataset


# %% load data

data_dir = os.getcwd() 
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_temporal_data = []

subject_data = {}
subject_results = {} # two empty dictionaries

import torch.nn as nn
#from torch_geometric.nn.models import TGNMemory
#from torch_geometric.nn import IdentityMessage, LastAggregator, TimeEncoder
        
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    )

from torch.nn import GRUCell
from torch_geometric.nn.models.tgn import TimeEncoder


# loop over each subject
for subject_number in tqdm(subject_numbers, desc="Processing Subjects"):
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data = mat_contents[f'Subject{subject_number}'][:, :]  # shape [1, 160]

    # aggregate TemporalData for this subject of all trials
    subject_temporal_data = aggregate_temporalData(subject_data) 
    # convert to TemporalData list: event(src, dst, t, msg, label)
    # subject_temporal_data is a list of TemporalData objects
    
    # append to total dataset if we want a cross-subject model
    #all_temporal_data.extend(subject_temporal_dataset)

    # k-fold cross validation, 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    highest_test_accuracies = []
    

    for fold, (train_idx, test_idx) in enumerate(kf.split(subject_temporal_data)):
        train_data = [subject_temporal_data[i] for i in train_idx]
        test_data = [subject_temporal_data[i] for i in test_idx]

        # create TemporalDataLoaders
        torch.manual_seed(12345)
        
        batch_size = 32
        
        #train_loader = TemporalDataLoader(train_data, batch_size=32) # batch_size=32 !!!
        #test_loader = TemporalDataLoader(test_data, batch_size=32) # batch_size=32 !!!
        train_loader = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
        test_loader = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
        

        
        batch_list = next(iter(train_loader))
        #data_sample = collate_temporal_batch(batch_list)
        data_sample = manual_collate_temporal_batch(batch_list)


        num_nodes = torch.cat([data_sample.src, data_sample.dst]).max().item() + 1
        # calculate the total number of nodes (electrodes) in the current batched temporal graph data_sample
        # -> this num_nodes is needed to initialize the memory module of the TGN. Each node will get its own memory vector.
        
        raw_msg_dim = data_sample.msg.size(-1) # how many features per message
        # get the feature dimension of the messages passed between nodes (i.e. the size of the message vector)
        # -> pass raw_msg_dim to TGN, telling it what the input dimension of messages is 
        # -> required for the first layer of the message processing network inside TGN


        # define the TGN model 
        class CustomTGNMemory(nn.Module): # 1. define a memory module for tgn
            
          def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim,
                       message_module, aggregator_module): # initialize the memory module with required hyperparameters
              # no.of nodes to save memory for, 19 electrodes;
              # message dimensionality before being combined with memory and time
              # how big each node's memory vector is;
              # how many dimensions for time encoding;
              # message module: combine memory, raw message and time encodings into a final message;
              # aggregator module: aggregate all messages to the same node
              
           super().__init__()
           self.num_nodes = num_nodes
           self.raw_msg_dim = raw_msg_dim
           self.memory_dim = memory_dim
           self.time_dim = time_dim

           self.msg_module = message_module
           self.aggr_module = aggregator_module

           self.time_enc = TimeEncoder(time_dim) # time encoder maps a scalar time difference into a vector
           self.gru = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)# update memory using the final aggregated message 

           self.register_buffer('memory', torch.empty(num_nodes, memory_dim)) 
           # create a non-trainable tensor that will store each node’s memory
           self.register_buffer('last_update', torch.empty(num_nodes)) # keep track of the last timestamp each node was updated
           self.reset_memory() # call a reset function to zero out memory and timestamp
           # 在模型刚创建时初始化 memory 为零

          def reset_memory(self):
           self.memory.fill_(0)
           self.last_update.fill_(0)
           
          def forward(self, n_id: torch.Tensor):
              """Fetch memory and last update time for given node indices"""
              return self.memory[n_id], self.last_update[n_id]
          # memory[n_id]: node-level memory vectors
          # last_update[n_id]: timestamps of last updates
          
          def detach(self):
              self.memory.detach_()
              
          def update_state(self, src, dst, t, raw_msg):
              """Update memory for dst nodes using (src, dst, time, message)"""
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
                self.conv = TransformerConv(
                                            in_channels=in_channels,  # node features (likely from memory)
                                            out_channels=out_channels // 2,
                                            heads=2, # 2 heads → final dim = (out_channels // 2) * 2 = out_channels
                                            dropout=0.1,
                                            edge_dim=edge_dim    # temporal + message edge features
                )
                
            def forward(self, x, last_update, edge_index, t, msg): 
                
                # x: node features; t: timestamps
                rel_t = last_update[edge_index[0]] - t
                rel_t_enc = self.time_enc(rel_t.to(msg.dtype))
                
                # project 1D message to msg_dim
                msg_proj = self.msg_mlp(msg)  # [num_edges, msg_dim]
        
                edge_attr = torch.cat([rel_t_enc, msg_proj], dim=-1)
                return self.conv(x, edge_index, edge_attr)
            
            
            
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
                #self.classifier = nn.Linear(hidden_channels, out_channels)
                self.classifier = nn.Linear(hidden_channels, out_channels)

                

            def forward(self, edge_index, t, msg, n_id, batch_idx):
                x, last_update = self.memory(n_id)
                x = self.embedding(x, last_update, edge_index, t, msg)
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

            
        hidden_channels = 32   # depend on the feature size
        out_channels = 2       # 'L' or 'R' binary classification
        # memory_dim = time_dim = embedding_dim = 100
        raw_msg_dim = 1        # due to '.unsqueeze(-1)'
        memory_dim = hidden_channels
        time_dim = hidden_channels
        
        #num_nodes = torch.cat([data_sample.src, data_sample.dst]).max().item() + 1
        
        # define the model                   
        model = TGNClassifier(num_nodes=num_nodes, #19 * batch_size,
                              #num_nodes=num_nodes,
                              raw_msg_dim=raw_msg_dim,
                              memory_dim=memory_dim,
                              time_dim=time_dim,

                              msg_module=CustomMessageModule(raw_msg_dim=raw_msg_dim,
                                                             memory_dim=memory_dim,time_dim=time_dim),
            
                              aggr_module=LastAggregator(),
                              in_channels=memory_dim,
                              hidden_channels=hidden_channels,
                              out_channels=out_channels
        ).to(device)

        #model.memory.reset_memory()
        
        # optimization and cross-entropy as loss function 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        #optimizer = torch.optim.Adam( 
            #set(memory.parameters()) | set(gnn.parameters())
            #| set(classifier.parameters()), lr=0.0001)
    
        criterion = torch.nn.CrossEntropyLoss()

            
        
        # % train and test the model
        def train():
            model.train()
            for batch_list in train_loader:
                
                model.memory.reset_memory()

                #batch = collate_temporal_batch(batch_list).to(device)
                batch = manual_collate_temporal_batch(batch_list).to(device)
                #batch = batch.to(device)
                
                edge_index = torch.stack([batch.src, batch.dst], dim=0)
                n_id = torch.cat([batch.src, batch.dst]).unique()
                batch_idx = batch.batch  # each edge belongs to which graph
                
                model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                out = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                loss = criterion(out, batch.y)  # out: [32, 2], y: [32]
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
             
             
        def test(loader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_list in loader:
                    
                    model.memory.reset_memory()

                    #batch = collate_temporal_batch(batch_list).to(device)
                    batch = manual_collate_temporal_batch(batch_list).to(device)
                    #batch = batch.to(device)
                    
                    edge_index = torch.stack([batch.src, batch.dst], dim=0)
                    n_id = torch.cat([batch.src, batch.dst]).unique()
                    batch_idx = batch.batch
            
                    out = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
                    pred = out.argmax(dim=1)
            
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            return correct / total if total > 0 else 0.0
        
        
        optimal = [0, 0, 0] # store the best training/testing c.a.
        for epoch in tqdm(range(1, 50), 
                          desc=f"Training Epochs for Subject {subject_number} Fold {fold+1}"):
            
            train() 
            # for each epoch, 
            # run one full training pass over all batches using the train() function defined earlier
            
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            av_acc = np.mean([train_acc, test_acc])
            if test_acc > optimal[2]: 
            # if this epoch’s test accuracy is better than the previous best test accuracy
                optimal = [av_acc, train_acc, test_acc]
                
        highest_test_accuracies.append(optimal[2]) # -> 10 test c.a. 
        # 在这 50 或 500 个 epoch 中，
        # 我们会比较每一次 epoch 得到的测试准确率，选出最高的那个测试准确率，作为该 fold 的最佳测试准确率.
        # select the highest test c.a. over 500 epoch 
        # execute once per fold, so it store the best c.a. for each fold
        
        ## end the cross validation loop ##
        
        
    
    meanhigh = np.mean(highest_test_accuracies)
    maxhigh = np.max(highest_test_accuracies)
    minhigh = np.min(highest_test_accuracies)
    
    # Save the results in the dictionary
    subject_results[subject_number] = {
        'mean': meanhigh,
        'max': maxhigh,
        'min': minhigh
    }
    
    # Print results for the current subject
    print(f'S{subject_number}: Mean: {meanhigh:.4f}, Max: {maxhigh:.4f}, Min: {minhigh:.4f}')
    
    ## end the loop over each subject ##
    
# %% c.a. collection

# Optionally, save the results to a file or print all results at the end
print("\nSummary of Results for All Subjects:")
for subject_number, results in subject_results.items():
    print(f'S{subject_number}: Mean: {results["mean"]:.4f}, Max: {results["max"]:.4f}, Min: {results["min"]:.4f}')
    
import json 

# Save the results to a JSON file
with open('TGN_ALSresults 11.json', 'w') as json_file:
    json.dump(subject_results, json_file, indent=4)

 
     