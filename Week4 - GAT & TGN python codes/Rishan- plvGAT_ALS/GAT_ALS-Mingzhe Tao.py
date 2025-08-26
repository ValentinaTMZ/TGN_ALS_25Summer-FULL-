#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 16:53:08 2025

@author: taomingzhe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.



https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import os
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from scipy.integrate import simps
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch import nn
from tqdm import tqdm
from torch_geometric.data import Data
 
#% Functions
def plvfcn(eegData):
    numElectrodes = eegData.shape[1] # No. of electrons = 22
    numTimeSteps = eegData.shape[0] # No. of time steps = 1288 在一个trial中EEG signals被采样的总次数
    plvMatrix = np.zeros((numElectrodes, numElectrodes)) # np.zeros((2, 8)): two rows, eight 0 elements for each row
    for electrode1 in range(numElectrodes): # electrode1 = 0, 1, ..., 21
        for electrode2 in range(electrode1 + 1, numElectrodes): # range(start, stop(required), step)
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1])) # eegData[:, electrode1] selects all the time points from one EEG channel
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2])) # to get the instantaneous phase of each signal
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps) # take average for the whole trial 
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def compute_plv(subject_data):
    idx = ['L', 'R']
    
    # subject_data['L'][0,1] extract 'Left' field and (first column) 2nd trial/row of some subject
    numElectrodes = subject_data['L'][0,1].shape[1] 
    
    # create a dictionary dict = {key1:value1, key2:value2, ... }
    # an effective data structure to organize and manage PLV matrices across multiple trials and L&R fields
    # subject_data.shape[1] refers to No. of trials, i.e. 160
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx} # key1=L, key2=R
    
    # to compute a PLV matrix for every trial within each condition (L or R)
    # enumerate() is a Python tool to loop through an iterable—like a list, tuple, or string—and have access to both the index and the element itself.
    for i, field in enumerate(idx): # for each field 'L' and 'R'
        for j in range(subject_data.shape[1]): # for each trial
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x) 
            
    l, r = plv['L'], plv['R'] # 按照field对plvMatrix分类
    yl = np.zeros((subject_data.shape[1], 1))
    yr = np.ones((subject_data.shape[1], 1)) 
    
    # np.concatenate() joins a sequence of arrays along an existing axis, 同时保持原本的图结构 for GAT to learn
    # axis=2 在第三個維度(深度方向)拼接, 變成320張圖
    img = np.concatenate((l, r), axis=2) 
    
    # axis=0：在「上下方向」拼接（加新的 row）
    # axis=1：在「左右方向」拼接（加新的 column）
    y = np.concatenate((yl, yr), axis=0) # y：未來要丟給分類模型的標籤, i.e. answers for each plvMatrix for the model to learn
    
    # convert 'array' into 'tensor' for Pytorch (deep learning, model training...) to calculate loss, autograd and fine-tuning weights...
    y = torch.tensor(y, dtype=torch.long) # torch.long: 64-bit integer, required as the classification labels for PyTorch 
    return img, y


def create_graphs(plv, threshold):
    # plv is now a 3-d PLV tensor with shape [No. of electrons, No. of electrons, No. of trials], [22, 22, 320]
    """ https://networkx.org/documentation/stable/tutorial.html#tutorial  """
    
    graphs = [] # generate an empty list, put graphs of each trial in afterwards
    for i in range(plv.shape[2]): # plv.shape[2] is the total No. of trials, 320
        G = nx.Graph() # use Network X to build an empty graph G
        
        # plv.shape[0] is the total No. of channels, 22
        G.add_nodes_from(range(plv.shape[0])) # No. of channels = No. of nodes
        
        for u in range(plv.shape[0]): # loop over all possible source electrodes
            for v in range(plv.shape[0]): # loop over all possible destination electrodes
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs


def aggregate_eeg_data(S1,band): #%% This is to get the feature vector
    """
    Aggregate EEG data for each class.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels, 
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).

    Returns:
        l (ndarray): Aggregated EEG data for class 'L'.
        r (ndarray): Aggregated EEG data for class 'R'.
    """
    idx = ['L', 'R']
    numElectrodes = S1['L'][0,1].shape[1];
    max_sizes = {field: 0 for field in idx} # output: max_sizes = {'L': 0,
                                            #                      'R': 0}

    # Find the maximum size of EEG data for each class
    for field in idx:
        for i in range(S1[field].shape[1]): # S1['L'].shape[1]: No. of trials for class 'L' of Subject 1
        
            # S1[field][0, i].shape[0]: timesteps of the ith trial for class 'L'/'R' of Subject 1
            max_sizes[field] = max(max_sizes[field], S1[field][0, i].shape[0]) 

    # Initialize arrays to store aggregated EEG data
    l = np.zeros((max_sizes['L'], numElectrodes, S1['L'].shape[1]))
    r = np.zeros((max_sizes['R'], numElectrodes, S1['R'].shape[1]))

    # Loop through each sample
    for i in range(S1['L'].shape[1]): # go through all trials for class 'L' of Subject 1
    
        for j, field in enumerate(idx):
            x = S1[field][0, i]  # EEG data for the current sample, 2-d array with shape (No.of timesteps, No.of electrons)
            # e.g. (1280, 22)
            
            # Resize x to match the maximum size
            resized_x = np.zeros((max_sizes[field], 22)) # e.g. (1296, 22)
            
            # 把 x 这张图的内容，贴到 resized_x 的前 1280 行里面
            resized_x[:x.shape[0], :] = x # x.shape[0] is time steps
            
            # Add the resized EEG data to the respective array
            if field == 'L':
                l[:, :, i] += resized_x # a += b is the same as a = a + b
            elif field == 'R':
                r[:, :, i] += resized_x

    l = l[..., np.newaxis] # add a new "frequency band" axis at the end, l.shape = (1296, 22, 160, 1)
    
    l = np.copy(l) * np.ones(len(band)-1) 
    # e.g. len(band) = 6 (there are 6 frequency boundaries, e.g., for 5 bands like delta, theta, alpha...)
    # then l.shape becomes (1296, 22, 160, 5) 

    r = r[..., np.newaxis]
    r = np.copy(r) * np.ones(len(band)-1)
    
    return l, r

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data


def bandpass1(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
# the same as: def bandpass1(data, edges, sample_rate, poles=5)
# data: a 2-d array with shape (timesteps, electrons)
# edges: a list of float, e.g. [8., 12.]; 只保留 8 到 12 Hz (alpha 波段) 的信號; float: 带小数点的数字
# sample rate: usually 250 Hz
# pole: Butterworth 滤波器的阶数（默认是 5），阶数越高，滤波越陡峭; 能更精準地把不想要的頻率給切掉

    # create a digital bandpass filter using a Butterworth filter design
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos') 
    # sos is a matrix (array of arrays) that contains all the filter coefficients needed to actually filter the signal
    
    # apply the filter above to the EEG data.
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data

def bandpower(data,low,high):
# data: EEG time-series signal (1D NumPy array)

    fs = 256
    
    # Define window length (2s) for Welch’s method
    # Welch’s method: 用来估计一个信号的功率谱密度（Power Spectral Density, PSD），它比普通的傅里叶变换更稳定，抗噪能力强
    # 1. 把长信号分成多个小窗口（比如每 2 秒一段），每段都有可能重叠
    # 2. 对每个窗口做傅里叶变换（FFT），得到频域信息
    # 3. 把所有窗口的频谱结果平均，得到一个更平滑、更稳定的功率谱
    win = 2* fs # 每一个“分析窗口”包含 2 秒钟的数据，也就是 512 个采样点
    freqs, psd = signal.welch(data, fs, nperseg=win) # 每一个频率成分上，信号中有多少能量（power）
    # freqs = [0.0, 0.25, 0.5, ..., 127.75, 128.0] up to fs/2, the max freq that can be captured is fs/2 Hz
    
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    # e.g. low = 8, high = 12
    # only the PSD values corresponding to 8–12 Hz will be True, and selected for calculating power
    
    # # Plot the power spectral density and fill the delta area
    # plt.figure(figsize=(7, 4))
    # plt.plot(freqs, psd, lw=2, color='k')
    # plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 40])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    
    # Frequency resolution: how far apart each frequency point is in the freqs array returned by signal.welch()
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    
    # Compute the absolute power by approximating the area under the curve
    power = simps(psd[idx_delta], dx=freq_res)
    
    return power

def bandpowercalc(l,band,fs):
# l: a 4D NumPy array → EEG data structured as [time_steps, electrodes, trials, bands]
# band: list of frequency band edges, like [8, 12, 30] for alpha and beta
# fs: sampling rate (e.g., 256 Hz)

    x = np.zeros([l.shape[0],l.shape[3],l.shape[2]])
    for i in range(l.shape[0]): #node
        for j in range(l.shape[2]): #sample
            for k in range(0,l.shape[3]): #band
            # e.g. band = [4,8,12,30,], then k = 0, 1, 2, 3
            
                data = l[i,:,j,k] # across all channels
                low = band[k]
                high = band[k+1]
                x[i,k,j] = bandpower(data,low,high)

    return x

# Preparing Data
data_dir = os.getcwd() #r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39] # a list of subject IDs
subject_data = {}
subject_results = {} # two empty dictionaries

# Efficiently load data for all subjects with a progress bar
for subject_number in tqdm(subject_numbers, desc="Processing Subjects"): # instantly make your loops show a smart progress meter 进度表
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat') # join data direction and .mat file name to form a complete file path
    mat_contents = sio.loadmat(mat_fname) # loads the contents of the .mat file into a Python dictionary
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:, :] # extract all rows and columns of the loaded subject data
    # For each subject in [1, 2, 5, 9, …]:
	# 1. build the .mat file path like S1.mat
	# 2. load it using scipy.io.loadmat()
	# 3. extract the 'Subject1', 'Subject2', … data from the file
	# 4. save that into subject_data['S1'], ['S2'], etc.
	# 5. temporarily store it as S1 to work with it directly


    # Compute PLV and Graphs
    plv, y = compute_plv(S1)
    threshold = 0 #0.3
    graphs = create_graphs(plv, threshold)
    numElectrodes = S1['L'][0, 1].shape[1]

    # Preallocate adjacency matrix
    adj = np.zeros((numElectrodes, numElectrodes, len(graphs))) # len(graphs) is the same as No. of trials; 为每个 trial 分配一个 adjacency matrix 的存储空间
    for i, G in enumerate(graphs): # 遍历每一个图 G，同时获得它的索引 i（第几个 trial）
        adj[:, :, i] = nx.to_numpy_array(G) # nx.to_numpy_array() converts each graph G into one adjacent matric in the form of numpy array

    # Vectorized Edge Indices Construction
    edge_indices = []
    for i in range(adj.shape[2]):
        source_nodes, target_nodes = np.where(adj[:, :, i] >= threshold) # np.where() returns the indices of entries ≥ threshold
        # for each trial, find those >= threshold matrix elements as effective edges, whose row indices = source node and column indices = target node
        
        # Convert to numpy array and then to tensor
        edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
        # edge_index has shape [2, num_edges]: [[source_1, source_2, ..., source_n],
        #                                       [target_1, target_2, ..., target_n]]
        
        # Append to list
        edge_indices.append(edge_index)
    
    # Stack edge indices efficiently
    edge_indices = torch.stack(edge_indices, dim=-1) 
    # stack all the individual edge_index tensors (from each trial) into a single 3D tensor along a new last dimension (dim = -1)
    # allow us to handle edge indices for all trials in one tensor

    band = list(range(8, 41, 4)) # band = [8, 12, 16, 20, 24, 28, 32, 36, 40]
    l, r = aggregate_eeg_data(S1, band) # shape: (max_time_length, num_electrodes, num_trials, num_bands-1)
    l, r = np.transpose(l, [1, 0, 2, 3]), np.transpose(r, [1, 0, 2, 3]) # reorder the axes (dimensions) of the EEG arrays into 
    # (num_electrodes, max_time_length, num_trials, num_bands-1)
    
    fs = 256

    # Efficient bandpass filtering
    for i in range(l.shape[3]): # for each band
        bp = [band[i], band[i + 1]]
        for j in range(l.shape[2]): # for each trial
            l[:, :, j, i] = bandpass(l[:, :, j, i], bp, sample_rate=fs)
            r[:, :, j, i] = bandpass(r[:, :, j, i], bp, sample_rate=fs)
        # The filtered output is:
        # still the same shape: (num_electrodes, time_length)
        # but noise and unrelated frequencies are removed, keeping only EEG activity in specifc band for each trial

    l = bandpowercalc(l, band, fs) # bandpowercalc() returns power values in a new arrary of shape (No.of electrons, No. of bands, No. of trials)
    r = bandpowercalc(r, band, fs)
    x = np.concatenate([l, r], axis=2)
    # Assuming l is (22, 8, 160) and r is (22, 8, 160):
	# axis=2 → concatenate along the trial dimension
    # x becomes: x.shape = (22, 8, 320)
    
    x = torch.tensor(x, dtype=torch.float32)

    # a list comprehension that creates one Data object per trial
    # data objects are the fundamental units of data that machine learning models operate on
    data_list = [Data(x=x[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]) for i in range(np.size(adj, 2))] 
    # adj.shape = (22, 22, 320) so np.size(adj, 2) = No. of trials = 320
    # x[:, :, i] → shape: [num_nodes, num_features] = EEG bandpower features per electrode
    # edge_index[:, :, i] → edges of the graph for trial i
    # y[i, 0] → class label (0 for L, 1 for R)
    # so each data object holds: 
    # Data(
    #      x = tensor of shape [22 nodes, 8 features],  # e.g., 8 bands
    #      edge_index = tensor of shape [2, num_edges], # edge connections
    #      y = scalar label (0 or 1)                    # left or right    )


    datal = data_list[:len(data_list) // 2]
    datar = data_list[len(data_list) // 2:]
    # split data_list into two halves:
	# datal → first half of trials (presumably the “L” class)
	# datar → second half (presumably the “R” class)
    
    data_list = [datal[i] for i in range(len(datal))] + [datar[i] for i in range(len(datar))] # output [l1, l2, ..., r1, r2, ...]

    # KFold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42) # 进行交叉验证前对data重新打乱，而不会在交叉验证进行间再次打乱
    # split the data randomly into 10 part, each part acts as the test set in turn, so output 10 models
    highest_test_accuracies = []

    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        train = [data_list[i] for i in train_idx]
        test = [data_list[i] for i in test_idx]

        torch.manual_seed(12345)
        train_loader = DataLoader(train, batch_size=32, shuffle=False) # 每个批次包含32个样本数量
        test_loader = DataLoader(test, batch_size=32, shuffle=False) # DataLoader is one class of PyTorch

        # define a neutral network model class, based on GAT
        class GAT(nn.Module): # define a class called GAT
            def __init__(self, hidden_channels, heads): # 构造函数
                super(GAT, self).__init__()
                # 接收两个超参数：
                # hidden_channels：每层隐藏单元的维度
                # heads：多头注意力机制的头数

                
                # Define GAT convolution layers
                self.conv1 = GATv2Conv(8, hidden_channels, heads=heads, concat=True)  # num node features
                self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                # concat=True：把不同 head 的输出拼接（concatenate）而不是求平均
                
                # Define GraphNorm layers 让图神经网络的训练更加稳定和高效，让模型更好地学习图结构中的重要模式
                self.gn1 = GraphNorm(hidden_channels * heads)
                self.gn2 = GraphNorm(hidden_channels * heads)
                self.gn3 = GraphNorm(hidden_channels * heads)
                
                # Define the final linear layer
                self.lin = nn.Linear(hidden_channels * heads, 2)  # num of classes
        
            def forward(self, x, edge_index, batch): # 定义前向传播逻辑
            # x: 节点特征矩阵 [num_nodes, 8]
            # edge_index: 图的边 [2, num_edges]
            # batch: 用于 batch-wise graph pooling 的索引张量（标明每个节点属于哪个图）

                # 同样的流程重复三次（卷积 → 激活 → 归一化）
                # Apply first GAT layer and normalization
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.gn1(x, batch)  # Apply GraphNorm
        
                # Apply second GAT layer and normalization
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = self.gn2(x, batch)  # Apply GraphNorm
        
                # Apply third GAT layer and normalization
                x = self.conv3(x, edge_index)
                x = self.gn3(x, batch)  # Apply GraphNorm
        
                # Global pooling
                x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
                # Apply dropout and final classifier
                x = F.dropout(x, p=0.50, training=self.training)
                x = self.lin(x)
        
                return x 
                # x：每个图的 logits（未归一化分类得分），用于后续交叉熵损失计算


        model = GAT(hidden_channels=22,heads=3) # initialize the defined GAT model
        # hidden_channels=22:
        # this means the size of the hidden feature vector per node is 22
        # often corresponds to the number of EEG electrodes (22 channels)
        # No. of heads = 3, more heads → capture more diverse patterns of node importance
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # used to update model parameters during training
        criterion = torch.nn.CrossEntropyLoss()

        def train(): # called once per epoch to train the model on the entire training set
            model.train() # put the model into training mode.
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                # This line passes the data into the GAT model:
                # data.x: Input node features (e.g., band power features of EEG)
	            # data.edge_index: Graph structure (which nodes are connected)
                # data.batch: Tells the model which nodes belong to which graph when training multiple graphs at once
                # Returns: Model’s predicted logits (before softmax), one per graph

                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step() # Apply the gradients to update model parameters
                optimizer.zero_grad()

        def test(loader): # define a function called test that takes in a PyG DataLoader (either train_loader or test_loader) as input
            model.eval() # put the model into the evaluation mode
            correct = 0 # initialize a counter to store the total number of correct predictions
            for data in loader:
                out = model(data.x, data.edge_index, data.batch) # pass the batch through the model to get predictions (logits)
                
                pred = out.argmax(dim=1) 
                # find the class with the highest score for each graph in the batch, giving the predicted class labels
                
                correct += int((pred == data.y).sum()) # (pred == data.y) is a Boolean tensor — True for correct predictions
            return correct / len(loader.dataset)


        optimal = [0, 0, 0] # store the best training/testing c.a.
        for epoch in tqdm(range(1, 500), desc=f"Training Epochs for Subject {subject_number} Fold {fold+1}"):
            # the main training loop, running for 499 epochs (from 1 to 499)
            
            train() # for each epoch, run one full training pass over all batches using the train() function defined earlier
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            av_acc = np.mean([train_acc, test_acc])
            if test_acc > optimal[2]: # if this epoch’s test accuracy is better than the previous best test accuracy
                optimal = [av_acc, train_acc, test_acc]

        highest_test_accuracies.append(optimal[2]) # execute once per fold, so it store the best c.a. for each fold

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

# Optionally, save the results to a file or print all results at the end
print("\nSummary of Results for All Subjects:")
for subject_number, results in subject_results.items():
    print(f'S{subject_number}: Mean: {results["mean"]:.4f}, Max: {results["max"]:.4f}, Min: {results["min"]:.4f}')
    
import json 

# Save the results to a JSON file
with open('GAT_ALSresults.json', 'w') as json_file:
    json.dump(subject_results, json_file, indent=4)
    