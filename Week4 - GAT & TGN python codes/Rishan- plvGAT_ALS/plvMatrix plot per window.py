#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 19:47:59 2025

@author: taomingzhe
"""

import os
import scipy.io as sio
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# ==== Load subject 1 data ====
mat_contents = sio.loadmat('S1.mat')  # Make sure S1.mat is in the current directory
subject_struct = mat_contents['Subject1'][:, :]  # get MATLAB struct inside (1,1) array

# ==== Extract 'L' trials ====
L_trials = subject_struct['L'][0]  # This is a list of trials; each is [time, 64]

# ==== Concatenate all trials along time axis for first 19 channels ====
long_eeg = np.concatenate([trial[:, :19] for trial in L_trials], axis=0)  # shape: [long_time, 19]

# ==== Get first window ====
#window_size = 256
#eeg_window = long_eeg[:window_size, :]  # [256, 19]
# Take window 1 (second 1-second window)

window_size = 256
start = 0 * window_size  # 256
end = 1 * window_size    # 512
eeg_window = long_eeg[start:end, :]  # [256, 19]



# ==== Compute PLV matrix ====
num_electrodes = eeg_window.shape[1]
plv_matrix = np.zeros((num_electrodes, num_electrodes))

for i in range(num_electrodes):
    for j in range(i + 1, num_electrodes):
        phase1 = np.angle(hilbert(eeg_window[:, i]))
        phase2 = np.angle(hilbert(eeg_window[:, j]))
        phase_diff = phase2 - phase1
        plv = np.abs(np.sum(np.exp(1j * phase_diff))) / window_size
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv  # symmetric

# ==== Print or plot result ====
np.set_printoptions(precision=3, suppress=True)
print("PLV Matrix for Subject 1, Label 'L', Window 0" )
print(plv_matrix)

plt.imshow(plv_matrix, cmap='viridis')

# Add colorbar with bold tick labels
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10, width=1.5)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Title and axis labels in bold
plt.title("PLV Matrix - Subject 1, Label 'L', Window 0", fontweight='bold', fontsize=12)
plt.xlabel("Electrode", fontweight='bold', fontsize=11)
plt.ylabel("Electrode", fontweight='bold', fontsize=11)

# Tick labels in bold
electrode_labels = [str(i+1) for i in range(num_electrodes)]
plt.xticks(ticks=range(num_electrodes), labels=electrode_labels, fontweight='bold', fontsize=10)
plt.yticks(ticks=range(num_electrodes), labels=electrode_labels, fontweight='bold', fontsize=10)

# Make axes spines thicker
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)


plt.tight_layout()


plt.savefig("plv_S1_L_W0.svg", format='svg')
plt.show()
print("✅ Figure saved as 'plv_S1_L_W0.svg'")

