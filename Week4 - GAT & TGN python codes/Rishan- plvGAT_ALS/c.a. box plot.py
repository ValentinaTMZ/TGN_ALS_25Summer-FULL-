#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:13:28 2025

@author: taomingzhe
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# === Load JSON files ===
with open('TGN_ALSresults(50 epochs).json', 'r') as file:
    ca_results = json.load(file)


# === Ensure common, sorted subject IDs ===
subject_ids = sorted(ca_results.keys(), key=lambda x: int(x))
x = np.arange(len(subject_ids))  # X positions

# === Extract TransformerConv metrics ===
ca_means = [ca_results[sub]['mean'] * 100 for sub in subject_ids]
ca_mins = [ca_results[sub]['min'] * 100 for sub in subject_ids]
ca_maxs = [ca_results[sub]['max'] * 100 for sub in subject_ids]
ca_lower = [mean - min_ for mean, min_ in zip(ca_means, ca_mins)]
ca_upper = [max_ - mean for mean, max_ in zip(ca_means, ca_maxs)]

# === Plot ===
plt.figure(figsize=(16, 10))

# TransformerConv
plt.errorbar(x, ca_means, yerr=[ca_lower, ca_upper], fmt='o',
             color='skyblue', markersize=20, capsize=18, capthick=4, elinewidth=4)

# === Format ===
plt.title('TGN Model Performance, 50 Epochs', fontweight='bold', fontsize=30)
plt.xlabel('ALS Subject ID', fontweight='bold', fontsize=25)
plt.ylabel('Classification Accuracy (%)', fontweight='bold', fontsize=25)
plt.xticks(x, subject_ids, fontweight='bold', fontsize=25)
plt.yticks(fontweight='bold', fontsize=25)
plt.ylim(0, 100)
plt.grid(axis='y')

# Horizontal threshold
plt.axhline(y=70, color='red', linestyle='--', linewidth=3.5, label='Threshold (70%)')

# Axis borders and legend
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.legend(loc='upper right', prop={'weight': 'bold', 'size': 23})

plt.tight_layout()
plt.savefig("c.a.(50 epochs)1.svg", format='svg')
plt.show()