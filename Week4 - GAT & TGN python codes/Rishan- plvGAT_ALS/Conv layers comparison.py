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
    transformer_results = json.load(file)

with open('TGN_ALSresults(RGATConv).json', 'r') as file:
    gat_results = json.load(file)

# === Ensure common, sorted subject IDs ===
subject_ids = sorted((transformer_results.keys()) & (gat_results.keys()), key=lambda x: int(x))
x = np.arange(len(subject_ids))  # X positions

# === Extract TransformerConv metrics ===
trans_means = [transformer_results[sub]['mean'] * 100 for sub in subject_ids]
trans_mins = [transformer_results[sub]['min'] * 100 for sub in subject_ids]
trans_maxs = [transformer_results[sub]['max'] * 100 for sub in subject_ids]
trans_lower = [mean - min_ for mean, min_ in zip(trans_means, trans_mins)]
trans_upper = [max_ - mean for mean, max_ in zip(trans_means, trans_maxs)]

# === Extract GATv2Conv metrics ===
gat_means = [gat_results[sub]['mean'] * 100 for sub in subject_ids]
gat_mins = [gat_results[sub]['min'] * 100 for sub in subject_ids]
gat_maxs = [gat_results[sub]['max'] * 100 for sub in subject_ids]
gat_lower = [mean - min_ for mean, min_ in zip(gat_means, gat_mins)]
gat_upper = [max_ - mean for mean, max_ in zip(gat_means, gat_maxs)]

# === Plot ===
plt.figure(figsize=(19, 12))

# TransformerConv
plt.errorbar(x, trans_means, yerr=[trans_lower, trans_upper], fmt='o',
             color='tab:blue', markersize=20, capsize=18, capthick=4, elinewidth=4, label='TransformerConv')

# GMMConv
plt.errorbar(x, gat_means, yerr=[gat_lower, gat_upper], fmt='s',
             color='tab:orange', markersize=20, capsize=18, capthick=4, elinewidth=4, label='RGATConv')

# === Format ===
plt.title('TGN Model Performance Comparison: TransformerConv vs RGATConv', fontweight='bold', fontsize=30)
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
plt.savefig("TransformerConv vs RGATConv 1.svg", format='svg')
plt.show()