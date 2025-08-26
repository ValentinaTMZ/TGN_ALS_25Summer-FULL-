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

with open('TGN_ALS_3Layers(per sub).json', 'r') as file:
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

# calculate s.d. for each subject
trans_subject_stds = [np.std([transformer_results[sub]['mean'], 
                              transformer_results[sub]['min'], 
                              transformer_results[sub]['max']]) * 100 
                      for sub in subject_ids]

# calculate s.d. for this model across all subjects
trans_across_subjects_std = np.std(trans_means)


# === Extract GATv2Conv metrics ===
gat_means = [gat_results[sub]['mean'] * 100 for sub in subject_ids]
gat_mins = [gat_results[sub]['min'] * 100 for sub in subject_ids]
gat_maxs = [gat_results[sub]['max'] * 100 for sub in subject_ids]
gat_lower = [mean - min_ for mean, min_ in zip(gat_means, gat_mins)]
gat_upper = [max_ - mean for mean, max_ in zip(gat_means, gat_maxs)]

# calculate s.d. for each subject
gat_subject_stds = [np.std([gat_results[sub]['mean'], 
                            gat_results[sub]['min'], 
                            gat_results[sub]['max']]) * 100 
                    for sub in subject_ids]

# calculate s.d. for this model across all subjects
gat_across_subjects_std = np.std(gat_means) 


# === Plot ===
plt.figure(figsize=(19, 12))

# TransformerConv
plt.errorbar(x, trans_means, yerr=[trans_lower, trans_upper], fmt='o',
             color='tab:blue', markersize=20, capsize=18, capthick=4, elinewidth=4, label='1-sec window(baseline)')

# GMMConv
plt.errorbar(x, gat_means, yerr=[gat_lower, gat_upper], fmt='s',
             color='tab:orange', markersize=20, capsize=18, capthick=4, elinewidth=4, label='3*Conv Layers')

# === Format ===
plt.suptitle('TGN Model Performance Comparison', fontweight='bold', fontsize=38)
plt.title('1-Sec Window vs 3*Conv Layers \n(no overlaps, no cross-trial windows)', fontweight='bold', fontsize=33)
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

# === Format SDs for annotation ===
trans_subject_std_avg = np.mean(trans_subject_stds)
gat_subject_std_avg = np.mean(gat_subject_stds)

trans_summary = (
    f"1-sec window (baseline):\n"
    f"  • Avg s.d. (per subject): {trans_subject_std_avg:.2f}%\n"
    f"  • s.d. of means: {trans_across_subjects_std:.2f}%"
)
gat_summary = (
    f"3*conv layers:\n"
    f"  • Avg s.d. (per subject): {np.mean(gat_subject_stds):.2f}%\n"
    f"  • s.d. of means: {gat_across_subjects_std:.2f}%"
)
annotation_text = trans_summary + "\n\n" + gat_summary

# === Add annotation box to plot ===
plt.gca().text(
    0.68, 0.04, annotation_text,
    transform=plt.gca().transAxes,
    fontsize=21,
    verticalalignment='bottom',
    horizontalalignment='left',
    fontweight='bold',
    bbox=dict(facecolor='whitesmoke', edgecolor='gray', boxstyle='round,pad=0.6')
)


plt.savefig("3*Conv Layers 11.svg", format='svg')
plt.show()