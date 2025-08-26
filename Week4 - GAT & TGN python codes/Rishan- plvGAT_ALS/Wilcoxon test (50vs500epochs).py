#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:13:19 2025

@author: taomingzhe
"""

import json
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# %% Load JSON
with open('TGN_ALSresults(50 epochs).json', 'r') as file:
    subject_results_50 = json.load(file)

with open('TGN_ALSresults(500 epochs).json', 'r') as file:
    subject_results_500 = json.load(file)

subject_ids = list(subject_results_50.keys())

# %% Apply Wilcoxon test

#means_50 = [subject_results_50[sub]['mean'] * 100 for sub in subject_ids]
#means_500 = [subject_results_500[sub]['mean'] * 100 for sub in subject_ids]

means_50 = [subject_results_50[i]['mean'] for i in subject_ids]
means_500 = [subject_results_500[i]['mean'] for i in subject_ids]
    
res = wilcoxon(means_50, means_500)
print(f"Wilcoxon test p-value = {res.pvalue:.4f}")

res_ttest = ttest_rel(means_500, means_50)
print(f"Paired t-test p-value = {res_ttest.pvalue:.4f}")

# %% Generate a table 
#table_pvalues = table_values = list(zip(subject_ids, p_values))
#headers = ["ALS Subject ID", "P Values"]
#print(tabulate(table_pvalues, headers=headers, tablefmt="grid"))

#for sid, m50, m500 in zip(subject_ids, means_50, means_500):
    #print(f"{sid}: 50ep = {m50:.4f}, 500ep = {m500:.4f}, Δ = {m500 - m50:.4f}")