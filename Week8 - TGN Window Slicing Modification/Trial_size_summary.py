#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:09:46 2025

@author: taomingzhe
"""

import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
data_dir = './'
fs = 256                        # sampling rate

rows = []

for subj in subject_numbers:
    mat = sio.loadmat(f'{data_dir}/S{subj}.mat')
    subj_data = mat[f'Subject{subj}'][:, :]

    for label in ['L','R']:
        trials = subj_data[label][0]
        for tri in trials:
            length_samples = tri.shape[0]
            length_sec = round(length_samples / fs, 2)  # 两位小数
            rows.append({'Subject':subj, 'Seconds':length_sec})

df = pd.DataFrame(rows)

# pivot summary table：
summary = df.pivot_table(index='Subject',
                         columns='Seconds',
                         aggfunc='size', fill_value=0)

rows = []
for subj in summary.index:
    row_sorted = summary.loc[subj].sort_values(ascending=False)
    rows.append(row_sorted)

final = pd.DataFrame(rows)
final.index = summary.index
print(final)

final = final.iloc[:, :10]  # keep only first 10 columns

n_rows = len(final)
n_cols = len(final.columns)

fig, ax = plt.subplots(figsize=(1.1*n_cols, 0.5*n_rows + 1.5))
ax.axis('off')

tbl = ax.table(cellText=final.values,
               colLabels=final.columns,
               rowLabels=final.index,
               cellLoc='center',
               loc='center')

tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)

# Bold the header row and left index
for (r, c), cell in tbl.get_celld().items():
    if r == 0 or c == -1:
        cell.get_text().set_fontweight('bold')

plt.text(-0.17, 0.5, 'Subject', transform=ax.transAxes,
         fontsize=14, fontweight='bold', rotation=90, va='center')

plt.text(0.5, 0.73, 'Trial Length (seconds)', transform=ax.transAxes,
         fontsize=14, fontweight='bold', ha='center')

fig.suptitle('Distribution of Trial Durations per Subject (Top 10 most frequent)', fontsize=16, fontweight='bold', y=0.755)

#fig.savefig('Trial_size_summary.svg', format='svg')
fig.savefig('Trial_size_summary.pdf', format='pdf', bbox_inches='tight')
fig.savefig('Trial_size_summary.png', format='png', bbox_inches='tight')
plt.show()
