
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Window slicing sanity check for Subject 1 (3s windows, 1s overlap).

It loads S1.mat from the current working directory (same assumption as your script),
concatenates trials for 'L' and 'R' separately and combined, then compares two policies:

1) strict (range upper bound only)  -> may drop tail
2) snap_to_end                      -> adds a final start so last window ends exactly at T

For each case it prints:
- total samples (T)
- window, hop (in samples)
- number of windows
- last stride (difference between last two starts; can be < hop)
- is_last_end_T (True means no miss)
- missed_tail_samples (strict only)
"""

import os
from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio

def collect_long_eeg(subject_data, field):
    """Concatenate ALL trials for a field ('L' or 'R'), keep first 19 channels."""
    trials = subject_data[field][0]            # [num_trials, time, 64]
    long_eeg = np.concatenate([t[:, :19] for t in trials], axis=0)  # [T, 19]
    return long_eeg

def make_starts(T, win, hop, snap_to_end=False):
    if hop <= 0:
        raise ValueError("Overlap must be smaller than window length.")
    # regular equally-spaced starts
    starts = list(range(0, max(0, T - win) + 1, hop))
    if snap_to_end and (len(starts) == 0 or starts[-1] + win < T):
        last_start = T - win
        if len(starts) == 0 or last_start != starts[-1]:
            starts.append(last_start)
    return np.array(starts, dtype=int)

def report(name, T, win, hop, starts, strict=False):
    print(f"\n=== {name} ===")
    print(f"T={T} samples | win={win} | hop={hop} | n_windows={len(starts)}")
    if len(starts) >= 2:
        diffs = np.diff(starts)
        print(f"stride stats -> min={diffs.min()}, max={diffs.max()}, last_stride={diffs[-1]}")
    else:
        print("stride stats -> <insufficient windows>")
    last_end = (starts[-1] + win) if len(starts)>0 else 0
    print(f"is_last_end_T: {last_end == T} (last_end={last_end})")
    if strict:
        missed = max(0, T - last_end)
        print(f"missed_tail_samples: {missed}")

def main(fs=256, win_sec=3, overlap_sec=1):
    data_dir = os.getcwd()
    mat = sio.loadmat(pjoin(data_dir, 'S1.mat'))
    subject_data = mat['Subject1'][:, :]

    win = int(fs * win_sec)
    hop = int(fs * (win_sec - overlap_sec))

    for field in ['L', 'R']:
        X = collect_long_eeg(subject_data, field)
        T = X.shape[0]

        starts_strict = make_starts(T, win, hop, snap_to_end=False)
        starts_snap   = make_starts(T, win, hop, snap_to_end=True)

        report(f"Subject1-{field} STRICT", T, win, hop, starts_strict, strict=True)
        report(f"Subject1-{field} SNAP_TO_END", T, win, hop, starts_snap, strict=False)

    # Combined L then R (only for coverage check)
    X_L = collect_long_eeg(subject_data, 'L')
    X_R = collect_long_eeg(subject_data, 'R')
    X_all = np.concatenate([X_L, X_R], axis=0)
    T_all = X_all.shape[0]
    starts_strict = make_starts(T_all, win, hop, snap_to_end=False)
    starts_snap   = make_starts(T_all, win, hop, snap_to_end=True)
    report("Subject1-ALL STRICT", T_all, win, hop, starts_strict, strict=True)
    report("Subject1-ALL SNAP_TO_END", T_all, win, hop, starts_snap, strict=False)

if __name__ == "__main__":
    main()
