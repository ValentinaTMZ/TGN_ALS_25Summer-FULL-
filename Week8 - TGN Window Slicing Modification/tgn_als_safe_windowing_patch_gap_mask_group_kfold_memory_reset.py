"""
Minimal, drop-in patch for Mingzhe Tao's TGN ALS pipeline to:
  (1) keep global time order (concat-first),
  (2) strictly prevent cross-trial windows,
  (3) avoid leakage with GroupKFold-by-trial,
  (4) reset TGN memory at trial boundaries,
  (5) (optional) compute ESS & adjacent-window correlation for reporting.

HOW TO USE
---------
1) Paste this file's functions into your project (or import it as a module) and
   replace the original counterparts. All names try to match your style.
2) Replace calls to aggregate_temporalData(...) with the new one here
   (returns dataset, meta). Use build_trial_loaders(...) + GroupKFold to split.
3) In train()/test(), call model.memory.reset_memory() once at the beginning of
   each batch (one trial per batch in this minimal change version).

Note: This patch assumes you already have:
 - temporalData([...]) factory returning a PyG TemporalData-like object
 - manual_collate_temporal_batch(batch_list)
 - model with model.memory.update_state(src, dst, t, msg) then forward(...)
 - criterion, optimizer defined outside
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from scipy.signal import hilbert
from sklearn.model_selection import GroupKFold
import torch

# =============================================================
# 1) Safe concat with GAP + mask + trial_id
# =============================================================

def concat_with_gaps(trials: List[np.ndarray], fs: int, L_max_sec: float, stride_sec: float):
    """Concatenate trials in global time order with explicit boundary gaps.

    Args:
        trials: list of arrays [Ti, C]
        fs: sampling rate (Hz)
        L_max_sec: max window (sec) you will consider in this run
        stride_sec: overlap step (sec). Use to set a conservative gap length.

    Returns:
        X:   [T_total, C] concatenated signal
        mask:[T_total] bool, True=valid signal, False=gap
        tid: [T_total] int, trial id per sample (-1 for gaps)
        ranges: list of (start, end, trial_id) on the global time axis
    """
    L_max = int(round(L_max_sec * fs))
    G = L_max + int(round(stride_sec * fs))  # CRITICAL: gap >= L_max

    X_list, M_list, TID_list, ranges = [], [], [], []
    cursor = 0
    for i, arr in enumerate(trials):
        if arr.ndim != 2:
            raise ValueError(f"trial {i} must be 2D [T,C], got shape {arr.shape}")
        T = arr.shape[0]
        X_list.append(arr)
        M_list.append(np.ones(T, dtype=bool))
        TID_list.append(np.full(T, i, dtype=int))
        ranges.append((cursor, cursor+T, i))
        cursor += T
        # append gap
        gap = np.zeros((G, arr.shape[1]), dtype=arr.dtype)
        X_list.append(gap)
        M_list.append(np.zeros(G, dtype=bool))
        TID_list.append(np.full(G, -1, dtype=int))
        cursor += G

    X   = np.vstack(X_list)
    mask= np.concatenate(M_list)
    tid = np.concatenate(TID_list)
    return X, mask, tid, ranges


# =============================================================
# 2) Strict non-cross-trial window slicing (global time preserved)
# =============================================================

def slice_windows_safe(X: np.ndarray, mask: np.ndarray, tid: np.ndarray,
                       fs: int, window_sec: float, overlap_sec: float,
                       snap_last: bool=True) -> Tuple[List[int], List[int]]:
    """Return global start/end indices of windows that:
        - do not touch gaps, and
        - start/end belong to the same trial (no cross-trial windows).
    """
    W = int(round(window_sec * fs))
    hop = int(round((window_sec - overlap_sec) * fs))
    if hop <= 0:
        raise ValueError("overlap_sec must be smaller than window_sec")

    T = X.shape[0]
    starts = list(range(0, max(0, T - W) + 1, hop))
    if snap_last and (len(starts) == 0 or starts[-1] + W < T):
        last_start = T - W
        if len(starts) == 0 or last_start != starts[-1]:
            starts.append(last_start)

    val_s, val_e = [], []
    for s in starts:
        e = s + W
        if not mask[s:e].all():
            continue  # touches a gap
        if tid[s] != tid[e-1]:
            continue  # crosses a trial boundary
        val_s.append(s); val_e.append(e)
    return val_s, val_e


# =============================================================
# 3) PLV computation on windows (reuses your logic)
# =============================================================

def plv_windows_from_indices(X: np.ndarray, starts: List[int], ends: List[int]) -> List[np.ndarray]:
    """Compute PLV matrices for each valid window from indices.
    X: [T, C]
    Returns list of [C, C] PLV matrices.
    """
    C = X.shape[1]
    plv_list = []
    for s, e in zip(starts, ends):
        eeg = X[s:e, :]
        plv = np.zeros((C, C), dtype=float)
        # Compute phase via Hilbert transform
        phases = np.angle(hilbert(eeg, axis=0))  # [W, C]
        for i in range(C):
            for j in range(i+1, C):
                dphi = phases[:, j] - phases[:, i]
                plv_ij = np.abs(np.mean(np.exp(1j * dphi)))
                plv[i, j] = plv[j, i] = plv_ij
        plv_list.append(plv)
    return plv_list


# =============================================================
# 4) Aggregate per subject with concat-first but safe slicing
# =============================================================

def aggregate_temporalData(subject_data: Dict[str, Any], sampling_freq: int=256,
                           window_sec: float=6.0, overlap_sec: float=1.0,
                           L_max_sec: float|None=None):
    """Build dataset for one subject using safe concat+slice.

    Returns:
        dataset: List[TemporalData]
        meta:    List[(label, trial_id, start_idx, end_idx)] for grouping/reset
    """
    dataset: List[Any] = []
    meta: List[Tuple[int,int,int,int]] = []

    for label, field in enumerate(['L', 'R']):  # keep your label mapping
        trials = subject_data[field][0]          # list of [T, n_channels]
        # Select your 19 electrodes; adapt indices as in your original code
        trials = [tr[:, :19] for tr in trials]

        Lmax = L_max_sec if L_max_sec is not None else window_sec
        X, mask, tid, _ = concat_with_gaps(trials, sampling_freq, Lmax, overlap_sec)

        starts, ends = slice_windows_safe(X, mask, tid, sampling_freq,
                                          window_sec, overlap_sec, snap_last=True)
        if not starts:
            continue

        plv_list = plv_windows_from_indices(X, starts, ends)
        for plv, s, e in zip(plv_list, starts, ends):
            tr_id = int(tid[s])
            temp = temporalData([plv], label, t_stamps=[s])  # keep your factory
            if temp.src.numel() == 0 or temp.dst.numel() == 0:
                continue
            dataset.append(temp)
            meta.append((label, tr_id, int(s), int(e)))

    return dataset, meta


# =============================================================
# 5) GroupKFold by trial (no window-level random split)
# =============================================================

def split_by_trial(subject_temporal_data: List[Any], meta: List[Tuple[int,int,int,int]], n_splits: int=10):
    groups = np.array([tr for (_, tr, _, _) in meta], dtype=int)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(subject_temporal_data, groups=groups)):
        yield fold, tr_idx, te_idx


# =============================================================
# 6) Build trial-wise loaders (each batch = one trial)
# =============================================================

def build_trial_loaders(data_list: List[Any], meta_list: List[Tuple[int,int,int,int]], indices: np.ndarray):
    per_trial: Dict[int, List[int]] = defaultdict(list)
    for k in indices:
        _, tr_id, _, _ = meta_list[k]
        per_trial[tr_id].append(k)
    loaders: List[List[Any]] = []
    for tr_id, idxs in per_trial.items():
        idxs = sorted(idxs, key=lambda x: meta_list[x][2])  # sort by start time
        loaders.append([data_list[i] for i in idxs])
    return loaders


# =============================================================
# 7) Training / Testing loops with memory reset per trial
# =============================================================

def train_one_subject(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for batch_list in train_loader:              # one batch = one trial
        model.memory.reset_memory()              # reset at trial boundary
        batch = manual_collate_temporal_batch(batch_list).to(device)
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        n_id = torch.cat([batch.src, batch.dst]).unique()
        batch_idx = batch.batch
        # update memory then forward (match your current ordering)
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        out = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
        loss = criterion(out, batch.y)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))


def test_one_subject(model, criterion, test_loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch_list in test_loader:           # one batch = one trial
            model.memory.reset_memory()
            batch = manual_collate_temporal_batch(batch_list).to(device)
            edge_index = torch.stack([batch.src, batch.dst], dim=0)
            n_id = torch.cat([batch.src, batch.dst]).unique()
            batch_idx = batch.batch
            out = model(edge_index, batch.t, batch.msg, n_id, batch_idx)
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total   += batch.y.size(0)
            total_loss += loss.item()
    acc = correct / total if total > 0 else 0.0
    return acc, total_loss / max(1, len(test_loader))


# =============================================================
# 8) Optional: Adjacent-window correlation & ESS reporter
# =============================================================

def vectorize_plv(plv: np.ndarray) -> np.ndarray:
    """Upper-triangle vectorization (excluding diagonal)."""
    C = plv.shape[0]
    iu = np.triu_indices(C, k=1)
    return plv[iu]


def estimate_adjacent_corr_and_ess(plv_list: List[np.ndarray]) -> Tuple[float, float]:
    """Estimate correlation between adjacent windows and derive ESS.
       Uses cosine similarity of vectorized PLVs as correlation proxy.
    """
    if len(plv_list) < 2:
        return 0.0, float(len(plv_list))
    vecs = np.stack([vectorize_plv(p) for p in plv_list], axis=0)
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs = vecs / norms
    # cosine between consecutive windows
    sims = np.sum(vecs[1:] * vecs[:-1], axis=1)
    rho = float(np.clip(np.mean(sims), -1.0, 1.0))
    Nprime = float(len(plv_list))
    ESS = Nprime * (1 - rho) / (1 + rho + 1e-8)
    return rho, ESS


# =============================================================
# 9) Example driver (sketch): use inside your per-subject loop
# =============================================================
"""
subject_temporal_data, window_meta = aggregate_temporalData(
    subject_data, sampling_freq=256, window_sec=6.0, overlap_sec=1.0, L_max_sec=6.0
)

for fold, tr_idx, te_idx in split_by_trial(subject_temporal_data, window_meta, n_splits=10):
    train_loader = build_trial_loaders(subject_temporal_data, window_meta, tr_idx)
    test_loader  = build_trial_loaders(subject_temporal_data, window_meta, te_idx)

    # ... init model/optimizer/criterion/device ...
    train_one_subject(model, optimizer, criterion, train_loader, device)
    acc, _ = test_one_subject(model, criterion, test_loader, device)
    print(f"fold {fold}: acc={acc:.3f}")
"""


# =============================================================
# 10) STEP1–STEP3: Global-min-window workflow (unified window size)
# =============================================================

# ---------- STEP 1: compute global minimum trial length (in seconds) ----------

def subject_min_trial_sec(subject_data: dict, fs: int=256) -> float:
    """Return the minimum trial duration (seconds) across L/R for one subject.
    Assumes subject_data['L'][0] and subject_data['R'][0] are lists of [T, C].
    """
    lens = []
    for field in ['L', 'R']:
        if field not in subject_data: 
            continue
        trials = subject_data[field][0]
        for tr in trials:
            lens.append(tr.shape[0])
    if not lens:
        raise ValueError("No trials found in subject_data.")
    return min(lens) / float(fs)


def global_min_trial_sec(all_subjects: list[dict], fs: int=256) -> float:
    """Compute the global minimum trial length (seconds) across all subjects."""
    mins = [subject_min_trial_sec(sd, fs) for sd in all_subjects]
    return float(min(mins))


# ---------- STEP 2/3 helpers: per-subject CV at a fixed (window, overlap) -----

def cross_validate_subject_fixed(subject_data: dict, fs: int, window_sec: float, overlap_sec: float,
                                 model_factory, n_splits: int=10, device: str='cuda') -> dict:
    """Run GroupKFold CV for one subject at a fixed (window, overlap).
    Returns a dict with mean_acc, fold_accs and optional stats.
    model_factory: a zero-arg callable that returns a fresh model instance.
    """
    data, meta = aggregate_temporalData(subject_data, sampling_freq=fs,
                                        window_sec=window_sec, overlap_sec=overlap_sec,
                                        L_max_sec=window_sec)
    fold_accs = []
    for fold, tr_idx, te_idx in split_by_trial(data, meta, n_splits=n_splits):
        train_loader = build_trial_loaders(data, meta, tr_idx)
        test_loader  = build_trial_loaders(data, meta, te_idx)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        # 简单训练一个 epoch（你可替换为原有的 epoch 循环）
        _ = train_one_subject(model, optimizer, criterion, train_loader, device)
        acc, _ = test_one_subject(model, criterion, test_loader, device)
        fold_accs.append(acc)
    mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
    std_acc  = float(np.std(fold_accs)) if fold_accs else 0.0
    return {"mean_acc": mean_acc, "std_acc": std_acc, "fold_accs": fold_accs}


# ---------- STEP 2 & 3: run grid of overlaps with a unified window size -------

def run_fixed_window_overlaps(all_subjects: list[dict], fs: int, model_factory,
                              window_sec: float|None=None,
                              overlap_list: list[float] = (0.0, 0.5, 1.0, 2.0),
                              n_splits: int=10, device: str='cuda') -> dict:
    """Main experiment harness.
    If window_sec is None, use the global minimum trial length (Step 1&2).
    Returns a nested dict: results[overlap_sec][subject_idx] -> metrics.
    """
    if window_sec is None:
        window_sec = global_min_trial_sec(all_subjects, fs)
    results: dict = {float(ov): {} for ov in overlap_list}
    for ov in overlap_list:
        for si, subj in enumerate(all_subjects):
            metrics = cross_validate_subject_fixed(
                subj, fs, window_sec, ov, model_factory, n_splits=n_splits, device=device
            )
            results[float(ov)][si] = metrics
    # aggregate top-level stats per overlap
    summary = {}
    for ov, per_subj in results.items():
        accs = [m["mean_acc"] for m in per_subj.values()]
        summary[ov] = {
            "mean_acc": float(np.mean(accs)) if accs else 0.0,
            "std_acc":  float(np.std(accs)) if accs else 0.0,
            "n_subjects": len(accs)
        }
    return {"window_sec": float(window_sec), "per_overlap": results, "summary": summary}


# ---------- (Optional) pretty print utility ----------------------------------

def print_fixed_window_summary(exp):
    print(f"Unified window size = {exp['window_sec']:.3f} sec")
    print("Overlap  |  mean_acc  ±  std_acc  |  n_subjects")
    for ov, s in sorted(exp['summary'].items()):
        print(f"{ov:7.3f} |  {s['mean_acc']:.4f}  ±  {s['std_acc']:.4f}  |  {s['n_subjects']}")


# ---------- Example usage (inside your main) ---------------------------------
"""
# all_subjects = [subject1_data, subject2_data, ...]  # 你加载好的每个被试的 dict
# def model_factory():
#     return YourTGNModel(...)
# exp = run_fixed_window_overlaps(all_subjects, fs=256, model_factory=model_factory,
#                                 window_sec=None, overlap_list=[0.0, 0.5, 1.0, 2.0],
#                                 n_splits=10, device='cuda')
# print_fixed_window_summary(exp)
"""
