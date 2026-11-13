"""
preprocessing/build_sequences.py  (abalation_study_v2)
================================
Build the DL [T,7] sequence cache (.npz) from a pressure CSV — self-contained
(no dependency on the Deep_Learning_Models prepare_sequences).

7 channels = 4 raw pressures + 3 derived CRRT-physiology channels (TMP, PFD, ARD),
matching the enriched classical feature pipeline so the DL encoders see the same signal.

Membership + labels come from the freshly built machine per-day tables (so the
sequence set matches the classical machine tables exactly). Per-channel mean/std
are fit on TRAIN timesteps only and applied to val/test (no leakage). Each
circuit-day stream is padded/right-cropped to MAX_LEN with a boolean mask.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import CHANNELS, DERIVED_CHANNELS, ALL_CHANNELS, KEY_COLS    # noqa: E402

MAX_LEN = 1440
PAD_VALUE = 0.0
TIME_COL = "Time"
NCH = len(ALL_CHANNELS)            # v2: 4 raw + 3 derived (TMP/PFD/ARD) = 7


def _with_derived(g):
    """Stack the 4 raw + 3 derived CRRT-physiology channels -> [T, 7] float32 array."""
    a = g["ACCESS_PRESSURE"].to_numpy(np.float32); f = g["FILTER_PRESSURE"].to_numpy(np.float32)
    e = g["EFFLUENT_PRESSURE"].to_numpy(np.float32); r = g["RETURN_PRESSURE"].to_numpy(np.float32)
    tmp = (f + r) / 2.0 - e; pfd = f - r; ard = r - a
    return np.column_stack([a, f, e, r, tmp, pfd, ard]).astype(np.float32)


def _stream_index(pressure_csv):
    raw = pd.read_csv(pressure_csv, usecols=KEY_COLS + [TIME_COL] + CHANNELS)
    raw = raw.sort_values(KEY_COLS + [TIME_COL])
    idx = {}
    for key, g in raw.groupby(KEY_COLS, sort=False):
        idx[tuple(int(k) for k in key)] = _with_derived(g)
    return idx


def _fit_stats(keys_df, index):
    chunks = [index[k] for k in (tuple(int(v) for v in r) for r in keys_df[KEY_COLS].to_numpy())
              if k in index and len(index[k])]
    stacked = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, NCH))
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0); std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _pad_crop(arr):
    out = np.full((MAX_LEN, NCH), PAD_VALUE, dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    t = arr.shape[0]
    if t == 0:
        return out, mask
    if t >= MAX_LEN:
        out[:] = arr[:MAX_LEN]; mask[:] = True
    else:
        out[:t] = arr; mask[:t] = True
    return out, mask


def _assemble(keys_df, index, mean, std):
    X, M, Y, K = [], [], [], []
    for _, r in keys_df.iterrows():
        key = (int(r.PID), int(r.Visit_no), int(r.Day))
        arr = index.get(key)
        if arr is None or len(arr) == 0:
            arr = np.zeros((0, NCH), np.float32)
        if arr.size:
            arr = arr.copy()
            inds = np.where(np.isnan(arr)); arr[inds] = np.take(mean, inds[1])
            arr = (arr - mean) / std
        padded, mask = _pad_crop(arr)
        X.append(padded); M.append(mask); Y.append(int(r["label"])); K.append(key)
    return np.stack(X), np.stack(M), np.array(Y, np.int64), np.array(K, np.int64)


def build_sequence_cache(pressure_csv, machine_dir, case, out_dir):
    """machine_dir holds the per-day {train,val,test}.csv (membership+labels)."""
    os.makedirs(out_dir, exist_ok=True)
    index = _stream_index(pressure_csv)
    keys = {s: pd.read_csv(os.path.join(machine_dir, f"{s}.csv"),
                           usecols=KEY_COLS + ["label"]) for s in ["train", "val", "test"]}
    mean, std = _fit_stats(keys["train"], index)
    summary = {"case": case, "mean": mean.tolist(), "std": std.tolist(), "splits": {}}
    for s in ["train", "val", "test"]:
        X, M, Y, K = _assemble(keys[s], index, mean, std)
        np.savez_compressed(os.path.join(out_dir, f"{case}_{s}.npz"),
                            X=X, mask=M, y=Y, keys=K, channels=np.array(ALL_CHANNELS))
        summary["splits"][s] = {"n": int(len(Y)), "n_pos": int(Y.sum()), "shape": list(X.shape)}
        print(f"  [{case}/{s}] N={len(Y)} pos={int(Y.sum())} X={X.shape}")
    with open(os.path.join(out_dir, f"{case}_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure-csv", required=True)
    ap.add_argument("--machine-dir", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    build_sequence_cache(a.pressure_csv, a.machine_dir, a.case, a.out_dir)
