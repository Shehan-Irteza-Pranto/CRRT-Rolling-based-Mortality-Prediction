"""
prepare_sequences.py
====================
Build raw [T, 4] pressure-stream tensors per circuit-day for the sequence models,
inheriting split membership and labels VERBATIM from the classical Model_Ready CSVs.

Why this matters: the sequence models' whole contribution is learning within-window
temporal/waveform structure that the 52 engineered summary stats discard. So we feed
the raw per-minute streams, not the summaries — but on the *exact same* patient-level
splits and labels as the classical pipeline, or the head-to-head is meaningless.

Pipeline
--------
  1. For each case (1day/2day) and split (train/val/test), read the classical machine
     split CSV  ->  the authoritative set of (PID, Visit_no, Day, label) keys.
  2. Join those keys 1:1 to the raw pressure stream rows; sort by Time -> [T, 4].
  3. Fit per-channel mean/std on TRAIN timesteps only; apply to val/test.
  4. Pad/crop to MAX_LEN, build a boolean mask (handles variable length / missing segments).
  5. Save one .npz per (case, split): X [N, MAX_LEN, 4], mask [N, MAX_LEN], y [N], keys [N, 3].

This module is framework-free (numpy/pandas) and runnable now — no torch required.

Usage
-----
    python prepare_sequences.py --cases 1day 2day --stream denoised
    python prepare_sequences.py --inspect-lengths        # print length percentiles, no write
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_READY_DIR, STREAM_FILES, DEFAULT_STREAM, SEQ_CACHE_DIR,
    CHANNELS, KEY_COLS, TIME_COL, TARGET_COL, CASES,
    MAX_LEN, PAD_VALUE, CROP_SIDE,
)

SPLITS = ["train", "val", "test"]


# --- Load classical split membership (the authoritative keys + labels) ---

def load_split_keys(case: str, split: str) -> pd.DataFrame:
    """Return the (PID, Visit_no, Day, label) rows for a classical machine split.

    Membership and labels come from Data/Processed/Model_Ready/{case}/denoised/{split}.csv —
    the same files the classical machine pipeline trains on. Sequence models are the *machine*
    branch, so they inherit machine-split membership exactly.
    """
    path = os.path.join(MODEL_READY_DIR, case, "denoised", f"{split}.csv")
    df = pd.read_csv(path, usecols=KEY_COLS + [TARGET_COL])
    return df


# --- Build [T, 4] arrays from the raw stream ---

def build_stream_index(stream_csv: str) -> dict:
    """Group the raw stream once into {(PID, Visit_no, Day): np.ndarray[T, 4]} (Time-sorted)."""
    usecols = KEY_COLS + [TIME_COL] + CHANNELS
    raw = pd.read_csv(stream_csv, usecols=usecols)
    raw = raw.sort_values(KEY_COLS + [TIME_COL])
    index = {}
    for key, grp in raw.groupby(KEY_COLS, sort=False):
        index[tuple(int(k) for k in key)] = grp[CHANNELS].to_numpy(dtype=np.float32)
    return index


def fit_channel_stats(keys_df: pd.DataFrame, stream_index: dict) -> dict:
    """Per-channel mean/std over all TRAIN timesteps (ignoring NaNs). Fit on train only."""
    chunks = []
    for _, r in keys_df.iterrows():
        arr = stream_index.get((int(r.PID), int(r.Visit_no), int(r.Day)))
        if arr is not None and len(arr):
            chunks.append(arr)
    stacked = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, len(CHANNELS)))
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def pad_crop(arr: np.ndarray):
    """Normalize length to MAX_LEN. Returns (padded [MAX_LEN, 4], mask [MAX_LEN] bool)."""
    t = arr.shape[0]
    out = np.full((MAX_LEN, len(CHANNELS)), PAD_VALUE, dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    if t == 0:
        return out, mask
    if t >= MAX_LEN:
        arr = arr[:MAX_LEN] if CROP_SIDE == "right" else arr[-MAX_LEN:]
        out[:] = arr
        mask[:] = True
    else:
        out[:t] = arr
        mask[:t] = True
    return out, mask


def assemble_split(keys_df, stream_index, stats):
    """Normalize, pad/crop, and stack a split into arrays."""
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    X, M, Y, K = [], [], [], []
    missing = 0
    for _, r in keys_df.iterrows():
        key = (int(r.PID), int(r.Visit_no), int(r.Day))
        arr = stream_index.get(key)
        if arr is None or len(arr) == 0:
            arr = np.zeros((0, len(CHANNELS)), dtype=np.float32)
            missing += 1
        # impute residual NaNs to the channel mean BEFORE normalizing, then standardize
        if arr.size:
            inds = np.where(np.isnan(arr))
            arr = arr.copy()
            arr[inds] = np.take(mean, inds[1])
            arr = (arr - mean) / std
        padded, mask = pad_crop(arr)
        X.append(padded); M.append(mask)
        Y.append(int(r[TARGET_COL])); K.append(key)
    return (np.stack(X), np.stack(M),
            np.array(Y, dtype=np.int64), np.array(K, dtype=np.int64), missing)


# --- Orchestration ---

def prepare_case(case: str, stream_index: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    keys = {s: load_split_keys(case, s) for s in SPLITS}

    # Rigor assertion 1: zero patient leakage across splits (mirrors classical guarantee).
    pid_sets = {s: set(keys[s].PID.unique()) for s in SPLITS}
    assert not (pid_sets["train"] & pid_sets["test"]), "PID leakage train/test!"
    assert not (pid_sets["train"] & pid_sets["val"]),  "PID leakage train/val!"
    assert not (pid_sets["val"] & pid_sets["test"]),   "PID leakage val/test!"

    # Fit normalization on TRAIN only.
    stats = fit_channel_stats(keys["train"], stream_index)

    summary = {"case": case, "channel_stats": stats, "splits": {}}
    for s in SPLITS:
        X, M, Y, K, missing = assemble_split(keys[s], stream_index, stats)
        np.savez_compressed(
            os.path.join(out_dir, f"{case}_{s}.npz"),
            X=X, mask=M, y=Y, keys=K, channels=np.array(CHANNELS),
        )
        summary["splits"][s] = {
            "n": int(len(Y)), "n_pos": int(Y.sum()),
            "pos_rate": float(Y.mean()) if len(Y) else 0.0,
            "missing_streams": int(missing),
        }
        print(f"  [{case}/{s}] N={len(Y)} pos={int(Y.sum())} "
              f"({100*Y.mean():.1f}%) missing_streams={missing}  X={X.shape}")
    with open(os.path.join(out_dir, f"{case}_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def inspect_lengths(stream_index: dict):
    """Print circuit-day stream-length percentiles to inform MAX_LEN."""
    lens = np.array([len(a) for a in stream_index.values()])
    pct = [50, 75, 90, 95, 99, 100]
    print(f"  circuit-days={len(lens)}  min={lens.min()} max={lens.max()}")
    for p in pct:
        print(f"  p{p:>3} length = {int(np.percentile(lens, p))} minutes")
    print(f"  current MAX_LEN={MAX_LEN}; coverage at MAX_LEN = "
          f"{100*(lens <= MAX_LEN).mean():.1f}% of windows kept whole")


def main():
    ap = argparse.ArgumentParser(description="Build [T,4] sequence tensors from raw pressure streams")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--stream", choices=list(STREAM_FILES), default=DEFAULT_STREAM,
                    help="denoised (match classical) or noised")
    ap.add_argument("--inspect-lengths", action="store_true",
                    help="Print stream-length percentiles to choose MAX_LEN, then exit")
    args = ap.parse_args()

    stream_csv = STREAM_FILES[args.stream]
    print(f"Indexing raw stream: {stream_csv}")
    stream_index = build_stream_index(stream_csv)
    print(f"  indexed {len(stream_index)} circuit-day streams")

    if args.inspect_lengths:
        inspect_lengths(stream_index)
        return

    out_root = os.path.join(SEQ_CACHE_DIR, args.stream)
    for case in args.cases:
        print(f"\n=== case {case} (stream={args.stream}) ===")
        prepare_case(case, stream_index, out_root)
    print(f"\nDone. Tensors -> {out_root}")


if __name__ == "__main__":
    main()
