"""
dl_data.py — z-scored, window-reduced per-day signals for the CNN1D (exp_3 Phase 2)
==================================================================================
Reuse the gap-aware physical-unit day signal (exp_3/features/day_signal.load_raw_days), z-score
each channel on TRAIN valid minutes (cached, reused for val/test), set masked minutes to 0, then
window-reduce with the parent window_aggregate. Returns per-day reduced sequences — one sample per
circuit-day, so val/test scoring is honest day-level (no augmentation).
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "features"))
from dlcfg import (CASE, N_CHANNELS, DAY_LEN, REDUCE, DATA_DIR, DL_REDUCE_DIR)  # noqa: E402
from day_signal import load_raw_days                                           # noqa: E402
sys.path.insert(0, DL_REDUCE_DIR)
from dl_reduce import reduce_stream                                            # noqa: E402

_NORM = {}


def _norm_path(case):
    return os.path.join(DATA_DIR, f"norm_{case}.npz")


def _fit_norm(X, M):
    mean = np.zeros(N_CHANNELS, np.float32); std = np.ones(N_CHANNELS, np.float32)
    for c in range(N_CHANNELS):
        v = X[:, c, :][M & ~np.isnan(X[:, c, :])]
        if v.size:
            mean[c] = float(v.mean()); s = float(v.std()); std[c] = s if s > 1e-6 else 1.0
    return mean, std


def load_reduced(split, case=CASE):
    """Return (Xr[N,W,F] float32, maskr[N,W] bool, y[N], pid[N], day[N])."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"reduced_{case}_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Xr"], z["maskr"], z["y"], z["pid"], z["day"]

    if case not in _NORM:
        if os.path.exists(_norm_path(case)):
            z = np.load(_norm_path(case)); _NORM[case] = (z["mean"], z["std"])
        else:
            Xtr, Mtr, ktr = load_raw_days("train", case)
            mean, std = _fit_norm(Xtr, Mtr)
            np.savez(_norm_path(case), mean=mean, std=std); _NORM[case] = (mean, std)
    mean, std = _NORM[case]

    X, M, keys = load_raw_days(split, case)
    Xn = (X - mean[None, :, None]) / std[None, :, None]
    Xn[np.isnan(Xn)] = 0.0
    Xseq = np.transpose(Xn, (0, 2, 1))                         # [N,T,C]
    Xr, maskr, _ = reduce_stream(Xseq, M, REDUCE)
    y = keys["label"].to_numpy(np.int64)
    pid = keys["PID"].to_numpy(np.int64); day = keys["Day"].to_numpy(np.int64)
    np.savez_compressed(cache, Xr=Xr, maskr=maskr, y=y, pid=pid, day=day)
    return Xr, maskr, y, pid, day


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default=CASE)
    a = ap.parse_args()
    for s in ("train", "val", "test"):
        Xr, m, y, pid, day = load_reduced(s, a.case)
        print(f"[dl_data] {a.case}/{s}: Xr={Xr.shape} mask={m.shape} pos={int(y.sum())}/{len(y)} "
              f"patients={len(set(pid))} valid-win={m.mean():.3f}")
