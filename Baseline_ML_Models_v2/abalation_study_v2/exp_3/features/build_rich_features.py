"""
build_rich_features.py — exp_3 Phase 1
======================================
Compute the circuit-instability features the parent set lacks, from each day's gap-aware
physical-unit signal, and concatenate them onto the existing 133-feature machine_only table.

Per channel (6 base [ACCESS,FILTER,EFFLUENT,RETURN,TMP,PFD] + derived ARD = RETURN-ACCESS):
  spectral   : spec_entropy, bandpow_lo (period>30min), bandpow_hi (period<10min), dom_period
  complexity : perm_entropy (order 3)
  excursion  : cross_rate (median-crossings/min), max_run_above (longest run above median, frac)
  end-of-day : eod_delta (last2h - first2h mean), eod_slope (slope over last 2h)
Day-level coupling scalars (clotting co-movement):
  TMP_PFD_corr, ACCESS_RETURN_corr, FILTER_EFFLUENT_corr, TMP_PFD_cotrend

All feature names start with a parent channel prefix so abl_config.is_machine_col() treats them as
machine features. NaN where a day is too short to compute a stat -> imputed (train median) in
classical_rich.py. Writes exp_3/features/data/rich/<case>/<split>.csv.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.signal import welch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fcfg import (CASE, KEY_COLS, TARGET_COL, machine_table, rich_table)       # noqa: E402
from day_signal import load_raw_days                                          # noqa: E402

# channel index -> parent column prefix (so names match is_machine_col)
PREFIX = {0: "ACCESS_PRESSURE", 1: "FILTER_PRESSURE", 2: "EFFLUENT_PRESSURE",
          3: "RETURN_PRESSURE", 4: "TMP", 5: "PFD"}
MIN_VALID = 30            # need at least this many valid minutes to compute a channel's features
EOD_WIN = 120            # end-of-day window (minutes)


def _contig(series):
    """series [1440] with NaN -> uniformly-sampled segment over [first..last] valid (interior
    NaN linearly interpolated), plus the raw valid values. None if too short."""
    obs = np.where(~np.isnan(series))[0]
    if len(obs) < MIN_VALID:
        return None, None
    first, last = obs[0], obs[-1]
    seg = series[first:last + 1].astype(float)
    nan = np.isnan(seg)
    if nan.any():                                  # interpolate remaining (large) interior gaps
        xp = np.where(~nan)[0]
        seg[nan] = np.interp(np.where(nan)[0], xp, seg[xp])
    return seg, series[obs]


def _perm_entropy(x, order=3):
    """Normalised permutation entropy (0..1) of 1D x."""
    n = len(x) - order + 1
    if n < order + 1:
        return np.nan
    # ordinal patterns of consecutive `order`-tuples
    patt = np.argsort(np.argsort(np.lib.stride_tricks.sliding_window_view(x, order), axis=1), axis=1)
    # encode each permutation as a tuple -> count
    _, counts = np.unique(patt, axis=0, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(math.factorial(order)))


def _spectral(seg):
    """spec_entropy, bandpow_lo, bandpow_hi, dom_period from Welch PSD (fs=1/min)."""
    nperseg = min(256, len(seg))
    f, pxx = welch(seg - seg.mean(), fs=1.0, nperseg=nperseg)
    pxx = pxx[1:]; f = f[1:]                        # drop DC
    tot = pxx.sum()
    if tot <= 0 or len(f) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pn = pxx / tot
    spec_ent = float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn)))
    lo = float(pn[f < 1.0 / 30].sum())             # slow drift (period > 30 min)
    hi = float(pn[f > 1.0 / 10].sum())             # fast fluctuation (period < 10 min)
    dom_period = float(1.0 / f[int(np.argmax(pxx))])
    return spec_ent, lo, hi, min(dom_period, 1440.0)


def _excursion(vals):
    """cross_rate (median crossings / n), max_run_above (longest run above median / n)."""
    med = np.median(vals)
    above = vals > med
    cross = int(np.sum(above[1:] != above[:-1]))
    # longest run of True
    best = run = 0
    for a in above:
        run = run + 1 if a else 0
        best = max(best, run)
    n = len(vals)
    return cross / n, best / n


def _eod(seg):
    """end-of-day deterioration: last2h-first2h mean delta, slope over last 2h."""
    if len(seg) < 2 * EOD_WIN:
        return np.nan, np.nan
    delta = float(seg[-EOD_WIN:].mean() - seg[:EOD_WIN].mean())
    tail = seg[-EOD_WIN:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    return delta, slope


def _channel_feats(series):
    """All per-channel features for one channel's [1440] series."""
    seg, vals = _contig(series)
    if seg is None:
        return dict.fromkeys(
            ["spec_entropy", "bandpow_lo", "bandpow_hi", "dom_period", "perm_entropy",
             "cross_rate", "max_run_above", "eod_delta", "eod_slope"], np.nan)
    se, lo, hi, dom = _spectral(seg)
    pe = _perm_entropy(seg)
    cr, mra = _excursion(vals)
    ed, es = _eod(seg)
    return {"spec_entropy": se, "bandpow_lo": lo, "bandpow_hi": hi, "dom_period": dom,
            "perm_entropy": pe, "cross_rate": cr, "max_run_above": mra,
            "eod_delta": ed, "eod_slope": es}


def _corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < MIN_VALID:
        return np.nan
    aa, bb = a[m], b[m]
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _slope(series):
    obs = np.where(~np.isnan(series))[0]
    if len(obs) < MIN_VALID:
        return np.nan
    return float(np.polyfit(obs, series[obs], 1)[0])


def _day_row(x6):
    """x6 [6,1440] physical NaN-masked -> dict of all new features for one day."""
    row = {}
    ard = x6[3] - x6[0]                              # RETURN - ACCESS
    chans = list(range(6))
    for c in chans:
        for k, v in _channel_feats(x6[c]).items():
            row[f"{PREFIX[c]}_{k}"] = v
    for k, v in _channel_feats(ard).items():
        row[f"ARD_{k}"] = v
    # cross-channel coupling
    row["TMP_PFD_corr"] = _corr(x6[4], x6[5])
    row["ACCESS_RETURN_corr"] = _corr(x6[0], x6[3])
    row["FILTER_EFFLUENT_corr"] = _corr(x6[1], x6[2])
    st, sp = _slope(x6[4]), _slope(x6[5])
    row["TMP_PFD_cotrend"] = (np.sign(st) * np.sign(sp)) if not (np.isnan(st) or np.isnan(sp)) else np.nan
    return row


def build_split(split, case=CASE):
    X, M, keys = load_raw_days(split, case)
    base = pd.read_csv(machine_table(split, case))
    assert len(base) == len(keys), f"row mismatch {len(base)} vs {len(keys)}"
    # sanity: same key order
    for k in KEY_COLS:
        assert (base[k].to_numpy() == keys[k].to_numpy()).all(), f"{split}: {k} order mismatch"

    new_rows = [_day_row(X[i]) for i in range(len(keys))]
    new_df = pd.DataFrame(new_rows)
    out_df = pd.concat([base.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    out = rich_table(split, case)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[rich] {case}/{split}: rows={len(out_df)} base_cols={base.shape[1]} "
          f"new_cols={new_df.shape[1]} total={out_df.shape[1]} "
          f"nan_frac_new={new_df.isna().mean().mean():.3f} -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default=CASE)
    a = ap.parse_args()
    for s in ("train", "val", "test"):
        build_split(s, a.case)
