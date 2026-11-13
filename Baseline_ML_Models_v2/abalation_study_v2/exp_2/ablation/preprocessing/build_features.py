"""
preprocessing/build_features.py  (abalation_study_v2)
=====================================================
Turn a per-minute pressure CSV into the ENRICHED machine per-day Model_Ready tables
(train/val/test) for a denoising variant, reusing the fixed patient splits so test
sets stay identical to the baseline.

Per-day machine features (~133) =
  (4 raw + 3 derived) channels x 13 distribution stats   = 91
  (4 raw + 3 derived) channels x  6 temporal-dynamics     = 42

Raw channels    : ACCESS / FILTER / EFFLUENT / RETURN _PRESSURE
Derived channels (per timestep, then 13 stats each):
  TMP = (FILTER + RETURN)/2 - EFFLUENT   transmembrane pressure (membrane fouling)
  PFD = FILTER - RETURN                  pressure drop across the filter (clotting)
  ARD = RETURN - ACCESS                  circuit pressure span / access instability
Dynamics stats (within-day, time-ordered series, per channel):
  slope, trend_r2, mad (mean |Δ|), max_jump (max |Δ|), ac1 (lag-1 autocorr),
  exc_frac (fraction of minutes with |z| > 2)

Usage
-----
    python build_features.py --variants Th70 Th80 Th90        # uses abl_config.DENOISE_VARIANTS
    python build_features.py --pressure-csv /path.csv --tag myvar --peak denoised
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats as _st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
from abl_config import (MODEL_READY_DIR, LABELS_DIR, SPLITS_JSON,        # noqa: E402
                        CHANNELS, DERIVED_CHANNELS, ALL_CHANNELS,
                        STAT_NAMES, DYN_STATS, DENOISE_VARIANTS, CASES, MAX_TREATMENT_DAY)

META_OUT = ["PID", "Visit_no", "Day", "Mortality", "days_until_death", "label"]
TIME_COL = "Time"


def _stats(vals):
    """13 distribution stats for one channel's day values (NaN-dropped)."""
    s = vals[~np.isnan(vals)]
    if len(s) == 0:
        return {k: np.nan for k in STAT_NAMES}
    q25, q75 = np.percentile(s, [25, 75])
    mean = float(np.mean(s)); std = float(np.std(s))
    return {
        "mean": mean, "std": std, "min": float(np.min(s)), "max": float(np.max(s)),
        "median": float(np.median(s)), "q25": float(q25), "q75": float(q75),
        "iqr": float(q75 - q25), "range": float(np.max(s) - np.min(s)),
        "skewness": float(_st.skew(s)) if len(s) > 2 else 0.0,
        "kurtosis": float(_st.kurtosis(s)) if len(s) > 3 else 0.0,
        "cv": float(std / mean) if mean != 0 else 0.0,
        "n_valid": int(len(s)),
    }


def _dyn(vals):
    """6 within-day temporal-dynamics stats on a time-ordered series (NaN-dropped).

    Captures clotting/fouling trends and pressure instability that order-invariant
    distribution stats discard. Gaps (removed downtime) are collapsed, not interpolated.
    """
    s = vals[~np.isnan(vals)].astype(float)
    if len(s) < 3:
        return {k: 0.0 for k in DYN_STATS}
    t = np.arange(len(s), dtype=float)
    # linear trend: slope + R^2 (trend strength)
    slope, intercept = np.polyfit(t, s, 1)
    fit = slope * t + intercept
    ss_res = float(np.sum((s - fit) ** 2)); ss_tot = float(np.sum((s - s.mean()) ** 2))
    trend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # minute-to-minute volatility / largest excursion
    d = np.diff(s)
    mad = float(np.mean(np.abs(d))); max_jump = float(np.max(np.abs(d)))
    # lag-1 autocorrelation (smoothness/instability)
    s0, s1 = s[:-1], s[1:]
    ac1 = float(np.corrcoef(s0, s1)[0, 1]) if s0.std() > 0 and s1.std() > 0 else 0.0
    # fraction of minutes that are spikes (|z| > 2)
    std = s.std()
    exc_frac = float(np.mean(np.abs((s - s.mean()) / std) > 2)) if std > 0 else 0.0
    return {"slope": float(slope), "trend_r2": float(trend_r2), "mad": mad,
            "max_jump": max_jump, "ac1": ac1, "exc_frac": exc_frac}


def _derived_streams(g):
    """Per-timestep derived CRRT-physiology channels from the 4 raw pressures."""
    a = g["ACCESS_PRESSURE"].to_numpy(float); f = g["FILTER_PRESSURE"].to_numpy(float)
    e = g["EFFLUENT_PRESSURE"].to_numpy(float); r = g["RETURN_PRESSURE"].to_numpy(float)
    return {"TMP": (f + r) / 2.0 - e, "PFD": f - r, "ARD": r - a}


def feature_columns():
    """Ordered list of the ~133 enriched machine feature columns."""
    dist = [f"{ch}_{s}" for ch in ALL_CHANNELS for s in STAT_NAMES]
    dyn  = [f"{ch}_{d}" for ch in ALL_CHANNELS for d in DYN_STATS]
    return dist + dyn


def pressure_to_day_features(pressure, case):
    """Compute the enriched per-day machine table merged with labels for `case`.

    `pressure` may be a path or a DataFrame with columns PID,Visit_no,Day,Time,<channels>.
    """
    df = pd.read_csv(pressure) if isinstance(pressure, str) else pressure
    if MAX_TREATMENT_DAY:                                  # cohort: first N treatment days only
        df = df[df["Day"] <= MAX_TREATMENT_DAY]
    if TIME_COL in df.columns:
        df = df.sort_values(["PID", "Visit_no", "Day", TIME_COL])
    rows = []
    for (pid, vis, day), g in df.groupby(["PID", "Visit_no", "Day"], sort=False):
        rec = {"PID": int(pid), "Visit_no": int(vis), "Day": int(day)}
        series = {ch: g[ch].to_numpy(dtype=float) for ch in CHANNELS}
        series.update(_derived_streams(g))               # TMP / PFD / ARD per timestep
        for ch in ALL_CHANNELS:
            v = series[ch]
            for stat, val in _stats(v).items():
                rec[f"{ch}_{stat}"] = val
            for stat, val in _dyn(v).items():
                rec[f"{ch}_{stat}"] = val
        rows.append(rec)
    feat = pd.DataFrame(rows)

    lab = pd.read_csv(os.path.join(LABELS_DIR, f"labels_{case}.csv"))
    lab = lab.rename(columns={f"label_{case}": "label"})
    keep = ["PID", "Visit_no", "Day", "Mortality", "days_until_death", "label"]
    keep = [c for c in keep if c in lab.columns]
    merged = feat.merge(lab[keep], on=["PID", "Visit_no", "Day"], how="inner")
    return merged[[c for c in META_OUT if c in merged.columns] + feature_columns()]


def write_splits(table, out_dir):
    """Split `table` by pid_splits.json and write train/val/test.csv."""
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in ["train", "val", "test"]:
        pids = set(int(p) for p in splits[split])
        sub = table[table["PID"].isin(pids)]
        sub.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        counts[split] = len(sub)
    return counts


def build_variant(tag, pressure_csv, case, peak="denoised", model_ready_dir=MODEL_READY_DIR):
    """Build + write the enriched machine table for one (tag, case, peak)."""
    table = pressure_to_day_features(pressure_csv, case)
    out_dir = os.path.join(model_ready_dir, "abalation", tag, case, peak)
    counts = write_splits(table, out_dir)
    print(f"[build_features] {tag}/{case}/{peak}: {counts}  ({table.shape[1]} cols)  -> {out_dir}")
    return out_dir, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[], help="keys in DENOISE_VARIANTS")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--peaks", nargs="+", default=["denoised", "noised"])
    ap.add_argument("--pressure-csv", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--peak", default="denoised")
    a = ap.parse_args()

    if a.pressure_csv:
        for case in a.cases:
            build_variant(a.tag or "custom", a.pressure_csv, case, a.peak)
        return
    for var in a.variants:
        for peak in a.peaks:
            src = DENOISE_VARIANTS[var][peak]
            if not os.path.exists(src):
                print(f"[skip] {var}/{peak}: missing {src}")
                continue
            for case in a.cases:
                build_variant(var, src, case, peak)


if __name__ == "__main__":
    main()
