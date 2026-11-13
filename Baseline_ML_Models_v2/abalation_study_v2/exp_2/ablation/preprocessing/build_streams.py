"""
preprocessing/build_streams.py
==============================
Parametrized preprocessing for the leave-one-out ablation: independently toggle
priming removal, downtime removal, and peak (artifact) removal, then produce a
per-minute pressure CSV ready for build_features.py.

Input is `machine_filtered_v1` — the priming-removal INPUT from the denoising
notebook (after the clinical-time anomaly filter + visit numbering). Export it
once from CRRT_stream_denoising_abalation.ipynb:

    machine_filtered_v1.to_csv(os.path.join(parent_dir, "Intermediate",
                               "machine_filtered_v1.csv"), index=False)

The priming/downtime rules are the validated vectorized rules from the notebook;
peak removal reuses the vendored smoothing function (preprocessing/smoothing.py).
"""

import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
sys.path.insert(0, _HERE)
from abl_config import CHANNELS                                       # noqa: E402
from smoothing import detect_and_smooth_outliers_skip_nan            # noqa: E402

EFFLUENT_DOSE_ML_KG_HR = 25.0


def _apply_priming_downtime(df, remove_priming, remove_downtime,
                            downtime="dose", Th=0.8, weight_map=None,
                            expected_step="1min"):
    STEP = pd.to_timedelta(expected_step)
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values(["PID", "Visit_no", "Time"])
    grp = df.groupby(["PID", "Visit_no"])
    valid_edge = grp["Time"].diff().eq(STEP)

    drop = pd.Series(False, index=df.index)
    if remove_priming:
        drop |= valid_edge & df["RUN_TIME"].eq(0.0) & grp["RUN_TIME"].shift().eq(0.0)
    if remove_downtime:
        if downtime == "flat":
            drop |= valid_edge & df["EFFLUENT"].eq(grp["EFFLUENT"].shift())
        else:  # dose
            dEFF = grp["EFFLUENT"].diff()
            W = df["PID"].map(weight_map or {})
            thr = Th * (EFFLUENT_DOSE_ML_KG_HR * W) / 60.0
            drop |= valid_edge & W.notna() & (dEFF < thr)
    return df.loc[~drop].reset_index(drop=True)


def _compute_day_cap14(df):
    d = df.copy()
    d["Date"] = d["Time"].dt.date
    d["Day"] = d.groupby(["PID", "Visit_no"])["Date"].rank(method="dense").astype(int)
    d = d.drop(columns="Date")
    return d[~(d["Day"] > 14)]


def clean_streams(machine_filtered_v1, *, remove_priming=True, remove_downtime=True,
                  remove_peak=True, downtime="dose", Th=0.8, weight_map=None):
    """Apply the three toggles and return a per-minute pressure DataFrame.

    Returns columns PID, Visit_no, Day, Time, <4 channels>.
    """
    cleaned = _apply_priming_downtime(machine_filtered_v1, remove_priming, remove_downtime,
                                      downtime=downtime, Th=Th, weight_map=weight_map)
    capped = _compute_day_cap14(cleaned)
    if remove_peak:
        capped = detect_and_smooth_outliers_skip_nan(
            capped, time_col="Time", pid_col="PID", feat_cols=CHANNELS,
            visit_col="Visit_no", window="40min", threshold=2, min_periods=1,
            neighbor_count=4, fill_nan=False)
    return capped[["PID", "Visit_no", "Day", "Time"] + CHANNELS]


# --- smoke test on synthetic data ---
if __name__ == "__main__":
    t0 = pd.Timestamp("2024-01-01")
    rows, eff = [], 1000.0
    for k in range(8):
        rt = 0.0 if k < 2 else 5.0           # first 2 min priming
        eff += (0 if k in (2,) else 30)      # k=2 is a stall (flat-ish)
        rows.append((1, 1, t0 + pd.Timedelta(minutes=k), rt, eff,
                     -58 + (50 if k == 5 else 0), -120, 30, 150))  # k=5 ACCESS peak
    cols = ["PID", "Visit_no", "Time", "RUN_TIME", "EFFLUENT"] + CHANNELS
    df = pd.DataFrame(rows, columns=cols)
    wmap = {1: 70.0}

    base = clean_streams(df, remove_priming=True, remove_downtime=True, remove_peak=True,
                         downtime="dose", Th=0.8, weight_map=wmap)
    no_prime = clean_streams(df, remove_priming=False, remove_downtime=True, remove_peak=True,
                             downtime="dose", Th=0.8, weight_map=wmap)
    no_peak = clean_streams(df, remove_priming=True, remove_downtime=True, remove_peak=False,
                            downtime="dose", Th=0.8, weight_map=wmap)
    print("all3   rows:", len(base), "cols:", list(base.columns))
    print("no_prime rows:", len(no_prime), "(>= all3, keeps priming):", len(no_prime) >= len(base))
    print("no_peak rows:", len(no_peak), "(peaks kept -> ACCESS max higher):",
          float(no_peak["ACCESS_PRESSURE"].max()) >= float(base["ACCESS_PRESSURE"].max()))
    assert len(no_prime) >= len(base) and len(no_peak) >= 1
    print("OK: toggles independent, day cap + peak step run.")
