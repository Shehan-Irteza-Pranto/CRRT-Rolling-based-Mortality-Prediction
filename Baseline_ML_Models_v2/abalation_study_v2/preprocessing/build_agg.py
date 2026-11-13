"""
preprocessing/build_agg.py
==========================
Cumulative (expanding-window) aggregation of any per-day feature table.

For each (PID, Visit_no) sorted by Day, every feature is expanded over days 1..D
into {mean, min, max, median}. Meta columns and the label are kept from day D.
Self-contained — works for machine / clinical / combined per-day tables.
"""

import os
import sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import TARGET_COL                                   # noqa: E402

META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
AGG_STATS = ["mean", "min", "max", "median"]


def expanding_agg(per_day_df):
    """Per-day table -> expanding {mean,min,max,median} over days 1..D per (PID,Visit_no)."""
    df = per_day_df.sort_values(["PID", "Visit_no", "Day"]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META]
    meta_cols = [c for c in df.columns if c in META]

    g = df.groupby(["PID", "Visit_no"], sort=False)[feat_cols]
    parts = []
    for stat in AGG_STATS:
        e = g.expanding(min_periods=1).agg(stat).reset_index(level=[0, 1], drop=True)
        e.columns = [f"{c}_{stat}" for c in feat_cols]
        parts.append(e)
    agg = pd.concat([df[meta_cols].reset_index(drop=True)] + parts, axis=1)
    return agg


def build_agg_dir(in_dir, out_dir):
    """Read {train,val,test}.csv from in_dir, write expanding-agg versions to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(in_dir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        agg = expanding_agg(pd.read_csv(path))
        agg.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        counts[split] = (len(agg), agg.shape[1])
    return counts


# --- smoke test ---
if __name__ == "__main__":
    df = pd.DataFrame({
        "PID": [1, 1, 1, 2], "Visit_no": [1, 1, 1, 1], "Day": [1, 2, 3, 1],
        "label": [0, 0, 1, 0], "Mortality": [1, 1, 1, 0],
        "days_until_death": [3, 2, 1, 99],
        "F": [10.0, 20.0, 30.0, 5.0],
    })
    out = expanding_agg(df)
    print(out.to_string(index=False))
    # PID1 Day3: expanding over [10,20,30] -> mean20,min10,max30,median20
    r = out[(out.PID == 1) & (out.Day == 3)].iloc[0]
    assert abs(r["F_mean"] - 20) < 1e-9 and r["F_min"] == 10 and r["F_max"] == 30 and r["F_median"] == 20
    assert "label" in out.columns and out.shape[1] == 6 + 4  # 6 meta + 4 stats of F
    print("OK: expanding agg (mean/min/max/median) correct, meta+label preserved.")
