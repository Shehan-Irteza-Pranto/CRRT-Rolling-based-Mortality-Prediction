"""
signal_pipeline_stats.py — data provenance / attrition through the CRRT signal-processing pipeline.
Computes per-stage minute-rows / circuit-days / circuits / patients, the peak-smoothed minute count
(diff pre-vs-post-peak), the gap-mask valid fraction (day_signal), and the final cohort. Writes
paper_materials/cohort/signal_pipeline_stats.csv (+ prints a markdown table).
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, _AS)
sys.path.insert(0, os.path.join(_AS, "exp_2", "ablation"))
sys.path.insert(0, os.path.join(_AS, "exp_2", "ablation", "models"))
sys.path.insert(0, os.path.join(_AS, "exp_2", "ablation", "preprocessing"))
import abl_config as A                                            # noqa: E402
import run_ablation as R                                         # noqa: E402
CH = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE"]
V = os.path.join(_AS, "exp_5_ablation", "data")
DENOISED = A.DENOISE_VARIANTS["current"]["denoised"]
MACH = lambda s: os.path.join(_AS, "exp_2", "ablation", "data", "abalation",
                              "current_denoised", "machine_only", "1day", f"{s}.csv")
OUT = os.path.join(_HERE, "cohort"); os.makedirs(OUT, exist_ok=True)


def _counts_pressure(path, has_day=True):
    cols = ["PID", "Visit_no", "Day"] if has_day else ["PID", "Visit_no"]
    d = pd.read_csv(path, usecols=cols)
    r = {"rows": len(d), "patients": d["PID"].nunique(),
         "circuits": d.groupby(["PID", "Visit_no"]).ngroups}
    r["circuit_days"] = d.groupby(["PID", "Visit_no", "Day"]).ngroups if has_day else np.nan
    return r


def _peak_smoothed():
    pre = pd.read_csv(os.path.join(V, "pressure_primdt.csv"))
    post = pd.read_csv(DENOISED)
    assert len(pre) == len(post), f"row mismatch {len(pre)} vs {len(post)}"
    # same base + order (peak preserves order); confirm on keys
    key = ["PID", "Visit_no", "Day", "Time"]
    assert (pre[key].head(1000).values == post[key].head(1000).values).all(), "key order mismatch"
    diff = np.zeros(len(pre), bool)
    for c in CH:
        a, b = pre[c].to_numpy(), post[c].to_numpy()
        d = ~np.isclose(a, b, atol=1e-6, equal_nan=True)
        diff |= d
    n_min = int(diff.sum())
    # per-channel cell changes
    cell = sum(int((~np.isclose(pre[c].to_numpy(), post[c].to_numpy(), atol=1e-6, equal_nan=True)).sum()) for c in CH)
    return n_min, len(pre), cell


def _gap_stats():
    sys.path.insert(0, os.path.join(_AS, "exp_3", "features"))
    import day_signal as DS
    valid = []
    for split in ("train", "val", "test"):
        X, M, keys, *_ = DS.load_raw_days(split, "1day")
        valid.append(M.mean(1))                                   # valid fraction per circuit-day
    v = np.concatenate(valid)
    return {"n_circuit_days": len(v), "mean_valid_frac": float(v.mean()),
            "median_valid_frac": float(np.median(v)), "frac_days_ge50": float((v >= 0.5).mean())}


def _cohort():
    rows = {}
    tot = {"cd": 0, "pat": 0, "deaths": 0}
    for s in ("train", "val", "test"):
        d = pd.read_csv(MACH(s), usecols=["PID", "label"])
        rows[s] = {"circuit_days": len(d), "patients": d["PID"].nunique(), "deaths": int(d["label"].sum())}
        tot["cd"] += len(d); tot["pat"] += d["PID"].nunique(); tot["deaths"] += int(d["label"].sum())
    return rows, tot


def main():
    print("counting stages ...", flush=True)
    st = {}
    st["0 Raw export (machine_filtered_v1)"] = _counts_pressure(R.MFV1, has_day=False)
    st["1 Day numbering + cap (Day<=14)"] = _counts_pressure(os.path.join(V, "pressure_raw.csv"))
    st["2 Priming removal (RUN_TIME=0)"] = _counts_pressure(os.path.join(V, "pressure_prim.csv"))
    st["3 Downtime removal (no-effluent)"] = _counts_pressure(os.path.join(V, "pressure_primdt.csv"))
    st["4 Peak denoising (40-min, |z|>2)"] = _counts_pressure(DENOISED)
    print("peak-smoothed minutes ...", flush=True)
    pk_min, pk_tot, pk_cell = _peak_smoothed()
    print("gap stats ...", flush=True)
    gap = _gap_stats()
    coh, tot = _cohort()

    rows = []
    prev = None
    for stage, c in st.items():
        drop = "" if prev is None else f"{prev - c['rows']:+,} rows"
        rows.append({"stage": stage, "minute_rows": c["rows"], "circuit_days": c["circuit_days"],
                     "circuits": c["circuits"], "patients": c["patients"], "delta_rows": drop})
        prev = c["rows"]
    rows.append({"stage": "5 Gap-aware reconstruction (interp<=10 / mask>10)", "minute_rows": "1440-min grid",
                 "circuit_days": gap["n_circuit_days"], "circuits": "", "patients": "",
                 "delta_rows": f"{gap['mean_valid_frac']*100:.1f}% minutes valid (mean/day)"})
    rows.append({"stage": "6 Day<=7 cap + label merge -> cohort", "minute_rows": tot["cd"],
                 "circuit_days": tot["cd"], "circuits": "", "patients": tot["pat"],
                 "delta_rows": f"{tot['deaths']} deaths"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "signal_pipeline_stats.csv"), index=False)

    print("\n=== signal-processing pipeline attrition ===")
    print(df.to_string(index=False))
    print(f"\npeak denoising: {pk_min:,} / {pk_tot:,} minutes had >=1 channel smoothed "
          f"({100*pk_min/pk_tot:.2f}%); {pk_cell:,} channel-cells changed ({100*pk_cell/(4*pk_tot):.2f}%)")
    print(f"gaps: mean {gap['mean_valid_frac']*100:.1f}% / median {gap['median_valid_frac']*100:.1f}% "
          f"of the 1440 minutes valid per circuit-day; {gap['frac_days_ge50']*100:.1f}% of days >=50% valid")
    print("cohort splits:", coh)
    # persist the derived stats too
    pd.DataFrame([{"peak_min_smoothed": pk_min, "peak_min_total": pk_tot, "peak_pct": round(100*pk_min/pk_tot, 3),
                   "peak_cells": pk_cell, **{f"gap_{k}": v for k, v in gap.items()},
                   "train_cd": coh["train"]["circuit_days"], "val_cd": coh["val"]["circuit_days"],
                   "test_cd": coh["test"]["circuit_days"], "train_deaths": coh["train"]["deaths"],
                   "val_deaths": coh["val"]["deaths"], "test_deaths": coh["test"]["deaths"]}]
                 ).to_csv(os.path.join(OUT, "signal_pipeline_derived.csv"), index=False)
    print(f"\nsaved -> {OUT}/signal_pipeline_stats.csv + signal_pipeline_derived.csv")


if __name__ == "__main__":
    main()
