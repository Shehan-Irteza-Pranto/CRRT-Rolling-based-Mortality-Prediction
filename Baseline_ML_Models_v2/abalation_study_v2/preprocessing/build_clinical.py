"""
preprocessing/build_clinical.py  (abalation_study_v2)
=====================================================
Build the v2 clinical table on the 70:10:20 split, with TWO methodological fixes vs the
baseline clinical pipeline:

  1. **Forward-fill ONLY** for labs (no backward-fill). The baseline carried lab values
     BACKWARD within a patient (a Day-5 lab filled Days 1-4) — a look-ahead leak for
     early-day mortality prediction. v2 carries labs forward only.
  2. **Lab-missingness indicators**: for every retained lab, a `<LAB>_missing` column = 1
     when no lab was drawn on that circuit-day's date (value is carried-forward or absent),
     0 when freshly measured. Captures the "was it measured" signal that imputation erases.

Labs are rebuilt from the RAW longitudinal labs (`Data/EHR/Lab/lab_transformed_longitudinal.csv`),
date-aligned to circuit-days by exact calendar date (PID, Date) from `labels_<case>.csv`. The
STATIC PTI/demographic features + meta + label are taken unchanged from the external
`Model_Ready/clinical_only` table (they need no temporal alignment). The merged per-day table is
then split by the v2 `pid_splits_v2.json`. Clinical is denoising-independent (runs once per case).

Usage:  python build_clinical.py            # all CASES
"""

import os
import sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import (CLINICAL_SRC, CLINICAL_ONLY, DATA_DIR, LABELS_DIR,  # noqa: E402
                        KEY_COLS, CASES, MAX_TREATMENT_DAY)
from build_features import write_splits                                   # noqa: E402  (v2 SPLITS_JSON)

LAB_PATH  = os.path.join(DATA_DIR, "EHR", "Lab", "lab_transformed_longitudinal.csv")
LAB_DATE  = "DailyLaboratoryDataDate"
META_KEEP = ["PID", "Visit_no", "Day", "Mortality", "days_until_death", "label"]


def _raw_daily_labs():
    """Raw longitudinal labs -> daily mean per (PID, Date), columns prefixed LAB_."""
    lab = pd.read_csv(LAB_PATH).drop(columns=["FormDataId", "TimeOfDay24Hour"], errors="ignore")
    val_cols = [c for c in lab.columns if c not in ("PID", LAB_DATE)]
    daily = lab.groupby(["PID", LAB_DATE])[val_cols].mean().reset_index()
    daily = daily.rename(columns={LAB_DATE: "Date", **{c: f"LAB_{c}" for c in val_cols}})
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily, [f"LAB_{c}" for c in val_cols]


def build_clinical_case(case):
    """Forward-fill-only labs + missingness indicators + static PTI, re-split by v2 splits."""
    # --- external clinical: take meta + PTI (static), and the retained-lab list (drops >40% missing) ---
    parts = [pd.read_csv(os.path.join(CLINICAL_SRC, case, f"{s}.csv"))
             for s in ("train", "val", "test")
             if os.path.exists(os.path.join(CLINICAL_SRC, case, f"{s}.csv"))]
    if not parts:
        print(f"[build_clinical] SKIP {case}: no external clinical at {CLINICAL_SRC}/{case}")
        return None
    ext = pd.concat(parts, ignore_index=True).drop_duplicates(subset=KEY_COLS)
    pti_cols  = [c for c in ext.columns if c.startswith("PTI_")]
    keep_labs = [c for c in ext.columns if c.startswith("LAB_")]            # 23 retained base labs
    meta = [c for c in META_KEEP if c in ext.columns]
    base = ext[KEY_COLS + [c for c in meta if c not in KEY_COLS] + pti_cols].copy()

    # --- rebuild labs from raw: date-align, missingness, FORWARD-FILL ONLY ---
    daily, lab_feat = _raw_daily_labs()
    keep = [c for c in keep_labs if c in lab_feat]                          # same labs as baseline
    labels = pd.read_csv(os.path.join(LABELS_DIR, f"labels_{case}.csv"))
    labels["Date"] = pd.to_datetime(labels["Date"])
    m = labels[KEY_COLS + ["Date"]].merge(daily[["PID", "Date"] + keep], on=["PID", "Date"], how="left")
    m = m.sort_values(KEY_COLS).reset_index(drop=True)

    miss = m[keep].isna().astype(int)                                       # 1 = not measured that date
    miss.columns = [f"{c}_missing" for c in keep]
    m[keep] = m.groupby("PID")[keep].ffill()                               # forward-fill ONLY (no bfill)
    labs = pd.concat([m[KEY_COLS], m[keep], miss], axis=1)

    full = base.merge(labs, on=KEY_COLS, how="inner")
    if MAX_TREATMENT_DAY:                                  # cohort: first N treatment days only
        full = full[full["Day"] <= MAX_TREATMENT_DAY]
    out_dir = os.path.join(CLINICAL_ONLY, case)
    counts = write_splits(full, out_dir)                                    # v2 pid_splits_v2.json
    print(f"[build_clinical] {case}: rows={len(full)} feats={full.shape[1]-len(meta)} "
          f"({len(keep)} labs + {len(keep)} miss-ind + {len(pti_cols)} PTI) "
          f"-> v2 split {counts}  ({out_dir})")
    return out_dir


def build_clinical(cases=CASES):
    for case in cases:
        build_clinical_case(case)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    a = ap.parse_args()
    build_clinical(a.cases)
