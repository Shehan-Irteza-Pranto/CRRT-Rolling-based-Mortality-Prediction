"""
preprocessing/build_splits.py  (abalation_study_v2)
===================================================
Build v2's OWN patient-level train/val/test split = 70 : 10 : 20.

- Split is on PID (a patient's circuit-days never straddle splits -> no leakage).
- Stratified so the POSITIVE RATE is similar across splits: the stratum is each
  patient's number of positive 1-day circuit-days, capped to {0, 1, 2+}. Because all
  positives come from these patients, balancing this balances the circuit-day positive
  rate for BOTH the 1day and 2day labels.

Writes data/splits/pid_splits_v2.json = {"train":[...], "val":[...], "test":[...]}.
abl_config.SPLITS_JSON points here, so refresh_data.py rebuilds every table/seq cache
against this split.

Usage:  python build_splits.py            # 75/5/20, seed = RANDOM_STATE
        python build_splits.py --test 0.2 --val 0.05
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import LABELS_DIR, SPLITS_JSON, RANDOM_STATE, CASES, MAX_TREATMENT_DAY  # noqa: E402


def _patient_strata():
    """Per-PID stratification label from the 1day positive-day count, capped to {0,1,2+}."""
    lab = pd.read_csv(os.path.join(LABELS_DIR, "labels_1day.csv"))
    if MAX_TREATMENT_DAY:                                  # match the first-N-days cohort
        lab = lab[lab["Day"] <= MAX_TREATMENT_DAY]
    pos = lab.groupby("PID")["label_1day"].sum()
    strat = pos.clip(upper=2).astype(int)              # 0, 1, 2+
    return strat.index.to_numpy(), strat.to_numpy()


def build_splits(out_json=None, test=0.20, val=0.10, seed=RANDOM_STATE):
    out_json = out_json or SPLITS_JSON
    pids, strat = _patient_strata()
    # 1) carve out the test fraction, stratified
    tr_pids, te_pids, tr_s, _ = train_test_split(
        pids, strat, test_size=test, random_state=seed, stratify=strat)
    # 2) carve val out of the remainder so val is `val` of the WHOLE -> val/(1-test)
    va_frac = val / (1.0 - test)
    tr_pids, va_pids = train_test_split(
        tr_pids, test_size=va_frac, random_state=seed, stratify=tr_s)
    splits = {"train": sorted(int(p) for p in tr_pids),
              "val":   sorted(int(p) for p in va_pids),
              "test":  sorted(int(p) for p in te_pids)}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(splits, f, indent=2)
    _report(splits, out_json)
    return splits


def _report(splits, out_json):
    n = {k: len(v) for k, v in splits.items()}
    tot = sum(n.values())
    print(f"[build_splits] PIDs  train={n['train']} val={n['val']} test={n['test']}  "
          f"({n['train']/tot:.0%}/{n['val']/tot:.0%}/{n['test']/tot:.0%})  -> {out_json}")
    pid2split = {p: s for s, ps in splits.items() for p in ps}
    for case in CASES:
        lab = pd.read_csv(os.path.join(LABELS_DIR, f"labels_{case}.csv"))
        lab["split"] = lab["PID"].map(pid2split)
        print(f"  {case}: " + "  ".join(
            f"{s} n={int((lab.split==s).sum())} pos={lab[lab.split==s][f'label_{case}'].mean():.3f}"
            for s in ["train", "val", "test"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=SPLITS_JSON)
    ap.add_argument("--test", type=float, default=0.20)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    a = ap.parse_args()
    build_splits(a.out, a.test, a.val, a.seed)


if __name__ == "__main__":
    main()
