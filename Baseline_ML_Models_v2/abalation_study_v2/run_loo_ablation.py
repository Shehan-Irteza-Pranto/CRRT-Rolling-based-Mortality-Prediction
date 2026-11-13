"""
run_loo_ablation.py
===================
Preprocessing leave-one-out on the best model + best pipeline.

For each of {all3, no_priming, no_downtime, no_peak} (abl_config.LOO_CONFIGS) it:
  1. clean_streams(machine_filtered_v1, **toggles)  -> per-minute pressure
  2. build the 52-stat machine table + fixed splits
  3. train the best model and evaluate with the resampled 1:1 / 1:2 protocol
isolating each preprocessing step's contribution. -> results/loo_ablation.csv

Prereq: export the denoising notebook's priming-removal INPUT once. In the notebook
(parent_dir = .../Data) add and run:
    machine_filtered_v1.to_csv(os.path.join(parent_dir, "Intermediate", "machine_filtered_v1.csv"), index=False)
-> lands at Data/Intermediate/machine_filtered_v1.csv (next to extracted_raw.csv).
The PTI weight table is already at Data/EHR/Demographics/pti_transformed_cleaned.csv.

Currently wired for the machine_only feature set (preprocessing changes the machine
features); combined extends this by re-merging clinical onto the regenerated table.
"""

import os
import sys
import argparse
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "preprocessing"))
sys.path.insert(0, os.path.join(_HERE, "models"))
from abl_config import (LOO_CONFIGS, MODEL_READY_DIR, RESULTS_DIR, PROCESSED_DIR,   # noqa: E402
                        DATA_DIR, BEST_DOWNTIME, BEST_TH, BEST_CASE, BEST_MODEL)
from build_streams import clean_streams                                     # noqa: E402
from build_features import pressure_to_day_features, write_splits          # noqa: E402
from classical import run_one as classical_run_one                         # noqa: E402

DEFAULT_MFV1 = os.path.join(DATA_DIR, "Intermediate", "machine_filtered_v1.csv")
PTI = os.path.join(DATA_DIR, "EHR", "Demographics", "pti_transformed_cleaned.csv")


def _weight_map():
    p = pd.read_csv(PTI, usecols=["PID", "CrrtInitiationWeight_kg"])
    p = p[p["CrrtInitiationWeight_kg"].gt(0)].dropna(subset=["CrrtInitiationWeight_kg"])
    return p.drop_duplicates("PID").set_index("PID")["CrrtInitiationWeight_kg"].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfv1", default=DEFAULT_MFV1, help="exported machine_filtered_v1.csv")
    ap.add_argument("--case", default=BEST_CASE)
    ap.add_argument("--model", default=BEST_MODEL)
    ap.add_argument("--configs", nargs="+", default=list(LOO_CONFIGS.keys()))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "loo_ablation.csv"))
    a = ap.parse_args()

    if not os.path.exists(a.mfv1):
        print(f"[loo] missing {a.mfv1}\n  Export it from the denoising notebook first:\n"
              f"  machine_filtered_v1.to_csv(r'{a.mfv1}', index=False)")
        sys.exit(1)

    mfv1 = pd.read_csv(a.mfv1)
    wmap = _weight_map()
    rows = []
    for cfg in a.configs:
        toggles = LOO_CONFIGS[cfg]
        print(f"\n[loo] === {cfg} === {toggles}")
        pressure = clean_streams(mfv1, downtime=BEST_DOWNTIME, Th=BEST_TH, weight_map=wmap,
                                 remove_priming=toggles["remove_priming"],
                                 remove_downtime=toggles["remove_downtime"],
                                 remove_peak=toggles["remove_peak"])
        table = pressure_to_day_features(pressure, a.case)
        ddir = os.path.join(MODEL_READY_DIR, "abalation", f"loo_{cfg}", a.case, "denoised")
        write_splits(table, ddir)
        row = classical_run_one("machine_only", a.case, agg=False, model_name=a.model,
                                data_dir=ddir, tag=f"loo_{cfg}")
        row["loo_config"] = cfg
        rows.append(row)
        print(f"[loo] {cfg}: full AUROC {row['full_test_AUROC']:.3f} | subpop {row['subpop_AUROC_mean']}±{row['subpop_AUROC_std']} "
              f"| 1:2 AUROC {row['r1to2_AUROC_mean']:.3f}±{row['r1to2_AUROC_std']:.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n[loo] wrote {a.out}")
    cols = ["loo_config", "model", "full_test_AUROC", "subpop_AUROC_mean", "subpop_AUROC_std",
            "r1to2_AUROC_mean", "r1to2_AUROC_std"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
