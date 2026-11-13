"""
exp_2/ablation/run_ablation.py
==============================
Ablation study for the BEST MACHINE-ONLY models (from exp_2):
    1day -> DecisionTree (per_day),   2day -> RF (per_day)

Two axes, machine_only only, on the Day<=7 / 70:10:20 cohort, full metric panel
(imbalanced + 1:1 + 1:2 + subpopulation, AUROC/AUPRC/F1/F2/F3/Precision/Recall):

  A. DENOISE VARIANTS : current / Th70 / Th80 / Th90  x  denoised / noised
  B. LOO CONFIGS      : all3 / no_priming / no_downtime / no_peak  (rebuilt from
                        machine_filtered_v1.csv via the priming/downtime/peak toggles)

Machine features for every variant are rebuilt INSIDE this folder (data/abalation),
so the parent and exp_2 stay untouched. Reuses the parent's 70:10:20 split + clinical
table (for the subpopulation clustering). Resume-safe + incremental CSV.

Usage:  python run_ablation.py                 # both axes, both cases
        python run_ablation.py --skip-loo      # denoise variants only
"""
import os
import sys
import argparse
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "models"))
sys.path.insert(0, os.path.join(_HERE, "eval"))
sys.path.insert(0, os.path.join(_HERE, "preprocessing"))

from abl_config import (DENOISE_VARIANTS, LOO_CONFIGS, CASES, feature_dir,      # noqa: E402
                        DATA_DIR, RESULTS_DIR, BEST_DOWNTIME, BEST_TH, MAX_TREATMENT_DAY)
from classical import run_one                                                  # noqa: E402
from build_features import pressure_to_day_features, write_splits              # noqa: E402
from build_streams import clean_streams                                        # noqa: E402

# Best machine-only model per case (from exp_2, per_day scope).
BEST_MACHINE = {"1day": "DecisionTree", "2day": "RF"}
DENOISE_VARS = ["current", "Th70", "Th80", "Th90"]
PEAKS        = ["denoised", "noised"]
MFV1 = os.path.join(DATA_DIR, "Intermediate", "machine_filtered_v1.csv")
PTI  = os.path.join(DATA_DIR, "EHR", "Demographics", "pti_transformed_cleaned.csv")
OUT  = os.path.join(RESULTS_DIR, "ablation_machine_only.csv")


def _have(d):
    return all(os.path.exists(os.path.join(d, f"{s}.csv")) for s in ("train", "val", "test"))


def _build_machine(pressure, variant, peak, case):
    """Build machine_only per-day features (Day<=7) for one (variant, peak, case)."""
    mdir = feature_dir(variant, peak, "machine_only", False, case)
    if not _have(mdir):
        write_splits(pressure_to_day_features(pressure, case), mdir)
    return mdir


def _weight_map():
    p = pd.read_csv(PTI, usecols=["PID", "CrrtInitiationWeight_kg"])
    p = p[p["CrrtInitiationWeight_kg"].gt(0)].dropna(subset=["CrrtInitiationWeight_kg"])
    return p.drop_duplicates("PID").set_index("PID")["CrrtInitiationWeight_kg"].to_dict()


def _done(out):
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        return set(), []
    df = pd.read_csv(out)
    keys = {(r.variant, r.case) for r in df.itertuples()}
    return keys, df.to_dict("records")


def _run_one(variant, peak, case, ablation, loo_cfg, acc, done):
    tag = f"{variant}_{peak}" if peak != "x" else variant
    if (tag, case) in done:
        print(f"[skip-done] {tag}/{case}"); return
    model = BEST_MACHINE[case]
    row = run_one("machine_only", case, False, model, variant=variant, peak=peak, tag=tag)
    row.update({"ablation": ablation, "loo_config": loo_cfg or "",
                "denoise_variant": variant, "peak": peak})
    acc.append(row); done.add((tag, case))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(acc).to_csv(OUT, index=False)
    print(f"[ok] {ablation}:{tag}/{case}/{model}: full AUROC {row['full_test_AUROC']:.3f} "
          f"AUPRC {row['full_test_AUPRC']:.3f} | subpop {row['subpop_AUROC_mean']}±{row['subpop_AUROC_std']}")


def denoise_axis(cases, acc, done):
    print("\n=== A) DENOISE VARIANTS (current/Th70/Th80/Th90 x denoised/noised) ===")
    for variant in DENOISE_VARS:
        for peak in PEAKS:
            src = DENOISE_VARIANTS[variant][peak]
            if not os.path.exists(src):
                print(f"[skip] {variant}/{peak}: missing {src}"); continue
            for case in cases:
                _build_machine(src, variant, peak, case)
                _run_one(variant, peak, case, "denoise", None, acc, done)


def loo_axis(cases, acc, done):
    if not os.path.exists(MFV1):
        print(f"\n[loo] SKIP — {MFV1} not found."); return
    print("\n=== B) LOO CONFIGS (all3/no_priming/no_downtime/no_peak) ===")
    mfv1 = pd.read_csv(MFV1)
    wmap = _weight_map()
    for cfg, tog in LOO_CONFIGS.items():
        variant, peak = f"loo_{cfg}", "x"
        if all((variant, c) in done for c in cases):
            print(f"[skip-done] {variant} (all cases)"); continue
        pressure = clean_streams(mfv1, downtime=BEST_DOWNTIME, Th=BEST_TH, weight_map=wmap,
                                 remove_priming=tog["remove_priming"],
                                 remove_downtime=tog["remove_downtime"],
                                 remove_peak=tog["remove_peak"])
        for case in cases:
            _build_machine(pressure, variant, peak, case)
            _run_one(variant, peak, case, "loo", cfg, acc, done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--skip-denoise", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    a = ap.parse_args()
    print(f"[ablation] machine_only best models {BEST_MACHINE}  | cohort Day<={MAX_TREATMENT_DAY}")
    done, acc = _done(OUT)
    if done:
        print(f"[resume] {len(done)} (variant,case) already done -> skipping")
    if not a.skip_denoise:
        denoise_axis(a.cases, acc, done)
    if not a.skip_loo:
        loo_axis(a.cases, acc, done)
    print(f"\n[ablation] wrote {OUT} ({len(acc)} rows)")


if __name__ == "__main__":
    main()
