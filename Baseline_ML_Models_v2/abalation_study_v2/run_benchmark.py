"""
run_benchmark.py
================
Classical benchmark across feature sets x cases x scope (per-day/agg) x 8 models,
evaluated with the resampled protocol (1:1 and 1:2, fixed positives, 100 iters).

Writes results/classical_benchmark.csv (one row per model config with
r1to1_* / r1to2_* mean±std and the full-test reference).

DL benchmark (aggregated 3-branch fusion) is driven separately once dl_data.py +
the training loop are wired; the model itself is in models/dl_aggregated.py.

Usage
-----
    conda activate crrt
    python run_benchmark.py                      # full classical matrix on current data
    python run_benchmark.py --feature-sets machine_only --cases 1day --scopes per_day --models RF GradientBoosting
"""

import os
import sys
import argparse
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "models"))
from abl_config import RESULTS_DIR, FEATURE_SETS, CASES                 # noqa: E402
from classical import run_matrix, build_model_registry                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--scopes", nargs="+", default=["per_day", "agg"],
                    choices=["per_day", "agg"])
    ap.add_argument("--models", nargs="+", default=list(build_model_registry().keys()))
    ap.add_argument("--variant", default="current", help="denoising variant (current/Th70/Th80/Th90)")
    ap.add_argument("--peak", default="denoised", choices=["denoised", "noised"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None,
                    help="defaults to results/classical_<variant>_<peak>.csv (per-peak, no collision)")
    a = ap.parse_args()

    if a.out is None:
        label = a.tag or f"{a.variant}_{a.peak}"
        a.out = os.path.join(RESULTS_DIR, f"classical_{label}.csv")
    scopes = [s == "agg" for s in a.scopes]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # run_matrix checkpoints to a.out after each config and resumes if it exists
    df = run_matrix(a.feature_sets, a.cases, scopes, a.models,
                    variant=a.variant, peak=a.peak, tag=a.tag, out=a.out)
    print(f"\n[run_benchmark] {a.out}  ({len(df)} rows total)")
    if len(df):
        cols = [c for c in ["feature_set", "scope", "case", "model", "machine_k",
                            "full_test_AUROC", "subpop_AUROC_mean", "subpop_AUROC_std",
                            "n_eval_subpops"] if c in df.columns]
        print(df.sort_values("full_test_AUROC", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
