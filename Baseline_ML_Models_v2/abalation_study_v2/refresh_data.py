"""
refresh_data.py
===============
Regenerate ALL model-ready artifacts for the Abalation_study from the denoising
pressure CSVs — in one command, entirely inside this folder. Run this whenever the
denoising notebook produces new pressure CSVs.

Per (variant, peak, case) it builds, under <Model_Ready>/abalation/<variant>_<peak>/:
  machine_only/{case}        52-stat per-day machine features  (from the pressure CSV)
  combined/{case}            machine + stable clinical features
  machine_only_agg/{case}    expanding-day aggregation of machine
  combined_agg/{case}        expanding-day aggregation of combined
  clinical_only_agg/{case}   expanding-day aggregation of the stable clinical table
  seq_cache/{case}_*.npz     DL [T,4] tensors (from the pressure CSV)
clinical_only (per-day) is denoising-independent -> read from the stable table.

Usage
-----
    python refresh_data.py                                   # current denoised+noised, 1day+2day
    python refresh_data.py --variants current Th70 Th80 Th90 --peaks denoised noised
    python refresh_data.py --skip-seq                        # tables only (no DL cache)
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "preprocessing"))
from abl_config import (DENOISE_VARIANTS, CASES, abl_root, feature_dir,           # noqa: E402
                        seq_cache_dir, CLINICAL_ONLY)
from build_features import pressure_to_day_features, write_splits                 # noqa: E402
from build_combined import build_combined_dir                                     # noqa: E402
from build_agg import build_agg_dir                                               # noqa: E402
from build_sequences import build_sequence_cache                                  # noqa: E402
from build_clinical import build_clinical_case                                    # noqa: E402


def _have_tables(d):
    return all(os.path.exists(os.path.join(d, f"{s}.csv")) for s in ("train", "val", "test"))


def refresh_one(variant, peak, case, skip_seq=False, force=False):
    src = DENOISE_VARIANTS[variant][peak]
    if not os.path.exists(src):
        print(f"[refresh] SKIP {variant}/{peak}: missing pressure CSV {src}")
        return
    print(f"\n[refresh] === {variant}/{peak}/{case} ===  ({src})")

    # 0) clinical per-day RE-SPLIT onto the v2 split (peak-agnostic; needed before combined)
    clin_dir = os.path.join(CLINICAL_ONLY, case)
    if force or not _have_tables(clin_dir):
        build_clinical_case(case)

    # 1) machine_only per-day (from pressure)
    mach_dir = feature_dir(variant, peak, "machine_only", False, case)
    if force or not _have_tables(mach_dir):
        write_splits(pressure_to_day_features(src, case), mach_dir)
        print(f"  machine_only -> {mach_dir}")
    else:
        print(f"  machine_only -> exists, skip")

    # 2) combined per-day (machine + stable clinical)
    comb_dir = feature_dir(variant, peak, "combined", False, case)
    if force or not _have_tables(comb_dir):
        build_combined_dir(mach_dir, case, comb_dir)
        print(f"  combined     -> {comb_dir}")

    # 3) aggregated versions
    for label, in_dir, out_dir in [
        ("machine_agg", mach_dir, feature_dir(variant, peak, "machine_only", True, case)),
        ("combined_agg", comb_dir, feature_dir(variant, peak, "combined", True, case)),
        ("clinical_agg", os.path.join(CLINICAL_ONLY, case),
         feature_dir(variant, peak, "clinical_only", True, case))]:
        if force or not _have_tables(out_dir):
            build_agg_dir(in_dir, out_dir)
    print("  agg (machine, combined, clinical) ready")

    # 4) DL sequence cache (from pressure)
    if not skip_seq:
        scd = seq_cache_dir(variant, peak)
        if force or not os.path.exists(os.path.join(scd, f"{case}_test.npz")):
            build_sequence_cache(src, mach_dir, case, scd)
        else:
            print("  seq_cache    -> exists, skip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["current"])
    ap.add_argument("--peaks", nargs="+", default=["denoised", "noised"])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--skip-seq", action="store_true", help="skip the DL .npz cache (tables only)")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs already exist")
    a = ap.parse_args()
    for v in a.variants:
        for p in a.peaks:
            for c in a.cases:
                refresh_one(v, p, c, skip_seq=a.skip_seq, force=a.force)
    print("\n[refresh] done. Roots under:", abl_root(a.variants[0], a.peaks[0]), "...")


if __name__ == "__main__":
    main()
