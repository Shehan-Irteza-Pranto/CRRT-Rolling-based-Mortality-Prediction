"""
run_ablation.py
===============
Stage 4 of the study. After the main benchmark (current denoised+noised), this:

  1. PICKS the single best model per pipeline (machine_only, combined) on the
     DENOISED set by 1:1 AUROC, across classical AND deep learning.
  2. Th SWEEP: re-runs each winner on Th70/Th80/Th90 (denoised).
  3. LOO: re-runs each winner with the signal-processing leave-one-out
     (all3 / no_priming / no_downtime / no_peak), rebuilding features (and the DL
     sequence cache) per config from the cleaned pressure stream.

Outputs: results/ablation_th.csv , results/ablation_loo.csv
LOO needs Data/Intermediate/machine_filtered_v1.csv (export from the denoising
notebook); it is skipped with a message if absent.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DL_TRAIN = os.path.join(_HERE, "models", "dl_train.py")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "models"))
sys.path.insert(0, os.path.join(_HERE, "preprocessing"))

from abl_config import (RESULTS_DIR, CASES, DATA_DIR, LOO_CONFIGS, BEST_TH, BEST_DOWNTIME,  # noqa: E402
                        feature_dir, seq_cache_dir, abl_root, CLINICAL_ONLY)
from classical import run_one as classical_run_one                          # noqa: E402
from build_streams import clean_streams                                     # noqa: E402
from build_features import pressure_to_day_features, write_splits           # noqa: E402
from build_combined import build_combined_dir                               # noqa: E402
from build_agg import build_agg_dir                                         # noqa: E402
from build_sequences import build_sequence_cache                            # noqa: E402
from refresh_data import refresh_one                                        # noqa: E402

MFV1 = os.path.join(DATA_DIR, "Intermediate", "machine_filtered_v1.csv")
PTI  = os.path.join(DATA_DIR, "EHR", "Demographics", "pti_transformed_cleaned.csv")
METRIC = "full_test_AUROC"        # subpop protocol: pick winners by full-test AUROC


# ----------------------------------------------------------------------------- pick best
def pick_best(results_dir=RESULTS_DIR, metric=METRIC):
    frames = []
    for name in ("classical_current_denoised.csv", "dl_current_denoised.csv"):
        p = os.path.join(results_dir, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("[ablation] no denoised benchmark results found — run stages 2-3 first.")
    df = pd.concat(frames, ignore_index=True)
    winners = {}
    for fs in ("machine_only", "combined"):
        sub = df[(df.feature_set == fs) & df[metric].notna()]
        if sub.empty:
            print(f"[ablation] no rows for {fs}; skipping."); continue
        b = sub.loc[sub[metric].idxmax()]
        winners[fs] = {"fs": fs, "family": b["family"], "model": b["model"],
                       "scope": b["scope"], metric: round(float(b[metric]), 4)}
        print(f"[best] {fs}: {b['family']}/{b['model']}/{b['scope']}  {metric}={winners[fs][metric]}")
    return winners


# ----------------------------------------------------------------------------- run a winner
def _train_winner(w, variant, peak, case, tag, dl_seeds, dl_search):
    """Return a list of result rows for one winner on one (variant, peak, case)."""
    fs, scope = w["fs"], w["scope"]
    agg = (scope == "agg")
    if w["family"] == "classical":
        row = classical_run_one(fs, case, agg, w["model"], variant=variant, peak=peak, tag=tag)
        return [row]
    # deep learning — run in a SUBPROCESS (isolates the DL `config` from the classical one)
    rows_dir = os.path.join(RESULTS_DIR, "ablation_dl_rows"); os.makedirs(rows_dir, exist_ok=True)
    rows = []
    for s in range(dl_seeds):
        out = os.path.join(rows_dir, f"{tag}_{fs}_{scope}_{w['model']}_{case}_s{s}.csv")
        if not (os.path.exists(out) and os.path.getsize(out) > 0):    # resume
            subprocess.run([sys.executable, DL_TRAIN, "--pipeline", fs, "--scope", scope,
                            "--encoder", w["model"], "--case", case, "--seed", str(s),
                            "--variant", variant, "--peak", peak, "--search-iter", str(dl_search),
                            "--tag", tag, "--out", out], check=True)
        rows.append(pd.read_csv(out).iloc[0].to_dict())
    return rows


# ----------------------------------------------------------------------------- Th sweep
def th_sweep(winners, cases, dl_seeds, dl_search, out):
    rows = []
    for fs, w in winners.items():
        need_seq = (w["family"] == "deep")
        for th in (70, 80, 90):
            variant = f"Th{th}"
            for case in cases:
                refresh_one(variant, "denoised", case, skip_seq=not need_seq)
                for r in _train_winner(w, variant, "denoised", case, f"th{th}", dl_seeds, dl_search):
                    r["ablation"] = "th_sweep"; rows.append(r)
                    print(f"[th{th}] {fs}/{w['model']}/{case}: 1:1 AUROC "
                          f"{r.get('full_test_AUROC')}")
    if rows:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[ablation] wrote {out} ({len(rows)} rows)")
    return rows


# ----------------------------------------------------------------------------- LOO
def _weight_map():
    p = pd.read_csv(PTI, usecols=["PID", "CrrtInitiationWeight_kg"])
    p = p[p["CrrtInitiationWeight_kg"].gt(0)].dropna(subset=["CrrtInitiationWeight_kg"])
    return p.drop_duplicates("PID").set_index("PID")["CrrtInitiationWeight_kg"].to_dict()


def _build_loo_data(pressure, variant, peak, case, need_combined, need_seq):
    """Build the model-ready data for one LOO config (shared across winners)."""
    mdir = feature_dir(variant, peak, "machine_only", False, case)
    write_splits(pressure_to_day_features(pressure, case), mdir)
    build_agg_dir(mdir, feature_dir(variant, peak, "machine_only", True, case))
    if need_combined:
        cdir = feature_dir(variant, peak, "combined", False, case)
        build_combined_dir(mdir, case, cdir)
        build_agg_dir(cdir, feature_dir(variant, peak, "combined", True, case))
    build_agg_dir(os.path.join(CLINICAL_ONLY, case),
                  feature_dir(variant, peak, "clinical_only", True, case))
    if need_seq:
        pcsv = os.path.join(abl_root(variant, peak), f"pressure_{case}.csv")
        os.makedirs(os.path.dirname(pcsv), exist_ok=True)
        pressure.to_csv(pcsv, index=False)
        build_sequence_cache(pcsv, mdir, case, seq_cache_dir(variant, peak))


def loo_ablation(winners, cases, dl_seeds, dl_search, out):
    if not os.path.exists(MFV1):
        print(f"[loo] SKIP — {MFV1} not found. Export it from the denoising notebook:\n"
              f"      machine_filtered_v1.to_csv(os.path.join(parent_dir,'Intermediate','machine_filtered_v1.csv'), index=False)")
        return []
    mfv1 = pd.read_csv(MFV1)
    wmap = _weight_map()
    need_combined = any(w["fs"] == "combined" for w in winners.values())
    need_seq = any(w["family"] == "deep" for w in winners.values())
    rows = []
    for cfg, toggles in LOO_CONFIGS.items():
        variant, peak = f"loo_{cfg}", "x"
        # clean_streams (the heavy 800MB step) ONCE per config, shared across winners
        pressure = clean_streams(mfv1, downtime=BEST_DOWNTIME, Th=BEST_TH, weight_map=wmap,
                                 remove_priming=toggles["remove_priming"],
                                 remove_downtime=toggles["remove_downtime"],
                                 remove_peak=toggles["remove_peak"])
        for case in cases:
            _build_loo_data(pressure, variant, peak, case, need_combined, need_seq)
        for fs, w in winners.items():
            for case in cases:
                for r in _train_winner(w, variant, peak, case, f"loo_{cfg}", dl_seeds, dl_search):
                    r["ablation"] = "loo"; r["loo_config"] = cfg; rows.append(r)
                    print(f"[loo:{cfg}] {fs}/{w['model']}/{case}: 1:1 AUROC "
                          f"{r.get('full_test_AUROC')}")
    if rows:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[ablation] wrote {out} ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--dl-seeds", type=int, default=3)
    ap.add_argument("--dl-search-iter", type=int, default=5)
    ap.add_argument("--skip-th", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    a = ap.parse_args()

    print("=== pick best (machine_only, combined) on denoised by 1:1 AUROC ===")
    winners = pick_best()
    if not winners:
        raise SystemExit("[ablation] no winners selected.")
    if not a.skip_th:
        print("\n=== Th70/80/90 sweep ===")
        th_sweep(winners, a.cases, a.dl_seeds, a.dl_search_iter,
                 os.path.join(RESULTS_DIR, "ablation_th.csv"))
    if not a.skip_loo:
        print("\n=== signal-processing leave-one-out ===")
        loo_ablation(winners, a.cases, a.dl_seeds, a.dl_search_iter,
                     os.path.join(RESULTS_DIR, "ablation_loo.csv"))


if __name__ == "__main__":
    main()
