#!/usr/bin/env bash
# =============================================================================
# run_all.sh — run the study end to end, one command.
#
#   bash run_all.sh            # full run (resumes if interrupted)
#   bash run_all.sh --fresh    # wipe results/ and train from scratch
#   bash run_all.sh --quick    # fast end-to-end smoke (1 seed, 2 HP, 3 epochs)
#
# Saves every trained model: classical -> results/saved_models/classical/*.joblib,
# DL -> results/saved_models/dl/*.pt. DL loss/AUROC curves -> results/curves/*.png,
# per-epoch history -> results/dl_history/*.csv.
#
# Scope: train classical + DL on the CURRENT denoised & noised sets (all pipelines),
# then auto-pick the best model per pipeline (machine_only, combined) on denoised
# and ablate it with the Th70/80/90 sweep + signal-processing leave-one-out.
#
# RESUMABLE: if interrupted, just run it again — refresh_data skips built data,
# the classical benchmark skips configs already in its CSV, the DL launcher skips
# per-job CSVs. Use --force / --fresh to redo. Reads the existing denoising
# pressure CSVs (no re-denoising); models are pressure-only.
# =============================================================================
set -euo pipefail

PY=python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RES="$HERE/results"; mkdir -p "$RES"

# ---- scope (edit here if needed) -------------------------------------------
PEAKS=(denoised noised)
CASES=(1day 2day)
SEEDS=5 ; SEARCH_ITER=12 ; GPUS="0 1" ; EPOCH_FLAG=""
MODELS=(XGBoost LightGBM LogisticRegression RF GradientBoosting KNN DecisionTree AdaBoost)
DL_PIPELINES=(machine_only combined)
DL_ENCODERS=(cnn1d rnn transformer)
ABL_FLAGS="--dl-seeds 3 --dl-search-iter 5"
: "${ABL_NJOBS:=8}" ; export ABL_NJOBS     # classical HP-search worker cap (neighborly on shared node)

QUICK=0; FRESH=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1;;
    --fresh) FRESH=1;;
    *) echo "unknown arg: $arg"; exit 2;;
  esac
done

if [[ $FRESH -eq 1 ]]; then
  echo "[run_all] FRESH: clearing results/ (models, curves, CSVs, logs) to retrain from scratch."
  rm -rf "$RES"; mkdir -p "$RES"
fi

if [[ $QUICK -eq 1 ]]; then
  echo "[run_all] QUICK mode: reduced budget for a fast end-to-end test."
  PEAKS=(denoised); CASES=(1day)
  MODELS=(LogisticRegression RF)
  DL_PIPELINES=(machine_only); DL_ENCODERS=(cnn1d)
  SEEDS=1; SEARCH_ITER=2; EPOCH_FLAG="--epochs 3"
  ABL_FLAGS="--dl-seeds 1 --dl-search-iter 1"
fi

banner() { echo; echo "=============================================================="; \
           echo "[$(date +%H:%M:%S)] $*"; echo "=============================================================="; }

# ---- Stage 1: data (features + DL sequence cache) for CURRENT --------------
banner "STAGE 1/5  Build features + sequence cache (current, denoised+noised)"
$PY refresh_data.py --variants current --peaks "${PEAKS[@]}" --cases "${CASES[@]}"

# ---- Stage 2: classical benchmark (current x peaks) ------------------------
banner "STAGE 2/5  Classical benchmark (8 models x clinical/machine/combined x per-day+agg)"
for p in "${PEAKS[@]}"; do
  echo "[classical] variant=current peak=$p"
  $PY run_benchmark.py --variant current --peak "$p" --cases "${CASES[@]}" \
      --scopes per_day agg --models "${MODELS[@]}" --out "$RES/classical_current_${p}.csv"
done

# ---- Stage 3: deep-learning benchmark (current, both GPUs) -----------------
banner "STAGE 3/5  Deep-learning benchmark (machine_only + combined, ${SEEDS} seeds x ${SEARCH_ITER} HP)"
for p in "${PEAKS[@]}"; do
  echo "[dl] variant=current peak=$p"
  $PY run_dl_benchmark.py --gpus $GPUS --variant current --peak "$p" \
      --pipelines "${DL_PIPELINES[@]}" --encoders "${DL_ENCODERS[@]}" \
      --cases "${CASES[@]}" --seeds "$SEEDS" --search-iter "$SEARCH_ITER" $EPOCH_FLAG \
      --out "$RES/dl_current_${p}.csv"
done

# ---- Stage 4: pick best per pipeline, then ablate (Th sweep + LOO) ----------
banner "STAGE 4/5  Ablation on the best machine_only & combined models (Th70/80/90 + LOO)"
$PY run_ablation.py --cases "${CASES[@]}" $ABL_FLAGS

# ---- Stage 5: merge result tables ------------------------------------------
banner "STAGE 5/5  Merge per-config CSVs into master tables"
RES="$RES" $PY - <<'PYMERGE'
import glob, os, pandas as pd
RES = os.environ["RES"]
for prefix, out in [("classical_", "ALL_classical.csv"), ("dl_", "ALL_dl.csv")]:
    files = sorted(f for f in glob.glob(os.path.join(RES, prefix + "*.csv"))
                   if os.path.basename(f) not in ("ALL_classical.csv", "ALL_dl.csv"))
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(RES, out), index=False)
        print(f"[merge] {out}: from {len(files)} files")
    else:
        print(f"[merge] no {prefix}*.csv to merge")
PYMERGE

banner "DONE.  Tables: results/ALL_classical.csv , results/ALL_dl.csv , results/ablation_th.csv , results/ablation_loo.csv"
