#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — end-to-end CRRT-on-AKI deep-learning pipeline (1-day focus)
#
#   Stage 1  data processing : raw pressure streams -> full-res [T,4] tensors
#   Stage 2  full-res training+eval : all 8 rungs, 5 seeds, 2 GPUs, report + curves
#   Stage 3  downsampling : build the 8 downsampled caches (10/15 min, mean/median, 0/50% overlap)
#   Stage 4  downsampled training+eval+compare : sequence/fusion rungs per variant + cross-variant table
#
# Each stage writes to its own folder; evaluation (aggregate + curves) is built into
# stages 2 and 4. Run stages individually by passing their numbers, e.g.:
#     ./run_pipeline.sh 3 4        # only build downsampled caches and run that study
#     ./run_pipeline.sh            # all stages
# =============================================================================
set -euo pipefail

PY=python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPUS="0,1"            # set to "0" for single-GPU; affects stages 2 and 4
STREAM="denoised"     # matches the classical denoised splits

STAGES="${*:-1 2 3 4}"
run() { echo " + $*"; "$@"; }

if [[ " $STAGES " == *" 1 "* ]]; then
  echo "================ Stage 1: data processing (full-res sequences) ================"
  run "$PY" data/prepare_sequences.py --stream "$STREAM"
fi

if [[ " $STAGES " == *" 2 "* ]]; then
  echo "================ Stage 2: full-resolution training + evaluation ================"
  # clean per-seed artifacts so the run is fresh, then train all rungs + aggregate + plot
  rm -f results/best_models/*.pt results/predictions/*.npz results/curves/*.png results/logs/*.log
  run "$PY" -u training/run_all.py --gpus "$GPUS"
fi

if [[ " $STAGES " == *" 3 "* ]]; then
  echo "================ Stage 3: build downsampled caches (8 variants) ================"
  run "$PY" data/prepare_downsampled.py --stream "$STREAM"
fi

if [[ " $STAGES " == *" 4 "* ]]; then
  echo "================ Stage 4: downsampled training + evaluation + comparison ======="
  # caches already built in stage 3 -> --skip-build
  run "$PY" -u training/downsample_experiment.py --gpus "$GPUS" --stream "$STREAM" --skip-build
fi

echo "================ pipeline complete ================"
echo "Full-res report   : results/reports/dl_vs_classical.md"
echo "Full-res curves   : results/curves/"
echo "Downsampled study : results_downsampled/COMPARISON.md"
echo "Per-variant output: results_downsampled/<variant>/{reports,curves,best_models}/"
