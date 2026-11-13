"""
config.py
=========
Central configuration for the Deep Learning half of the CRRT-on-AKI mortality
benchmark. Mirrors the conventions of the classical pipelines
(../Machine_Only_Models/config.py etc.) so the DL comparison cannot silently
drift from the classical one.

Key data decision: sequence models consume the RAW per-minute pressure streams
shaped [T, 4] (access / filter / effluent / return) per circuit-day — NOT the 52
engineered summary features. Split membership and labels are inherited verbatim
from the classical Model_Ready CSVs (one row == one (PID, Visit_no, Day)).
"""

import os

# --- Directory paths ---
_HERE     = os.path.dirname(os.path.abspath(__file__))
_V2_DIR   = os.path.dirname(_HERE)                 # Baseline_ML_Models_v2/
_PROJ_DIR = os.path.dirname(_V2_DIR)               # CRRTnet_Data_Prep/

DATA_DIR        = os.path.join(_PROJ_DIR, "Data", "Processed")
MODEL_READY_DIR = os.path.join(DATA_DIR, "Model_Ready")      # classical splits (defines membership)
PRESSURE_DIR    = os.path.join(DATA_DIR, "Machine", "pressure")

# SEQ_CACHE_DIR and RESULTS_DIR are env-overridable so the downsampling experiment can
# point training at a downsampled cache variant and an isolated results folder without
# any code change (see data/prepare_downsampled.py, training/downsample_experiment.py).
SEQ_CACHE_DIR   = os.environ.get("CRRT_SEQ_CACHE_DIR") or os.path.join(_HERE, "data", "sequence_cache")

RESULTS_DIR = os.environ.get("CRRT_RESULTS_DIR") or os.path.join(_HERE, "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "best_models")          # saved best checkpoints per seed
SALIENCY_DIR = os.path.join(RESULTS_DIR, "saliency")
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")
REPORT_DIR  = os.path.join(RESULTS_DIR, "reports")
HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")     # per-epoch train/val loss + AUROC
PRED_DIR    = os.path.join(RESULTS_DIR, "predictions")          # saved test (y_true, y_score) per seed
CURVES_DIR  = os.path.join(RESULTS_DIR, "curves")               # loss / AUROC / PR curve figures

# Classical balanced-threshold results (6 pipelines x 2 cases), used by
# evaluate.py to build the classical-vs-deep master comparison table.
CLASSICAL_METRICS_JSON = os.path.join(_V2_DIR, "results_balanced_threshold", "balanced_f1_metrics.json")

# --- Raw stream definition ---
# 4 channels, in fixed order. This order is the [T, 4] channel axis everywhere.
CHANNELS = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE"]
N_CHANNELS = len(CHANNELS)
KEY_COLS   = ["PID", "Visit_no", "Day"]          # 1:1 join key between streams and labels
TIME_COL   = "Time"

STREAM_FILES = {                                  # which raw CSV backs each variant
    "denoised": os.path.join(PRESSURE_DIR, "Denoised", "CRRT_pressure_wo_peak.csv"),
    "noised":   os.path.join(PRESSURE_DIR, "Noised",   "CRRT_pressure_with_peak.csv"),
}
DEFAULT_STREAM = "denoised"                       # match the classical denoised splits

# --- Experiment dimensions (kept identical to the classical matrix) ---
# Focused on 1-day prediction only. (2day dropped; add it back here to re-enable.)
CASES          = ["1day"]
TARGET_COL     = "label"
META_COLS      = ["PID", "Visit_no", "Day", "Date", "Death_Date",
                  "Mortality", "days_until_death", "label"]

# Where each branch's tabular features come from (for MLP rungs + fusion clinical branch)
TABULAR_SOURCES = {
    "clinical": os.path.join(MODEL_READY_DIR, "clinical_only"),       # {case}/{train,val,test}.csv
    "machine":  os.path.join(MODEL_READY_DIR),                        # {case}/denoised/{...}.csv  (52 feats)
    "combined": os.path.join(MODEL_READY_DIR, "machine_clinical"),    # {case}/{train,val,test}.csv
}

# --- Sequence windowing ---
# Raw streams are ~1 sample/minute, up to ~1440/day. Pad-and-mask to MAX_LEN; longer
# windows are right-cropped. Set MAX_LEN from the data after inspecting length percentiles.
MAX_LEN       = 1440          # minutes in a day; revisit against the 95th pct of observed lengths
PAD_VALUE     = 0.0           # post-normalization pad value (masked out, so value is inert)
CROP_SIDE     = "right"       # keep the start of the window if longer than MAX_LEN

# --- Reproducibility ---
SEEDS = [0, 1, 2, 3, 4]       # >= 5 seeds; report mean +/- std. Non-negotiable for credibility.
RANDOM_STATE = 42             # matches the classical pipeline's split/balance RNG

# --- Class imbalance ---
LOSS          = "weighted_bce"   # one of: "weighted_bce", "focal"
FOCAL_GAMMA   = 2.0
EARLY_STOP_METRIC = "val_auroc"  # early-stop / model selection signal
EARLY_STOP_PATIENCE = 30

# --- Optimization / regularization defaults (anti-overfitting) ---
# These were hardcoded (lr=1e-3, wd=1e-4) and the rungs overfit within a few epochs
# (train loss -> 0, val loss climbing). Lower LR + stronger weight decay + an adaptive
# LR schedule + gradient clipping flatten the val curve. Any of these can be overridden
# per run via --hp '{"lr": ..., "weight_decay": ...}'.
LR            = 3e-4             # default AdamW learning rate (was 1e-3)
WEIGHT_DECAY  = 1e-3             # default L2 (was 1e-4) — 10x stronger
GRAD_CLIP     = 1.0             # max grad-norm; <=0 disables clipping
USE_LR_SCHEDULER  = True         # ReduceLROnPlateau on val AUROC
LR_SCHED_FACTOR   = 0.5          # multiply LR by this on plateau
LR_SCHED_PATIENCE = 5            # epochs without val-AUROC improvement before LR drop
MIN_LR            = 1e-6         # floor for the scheduler

# --- Fair tuning budget (comparable to GBM: 30-iter RandomizedSearch x 5-fold) ---
# Documented search space. search.py samples N_SEARCH_ITER configs per model.
N_SEARCH_ITER = 30
MAX_EPOCHS    = 150
SEARCH_SPACE = {
    # biased toward regularization: dropped the overfit-prone 3e-3 LR and the
    # un-regularized 0.0 dropout / 0.0-1e-5 weight-decay corners; added 1e-2 wd.
    "lr":           [1e-4, 3e-4, 5e-4],
    "batch_size":   [32, 64],
    "dropout":      [0.2, 0.3, 0.5],          # added 0.5 — the transformer overfits, needs it
    "weight_decay": [1e-3, 1e-2],
    # per-architecture width/depth knobs:
    "cnn1d":       {"channels": [32, 64, 128], "n_blocks": [2, 3, 4], "kernel": [3, 5, 7]},
    "rnn":         {"hidden": [64, 128, 256], "layers": [1, 2], "cell": ["lstm", "gru"]},
    # transformer: added smaller-capacity corners (d_model 32, n_layers 1) so the search can
    # shrink the model that overfits worst.
    "transformer": {"d_model": [32, 64, 128], "n_heads": [4, 8], "n_layers": [1, 2, 3], "ff": [128, 256]},
    "mlp":         {"hidden": [[128], [256, 128], [256, 128, 64]]},
    "fusion":      {"seq_encoder": ["cnn1d", "rnn", "transformer"], "fusion_dim": [64, 128]},
}

# --- Evaluation ---
N_BOOTSTRAP = 1000            # bootstrap resamples for test-metric CIs
BOOTSTRAP_CI = 0.95
