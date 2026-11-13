"""
config.py — abalation_study_v2
==============================
Central config for the ENRICHED-FEATURE preprocessing + resampled-evaluation study.

v2 differs from Abalation_study only in the machine feature representation: in
addition to the 13 marginal-distribution stats per raw pressure channel, it adds
(a) cross-channel CRRT physiology channels — transmembrane pressure (TMP), filter
pressure drop (PFD), access-return span (ARD) — each summarized with the same 13
stats, and (b) within-day temporal-dynamics stats (slope/trend/volatility/excursions)
for every channel. Everything else (splits, resampled-eval protocol, models, mRMR)
is identical, so test sets remain comparable to the baseline.

Processed data is written INSIDE this folder (./data) so the study is self-contained
and never touches the baseline Model_Ready tables.
"""

import os

# --- Directory paths ---
_HERE     = os.path.dirname(os.path.abspath(__file__))
_V2_DIR   = os.path.dirname(_HERE)                       # Baseline_ML_Models_v2/
_PROJ_DIR = os.path.dirname(_V2_DIR)                     # CRRTnet_Data_Prep/

DATA_DIR        = os.path.join(_PROJ_DIR, "Data")
PROCESSED_DIR   = os.path.join(DATA_DIR, "Processed")
MODEL_READY_DIR = os.path.join(PROCESSED_DIR, "Model_Ready")
PRESSURE_DIR    = os.path.join(PROCESSED_DIR, "Machine", "pressure")
ABLATION_DOWNTIME_DIR = os.path.join(PROCESSED_DIR, "Machine", "ablation_downtime")
LABELS_DIR      = os.path.join(PROCESSED_DIR, "Labels")
# v2 uses its OWN stratified 75:5:20 patient-level split (built by preprocessing/build_splits.py).
SPLITS_JSON     = os.path.join(_HERE, "data", "splits", "pid_splits_v2.json")

# Sibling pipelines we import from (added to sys.path by importers).
MACHINE_CLINICAL_DIR = os.path.join(_V2_DIR, "Machine_Clinical_Models")
MACHINE_ONLY_DIR     = os.path.join(_V2_DIR, "Machine_Only_Models")
CLINICAL_ONLY_DIR    = os.path.join(_V2_DIR, "Clinical_Only_Models")
DL_DIR               = os.path.join(_V2_DIR, "Deep_Learning_Models")

RESULTS_DIR = os.path.join(_HERE, "results")
ABL_STREAMS_DIR  = os.path.join(PROCESSED_DIR, "Machine", "abalation_study")  # generated variants land here

# --- Reproducibility ---
RANDOM_STATE = 42

# --- Evaluation protocol ------------------------------------------------------
# "both"     : full-test (imbalanced) + resampled 1:1/1:2 + per-subpopulation mean+/-std.
# "subpop"   : full-test + per-subpopulation std only.
# "resampled": full-test + resampled 1:1/1:2 only.
# Every block reports AUROC, F1, F2, F3, Precision, Recall (+ AUPRC etc.).
EVAL_PROTOCOL  = "both"
SUBPOP_N       = 4           # number of clinical subpopulations (Ward maxclust)
SUBPOP_MIN_POS = 10          # a subpop needs >= this many positives to be evaluable (EPV rule)

# Resampled protocol params (used only when EVAL_PROTOCOL == "resampled")
RESAMPLE_RATIOS = (1, 2)     # negatives : positives  -> 1:1 and 1:2
N_ITER          = 100        # resampling iterations
SEED0           = 0          # iteration i uses RandomState(SEED0 + i); shared across all models -> paired

# --- Column conventions (match the model pipelines) ---
TARGET_COL = "label"
KEY_COLS   = ["PID", "Visit_no", "Day"]
CHANNELS   = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE"]
STAT_NAMES = ["mean", "std", "min", "max", "median", "q25", "q75",
              "iqr", "range", "skewness", "kurtosis", "cv", "n_valid"]

# --- v2 enrichment: derived CRRT-physiology channels (computed per timestep) -----
# TMP = (FILTER + RETURN)/2 - EFFLUENT  (transmembrane pressure; membrane fouling)
# PFD = FILTER - RETURN                 (pressure drop across the filter; clotting)
# ARD = RETURN - ACCESS                 (overall circuit pressure span / access instability)
# Each derived channel is summarized with the same STAT_NAMES, and ALL 7 channels
# (4 raw + 3 derived) also get the temporal-dynamics stats below.
DERIVED_CHANNELS = ["TMP", "PFD", "ARD"]
ALL_CHANNELS     = CHANNELS + DERIVED_CHANNELS
DYN_STATS        = ["slope", "trend_r2", "mad", "max_jump", "ac1", "exc_frac"]

# --- Denoising variants (downtime strategy x peak) feeding the benchmark ---
# Each maps to a denoised (peaks smoothed) and noised (peaks kept) pressure CSV.
DENOISE_VARIANTS = {
    "current": {
        "denoised": os.path.join(PRESSURE_DIR, "denoised", "CRRT_pressure_wo_peak.csv"),
        "noised":   os.path.join(PRESSURE_DIR, "noised",   "CRRT_pressure_with_peak.csv"),
    },
    "Th70": {
        "denoised": os.path.join(ABLATION_DOWNTIME_DIR, "Th70", "pressure", "denoised", "CRRT_pressure_wo_peak_Th70.csv"),
        "noised":   os.path.join(ABLATION_DOWNTIME_DIR, "Th70", "pressure", "noised",   "CRRT_pressure_with_peak_Th70.csv"),
    },
    "Th80": {
        "denoised": os.path.join(ABLATION_DOWNTIME_DIR, "Th80", "pressure", "denoised", "CRRT_pressure_wo_peak_Th80.csv"),
        "noised":   os.path.join(ABLATION_DOWNTIME_DIR, "Th80", "pressure", "noised",   "CRRT_pressure_with_peak_Th80.csv"),
    },
    "Th90": {
        "denoised": os.path.join(ABLATION_DOWNTIME_DIR, "Th90", "pressure", "denoised", "CRRT_pressure_wo_peak_Th90.csv"),
        "noised":   os.path.join(ABLATION_DOWNTIME_DIR, "Th90", "pressure", "noised",   "CRRT_pressure_with_peak_Th90.csv"),
    },
}

# --- Preprocessing leave-one-out (3 toggles: priming, downtime, peak) ---
# Baseline = all three removed; each LOO config turns exactly one off.
# Keys map to clean_streams() kwargs in preprocessing/build_streams.py.
LOO_CONFIGS = {
    "all3":        {"remove_priming": True,  "remove_downtime": True,  "remove_peak": True},   # == denoised
    "no_priming":  {"remove_priming": False, "remove_downtime": True,  "remove_peak": True},
    "no_downtime": {"remove_priming": True,  "remove_downtime": False, "remove_peak": True},
    "no_peak":     {"remove_priming": True,  "remove_downtime": True,  "remove_peak": False},  # == noised
}

# --- Best pipeline + model for the LOO ablation (set after the benchmark) ---
# Defaults are placeholders; update once the benchmark identifies the winner.
BEST_VARIANT     = "current"     # which downtime strategy
BEST_DOWNTIME    = "dose"        # 'flat' (current logic) or 'dose'
BEST_TH          = 0.8           # threshold for dose downtime
BEST_FEATURE_SET = "combined"    # clinical_only | machine_only | combined
BEST_CASE        = "1day"
BEST_MODEL       = "GradientBoosting"

# --- Cohort: restrict to the first N CRRT treatment days (Day <= MAX_TREATMENT_DAY) ---
# The study collected up to 14 treatment days; set to 7 to use only the first week.
# None (or >= 14) keeps all days. Applied at data-build time (machine + clinical), so
# combined / agg / sequences all inherit the restriction.
MAX_TREATMENT_DAY = 7

# --- Feature sets -> model-ready locations (per-day and aggregated) ---
FEATURE_SETS = ["clinical_only", "machine_only", "combined"]
CASES        = ["1day", "2day"]

# --- mRMR feature selection (fit on TRAIN, reused for val/test) ---
# v2 has a richer machine set (~133 per-day features), so machine budgets are raised
# 20 -> 30 to let mRMR sample the derived-physiology + dynamics features.
MRMR_ENABLED     = True
N_MRMR_MACHINE   = 30    # combined (legacy fixed grouping): top-N machine
N_MRMR_CLINICAL  = 40    # combined / clinical_only: top-N clinical (also DL clinical branch)
N_MRMR           = 30    # machine_only: top-N

# --- v2 combined: fixed clinical (mRMR top-N_clinical) + DYNAMIC machine count ----
# Instead of a fixed 30-machine grouping, keep ALL the best clinical features and search
# how many mRMR-ranked machine features to add — the count that maximizes VALIDATION
# AUROC (per model, per case) — so the combined model gets its best shot at beating the
# clinical-only model. k=0 (clinical-only) is in the grid as the floor.
COMBINED_DYNAMIC   = True
N_MRMR_MACHINE_MAX = 100                                   # cap the machine ranking length
# Forward-greedy machine selection (combined): keep the best clinical features fixed, then add
# machine features in mRMR-rank order, accepting each only if it raises VALIDATION AUROC by more
# than MACHINE_FWD_TOL. So combined-on-val >= clinical-on-val, pushing combined to beat clinical.
MACHINE_FWD_TOL      = 0.0                                 # strict val-AUROC improvement to keep a feature
MACHINE_FWD_PATIENCE = 20                                  # stop after this many consecutive non-improving features
MACHINE_K_GRID       = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100]   # (legacy; unused by greedy path)
MACHINE_K_MARGIN     = 0.01                                # (legacy)


def is_machine_col(col):
    """A feature column is 'machine' iff it starts with a raw OR derived pressure channel."""
    return col.startswith(tuple(CHANNELS + DERIVED_CHANNELS))


# --- DL signal reduction (downsample the 1440-step current-day stream) ---
# Per encoder: window-stat aggregation for conv/recurrent, patching for transformer.
#   window_agg: non-overlapping k-min windows -> [mean,min,max,std,slope]/channel (F = 4*5)
#   patch     : non-overlapping P-min patches, flattened, linear-embedded     (F = 4*P)
# Class-balanced minibatches: oversample the minority via WeightedRandomSampler
# (~50/50 per batch) and drop the loss pos_weight. False -> weighted loss + random batches.
DL_BALANCED_SAMPLER = True

DL_REDUCE_ENABLED = True
DL_REDUCE = {
    "cnn1d":       ("window_agg", {"k": 10}),
    "rnn":         ("window_agg", {"k": 10}),
    "transformer": ("patch",      {"P": 60}),
}

# --- Self-contained abalation data layout ---------------------------------
# refresh_data.py regenerates everything here from the denoising pressure CSVs,
# so the study neither depends on nor overwrites the baseline Model_Ready tables.
#   <ABL_DATA>/<variant>_<peak>/{machine_only,combined,machine_only_agg,
#                                combined_agg,clinical_only_agg}/{case}/{split}.csv
#   <ABL_DATA>/<variant>_<peak>/seq_cache/{case}_{split}.npz   (DL)
# clinical_only is RE-SPLIT onto the v2 70:10:20 split and stored INSIDE v2 (build_clinical.py).
# The external Model_Ready/clinical_only table is the read-only SOURCE — it is pre-split on the
# v1 baseline split, so it must NOT be read directly (that caused the combined-split bug).
ABL_DATA      = os.path.join(_HERE, "data", "abalation")          # v2: processed data INSIDE the folder
CLINICAL_SRC  = os.path.join(MODEL_READY_DIR, "clinical_only")     # external source (v1-split, read-only)
CLINICAL_ONLY = os.path.join(_HERE, "data", "clinical_only")       # v2-resplit per-day clinical (used everywhere)


def abl_root(variant="current", peak="denoised"):
    return os.path.join(ABL_DATA, f"{variant}_{peak}")


def feature_dir(variant, peak, feature_set, agg, case):
    """Resolve the {train,val,test}.csv directory for a (feature_set, scope, case)."""
    root = abl_root(variant, peak)
    if feature_set == "clinical_only" and not agg:
        return os.path.join(CLINICAL_ONLY, case)                   # v2-resplit per-day clinical
    sub = {
        ("machine_only", False): "machine_only", ("machine_only", True): "machine_only_agg",
        ("combined", False): "combined",         ("combined", True): "combined_agg",
        ("clinical_only", True): "clinical_only_agg",
    }[(feature_set, agg)]
    return os.path.join(root, sub, case)


def seq_cache_dir(variant="current", peak="denoised"):
    return os.path.join(abl_root(variant, peak), "seq_cache")
