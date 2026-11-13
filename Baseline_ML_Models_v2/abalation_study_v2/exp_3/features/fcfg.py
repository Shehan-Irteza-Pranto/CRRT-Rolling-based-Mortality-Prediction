"""
config.py — exp_3 / Phase 1 (richer physiology features for the classical machine-only model)
=============================================================================================
Goal: break the 1-day machine-only AUROC ceiling (~0.594 classical) by ADDING the circuit-
instability signatures the parent feature set (~133 features) lacks: spectral band-power,
signal complexity (permutation entropy), within-day excursion structure, cross-channel coupling,
and end-of-day deterioration. New columns are concatenated to the existing 133 and the parent
mRMR + classical grid is reused. All data/results stay inside exp_3; parent stays read-only.
"""
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))                 # .../exp_3/features
_EXP3   = os.path.dirname(_HERE)                                     # .../exp_3
_AS_DIR = os.path.dirname(_EXP3)                                     # .../abalation_study_v2

DATA_DIR    = os.path.join(_HERE, "data")
RESULTS_DIR = os.path.join(_HERE, "results")

sys.path.insert(0, _AS_DIR)
from abl_config import (DENOISE_VARIANTS as _DV, feature_dir as _feature_dir,   # noqa: E402
                        KEY_COLS, TARGET_COL)

RAW_PRESSURE = _DV["current"]["denoised"]
DL_DIR = os.path.join(os.path.dirname(_AS_DIR), "Deep_Learning_Models")

CASE = "1day"
DAY_LEN = 1440
N_CHANNELS = 6
CHANNEL_NAMES = ["ACCESS", "FILTER", "EFFLUENT", "RETURN", "TMP", "PFD"]
GAP_MAX_MIN = 10

# baseline to beat (best 1-day machine-only classical anywhere in the project)
BASELINE_1DAY_CLASSICAL = 0.594

N_SEEDS = 5                 # multi-seed model fit -> mean +/- sd (kill single-seed luck)
N_MRMR  = 30               # machine_only mRMR budget (matches parent N_MRMR)
N_BOOTSTRAP = 1000         # patient-level bootstrap CI


def machine_table(split, case=None):
    """The parent machine_only per-day table (KEY_COLS + label + 133 features)."""
    return os.path.join(_feature_dir("current", "denoised", "machine_only", False, case or CASE),
                        f"{split}.csv")


def rich_table(split, case=None):
    """exp_3 enriched machine_only table (133 + new physiology columns)."""
    d = os.path.join(DATA_DIR, "rich", case or CASE)
    return os.path.join(d, f"{split}.csv")
