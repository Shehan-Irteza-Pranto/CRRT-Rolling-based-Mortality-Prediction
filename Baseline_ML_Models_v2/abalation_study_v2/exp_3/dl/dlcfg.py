"""
dlcfg.py — exp_3 Phase 2 (attention CNN1D, per-day, machine-only)
=================================================================
Plain per-day CNN1D (no augmentation — recombination was shown not to help) on the gap-aware
window-reduced day signal, with ATTENTION pooling (so the model can weight the end-of-day
deterioration windows that Phase-1 showed carry signal) and a 5-seed ensemble (mean test prob)
to kill single-seed luck. Reuses the parent CNN1DEncoder + window_aggregate. Self-contained:
all data/results inside exp_3/dl; parent read-only.
"""
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))                 # .../exp_3/dl
_EXP3   = os.path.dirname(_HERE)
_AS_DIR = os.path.dirname(_EXP3)                                     # abalation_study_v2
_FEAT   = os.path.join(_EXP3, "features")

DATA_DIR    = os.path.join(_HERE, "data")
RESULTS_DIR = os.path.join(_HERE, "results")

sys.path.insert(0, _AS_DIR)
from abl_config import DENOISE_VARIANTS as _DV, feature_dir as _feature_dir     # noqa: E402
RAW_PRESSURE = _DV["current"]["denoised"]
DL_DIR        = os.path.join(os.path.dirname(_AS_DIR), "Deep_Learning_Models")
DL_REDUCE_DIR = os.path.join(_AS_DIR, "models")

CASE = "1day"
DAY_LEN = 1440
N_CHANNELS = 6
GAP_MAX_MIN = 10

REDUCE = ("window_agg", {"k": 10})        # -> in_ch = 6*5 = 30
CNN_CHANNELS = 64
CNN_BLOCKS = 3
CNN_KERNEL = 5
CNN_DROPOUT = 0.3
ATTENTION = True                          # attention pooling vs masked-mean

LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH = 64
MAX_EPOCHS = 120
EARLY_STOP_PATIENCE = 15
N_SEEDS = 5
GPUS = [0, 1]

BASELINE_1DAY_CNN = 0.606                 # parent plain CNN1D, machine_only per_day, 1day
N_BOOTSTRAP = 1000


def machine_table(split, case=None):
    return os.path.join(_feature_dir("current", "denoised", "machine_only", False, case or CASE),
                        f"{split}.csv")


def reduced_in_ch():
    return N_CHANNELS * 5
