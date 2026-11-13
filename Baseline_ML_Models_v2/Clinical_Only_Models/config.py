"""
config.py
=========
Central configuration for the Clinical-Only ML Pipeline.

Trains models using only clinical features (lab + PTI) — no machine
pressure data. Useful as a baseline to compare against machine-only
and machine+clinical models.
"""

import os
import numpy as np

# --- Directory Paths ---
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ML_DIR   = os.path.dirname(_HERE)                          # Baseline_ML_Models/
_PROJ_DIR = os.path.dirname(_ML_DIR)                        # CRRTnet_Data_Prep/

DATA_DIR    = os.path.join(_PROJ_DIR, "Data", "Processed", "Model_Ready", "clinical_only")
RESULTS_DIR = os.path.join(_HERE, "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "best_models")
SHAP_DIR    = os.path.join(RESULTS_DIR, "shap_plots")
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")
REPORT_DIR  = os.path.join(RESULTS_DIR, "reports")

# --- Experiment Dimensions ---
CASES            = ["1day", "2day"]
TRAINING_METHODS = ["full_train", "balanced_train"]
FEATURE_SETS     = ["mrmr"]   # mRMR — top 40 clinical features

# --- Column Configuration ---
META_COLS  = ["PID", "Visit_no", "Day", "Mortality", "days_until_death", "label"]
TARGET_COL = "label"

# --- Feature Selection ---
N_FEATURES_SELECT = 40   # top 40 clinical features (lab + PTI)

# --- Reproducibility ---
RANDOM_STATE = 42

# --- Hyperparameter Tuning ---
RANDOMIZED_SEARCH_ITER = 30
CV_FOLDS               = 5
SCORING                = "roc_auc"

# --- Model-Specific Hyperparameter Grids ---
PARAM_GRIDS = {

    "KNN": {
        "n_neighbors": [5, 11, 21],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },

    "DecisionTree": {
        "max_depth":        [None, 6, 10],
        "min_samples_split":[2, 10],
        "min_samples_leaf": [1, 5],
        "class_weight":     [None, "balanced"],
    },

    "RF": {
        "n_estimators":     [200, 500],
        "max_depth":        [None, 10, 20],
        "min_samples_leaf": [1, 4],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     [None, "balanced"],
    },

    "GradientBoosting": {
        "n_estimators":     [200, 300],
        "max_depth":        [3, 5],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8, 1.0],
    },

    "AdaBoost": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1, 0.5],
    },

    "LightGBM": {
        "n_estimators":      [200, 300],
        "max_depth":         [-1, 8],
        "learning_rate":     [0.05, 0.1],
        "num_leaves":        [31, 63],
        "min_child_samples": [20, 50],
        "class_weight":      [None, "balanced"],
    },

    "XGBoost": {
        "n_estimators":    [200, 300],
        "max_depth":       [5, 7],
        "learning_rate":   [0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0],
        "scale_pos_weight":[1, 10],
    },

    "LogisticRegression": {
        "C":            [0.01, 0.1, 1, 10],
        "penalty":      ["l1", "l2"],
        "solver":       ["saga"],
        "max_iter":     [5000],
        "class_weight": [None, "balanced"],
    },
}

# For balanced_train, remove class_weight from grids
PARAM_GRIDS_BALANCED = {
    name: {k: v for k, v in grid.items() if k != "class_weight"}
    for name, grid in PARAM_GRIDS.items()
}
PARAM_GRIDS_BALANCED["XGBoost"]["scale_pos_weight"] = [1]
