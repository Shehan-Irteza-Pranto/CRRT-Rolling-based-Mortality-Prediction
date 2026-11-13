"""
pipeline_utils.py
=================
Reusable utilities for the Machine + Clinical ML Pipeline.

Key difference from baseline: uses forward-fill + backward-fill for lab
features instead of median imputation, while still using median imputation
for machine features and PTI features where needed.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score,
)

from config import (META_COLS, TARGET_COL, RANDOM_STATE,
                    N_MACHINE_FEATURES, N_CLINICAL_FEATURES)

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# --- Feature helpers ---

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature column names (non-meta columns)."""
    return [c for c in df.columns if c not in META_COLS]


def load_feature_info(data_dir: str, case: str) -> dict:
    """Load the feature_info.json that describes feature groups."""
    path = os.path.join(data_dir, case, "feature_info.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# --- Data loading ---

def load_data(data_dir: str, case: str):
    """Load train/val/test CSVs for a given case. Returns (train, val, test)."""
    base = os.path.join(data_dir, case)
    train = pd.read_csv(os.path.join(base, "train.csv"))
    val   = pd.read_csv(os.path.join(base, "val.csv"))
    test  = pd.read_csv(os.path.join(base, "test.csv"))
    return train, val, test


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    """Split DataFrame into X (numpy) and y (numpy)."""
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return X, y


# --- Scaling ---
# Lab features were already forward/backward filled in the notebook.
# Any remaining NaNs (PIDs with all-missing for a lab) get median imputation here.
# Machine and PTI features also get median imputation for any NaNs.
#
# Binary PTI features are NOT scaled — only continuous features (machine + lab)
# are StandardScaled. Binary columns are identified as those with only 0/1 values.

def _identify_binary_cols(X: np.ndarray) -> np.ndarray:
    """Return boolean mask: True for columns that are binary (only 0 and 1)."""
    binary_mask = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            unique_vals = np.unique(valid)
            binary_mask[j] = set(unique_vals).issubset({0.0, 1.0})
    return binary_mask


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Median-impute remaining NaNs, then StandardScale continuous columns only.
    Binary (0/1) columns are imputed but NOT scaled.
    Fit on X_train, apply to both splits.
    Returns (X_train_processed, X_test_processed, (imputer, scaler, binary_mask)).
    """
    # Impute all columns
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    # Identify binary columns (before scaling)
    binary_mask = _identify_binary_cols(X_train_imp)
    cont_mask = ~binary_mask

    # Scale only continuous columns
    scaler = StandardScaler()
    if cont_mask.any():
        X_train_imp[:, cont_mask] = scaler.fit_transform(X_train_imp[:, cont_mask])
        X_test_imp[:, cont_mask]  = scaler.transform(X_test_imp[:, cont_mask])

    return X_train_imp, X_test_imp, (imputer, scaler, binary_mask)


def transform_with_preprocessor(prep_tuple, X: np.ndarray) -> np.ndarray:
    """Apply fitted (imputer, scaler, binary_mask) to a new array."""
    imputer, scaler, binary_mask = prep_tuple
    X_imp = imputer.transform(X)
    cont_mask = ~binary_mask
    if cont_mask.any():
        X_imp[:, cont_mask] = scaler.transform(X_imp[:, cont_mask])
    return X_imp


# --- Class balancing ---

def balance_train(X_train: np.ndarray, y_train: np.ndarray):
    """Undersample negatives to match positives (50/50)."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos   = len(pos_idx)

    if n_pos == 0 or n_pos >= len(neg_idx):
        return X_train, y_train

    rng         = np.random.RandomState(RANDOM_STATE)
    neg_sampled = rng.choice(neg_idx, size=n_pos, replace=False)
    idx         = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


def create_balanced_test(X_test: np.ndarray, y_test: np.ndarray):
    """Undersample majority class in test set to match minority."""
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    n_pos   = len(pos_idx)

    if n_pos == 0 or n_pos >= len(neg_idx):
        return X_test, y_test

    rng = np.random.RandomState(RANDOM_STATE)
    neg_sampled = rng.choice(neg_idx, size=n_pos, replace=False)
    idx = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(idx)
    return X_test[idx], y_test[idx]


# --- Feature selection ---

def _mrmr_select(X_tr, y_tr, feat_names_subset, k):
    """
    Run mRMR on a subset of features (given as numpy array + column names).
    Returns ordered list of selected feature names (length min(k, len(feat_names_subset))).
    """
    from mrmr import mrmr_classif
    k = min(k, X_tr.shape[1])
    if k == 0:
        return []
    X_df = pd.DataFrame(X_tr, columns=feat_names_subset)
    y_ser = pd.Series(y_tr)
    selected = mrmr_classif(X=X_df, y=y_ser, K=k, show_progress=False)
    return list(selected)


def _fs_mrmr_grouped(X_tr, y_tr, X_te, feat_names,
                     machine_feats, clinical_feats,
                     n_machine, n_clinical):
    """
    Run mRMR separately on machine and clinical feature groups:
    - top `n_machine` from machine features
    - top `n_clinical` from clinical features (lab + PTI)
    Selected features are the union of both groups.
    """
    name_to_idx = {f: i for i, f in enumerate(feat_names)}

    # Build machine subset
    machine_idx = [name_to_idx[f] for f in machine_feats if f in name_to_idx]
    machine_names = [feat_names[i] for i in machine_idx]
    X_tr_m = X_tr[:, machine_idx]
    chosen_m = _mrmr_select(X_tr_m, y_tr, machine_names, n_machine) if machine_idx else []

    # Build clinical subset
    clinical_idx = [name_to_idx[f] for f in clinical_feats if f in name_to_idx]
    clinical_names = [feat_names[i] for i in clinical_idx]
    X_tr_c = X_tr[:, clinical_idx]
    chosen_c = _mrmr_select(X_tr_c, y_tr, clinical_names, n_clinical) if clinical_idx else []

    # Combine selected indices (sorted to keep column order stable)
    chosen = chosen_m + chosen_c
    selected_idx = sorted(name_to_idx[f] for f in chosen)
    chosen_sorted = [feat_names[i] for i in selected_idx]

    Xtr = X_tr[:, selected_idx]
    Xte = X_te[:, selected_idx]
    return Xtr, Xte, chosen_sorted, None


def apply_feature_selection(method: str,
                            X_tr, y_tr, X_te,
                            feat_names: list,
                            machine_feats: list = None,
                            clinical_feats: list = None):
    """
    Dispatch to the correct feature-selection routine.

    For 'mrmr_grouped', requires machine_feats and clinical_feats.
    """
    if method == "mrmr_grouped":
        if machine_feats is None or clinical_feats is None:
            raise ValueError("mrmr_grouped requires machine_feats and clinical_feats")
        return _fs_mrmr_grouped(
            X_tr, y_tr, X_te, feat_names,
            machine_feats, clinical_feats,
            N_MACHINE_FEATURES, N_CLINICAL_FEATURES)
    else:
        raise ValueError(f"Unknown feature_set: '{method}'")


# --- Evaluation metrics ---

def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray) -> float:
    """Find probability threshold that maximises F1 on held-out set."""
    from sklearn.metrics import precision_recall_curve
    if y_true.sum() < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def compute_day_metrics(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        days: np.ndarray,
                        min_pos: int = 2) -> dict:
    """Compute AUROC for each unique day value."""
    results = {}
    for day in sorted(np.unique(days)):
        mask = (days == day)
        yt, yp = y_true[mask], y_prob[mask]
        if int(yt.sum()) < min_pos:
            results[int(day)] = np.nan
        else:
            results[int(day)] = float(roc_auc_score(yt, yp))
    return results


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """Compute comprehensive classification metrics."""
    has_both = len(np.unique(y_true)) > 1
    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_arr.ravel()

    return {
        "AUROC":             roc_auc_score(y_true, y_prob) if has_both else np.nan,
        "AUPRC":             average_precision_score(y_true, y_prob) if has_both else np.nan,
        "F1":                f1_score(y_true, y_pred, zero_division=0),
        "F1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall":            recall_score(y_true, y_pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N_total":    int(len(y_true)),
        "N_positive": int(y_true.sum()),
        "N_negative": int((1 - y_true).sum()),
    }


# --- Confusion-matrix plot ---

def save_confusion_matrix_plot(tn, fp, fn, tp, title: str, save_path: str):
    arr = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    if HAS_SNS:
        import seaborn as sns
        sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
    else:
        im = ax.imshow(arr, cmap="Blues")
        for (r, c), val in np.ndenumerate(arr):
            ax.text(c, r, str(val), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# --- SHAP analysis ---

_TREE_MODELS   = {"RF", "GradientBoosting",
                  "DecisionTree", "LightGBM", "XGBoost"}
_KERNEL_MODELS = {"KNN", "AdaBoost"}


def run_shap_analysis(model,
                      model_name: str,
                      X_train: np.ndarray,
                      X_test: np.ndarray,
                      feature_names: list,
                      title: str,
                      save_dir: str,
                      max_explain: int = 300) -> dict:
    """Compute SHAP values, save plots, return top-10 feature dict."""
    if not HAS_SHAP:
        logger.warning("shap not installed - skipping SHAP analysis.")
        return {}

    import shap as _shap
    os.makedirs(save_dir, exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "-").replace(":", "-")

    try:
        if model_name in _TREE_MODELS:
            explainer  = _shap.TreeExplainer(model)
            sv_raw     = explainer.shap_values(X_test[:max_explain])
        else:
            n_bg       = min(100, X_train.shape[0])
            background = _shap.kmeans(X_train, min(50, n_bg))
            def _pred(x):
                return model.predict_proba(x)[:, 1]
            explainer  = _shap.KernelExplainer(_pred, background)
            X_explain  = X_test[:min(100, max_explain)]
            sv_raw     = explainer.shap_values(X_explain)

        if isinstance(sv_raw, list):
            sv = sv_raw[1] if len(sv_raw) == 2 else sv_raw[0]
        elif sv_raw.ndim == 3:
            sv = sv_raw[:, :, 1]
        else:
            sv = sv_raw

        mean_abs = np.abs(sv).mean(axis=0)
        top15_idx  = np.argsort(mean_abs)[::-1][:15]
        top15_feat = [feature_names[i] for i in top15_idx]
        top15_vals = mean_abs[top15_idx]

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(15), top15_vals[::-1], color="steelblue")
        ax.set_yticks(range(15))
        ax.set_yticklabels([f[:45] for f in top15_feat[::-1]])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top-15 Features - {title}", fontsize=11)
        plt.tight_layout()
        bar_path = os.path.join(save_dir, f"{safe_title}_top15_bar.png")
        plt.savefig(bar_path, dpi=100, bbox_inches="tight")
        plt.close()

        # Summary plot
        try:
            X_show = X_test[:sv.shape[0], :][:, top15_idx]
            fig2   = plt.figure(figsize=(10, 8))
            _shap.summary_plot(sv[:, top15_idx], X_show,
                               feature_names=[f[:35] for f in top15_feat],
                               show=False, max_display=15)
            summary_path = os.path.join(save_dir, f"{safe_title}_shap_summary.png")
            plt.savefig(summary_path, dpi=100, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")

        logger.info(f"SHAP saved -> {bar_path}")
        return dict(zip(top15_feat, [float(v) for v in top15_vals]))

    except Exception as e:
        logger.warning(f"SHAP analysis failed for {model_name}: {e}")
        return {}
