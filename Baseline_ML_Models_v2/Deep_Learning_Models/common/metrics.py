"""
common/metrics.py
=================
THE classical metric protocol, reused verbatim so the DL comparison is apples-to-apples.

These functions are copied from ../Machine_Only_Models/pipeline_utils.py intentionally
(identical AUROC/AUPRC/F1/precision/recall, identical validation-derived F1 threshold,
identical balanced-test construction with the same RANDOM_STATE). Do NOT "improve" them
here — any divergence from the classical pipeline invalidates the head-to-head.

Adds: bootstrap confidence intervals (paired with the >=5-seed mean +/- std reporting).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score,
    precision_recall_curve,
)

RANDOM_STATE = 42   # mirrors the classical pipeline


# --- Validation-derived F1 threshold (identical to classical) ---

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Probability threshold maximising F1 on a held-out (validation) set."""
    if y_true.sum() < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    return float(thresholds[int(np.argmax(f1))])


# --- Full metric dict (identical to classical compute_metrics) ---

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    has_both = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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


# --- Balanced 50/50 test set (identical RNG to classical create_balanced_test) ---

def create_balanced_test(idx_pos_neg_arrays, y_test: np.ndarray):
    """Return indices of an undersampled 50/50 test subset (majority down to minority).

    Returns an index array into the original test order so callers can subset both
    probabilities and any per-sample artefacts consistently.
    """
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    n_pos = len(pos_idx)
    if n_pos == 0 or n_pos >= len(neg_idx):
        return np.arange(len(y_test))
    rng = np.random.RandomState(RANDOM_STATE)
    neg_sampled = rng.choice(neg_idx, size=n_pos, replace=False)
    idx = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(idx)
    return idx


# --- Bootstrap CIs (the DL-specific addition, paired with multi-seed mean +/- std) ---

def bootstrap_metric_ci(y_true, y_prob, metric="auroc",
                        n_boot=1000, ci=0.95, seed=RANDOM_STATE):
    """Percentile bootstrap CI for a single test set's metric.

    metric: 'auroc' or 'auprc'. Returns (point, lo, hi).
    """
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    point = float(fn(y_true, y_prob))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        b = rng.randint(0, n, n)
        if len(np.unique(y_true[b])) < 2:
            continue
        stats.append(fn(y_true[b], y_prob[b]))
    if not stats:
        return point, np.nan, np.nan
    lo = float(np.percentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(stats, 100 * (1 + ci) / 2))
    return point, lo, hi


def aggregate_seeds(per_seed_metrics: list, keys=("AUROC", "AUPRC", "F1", "Precision", "Recall")):
    """Reduce a list of per-seed metric dicts to mean +/- std per key (the headline reporting)."""
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics if k in m and m[k] == m[k]], dtype=float)
        if len(vals):
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            out[f"{k}_n"]    = int(len(vals))
    return out
