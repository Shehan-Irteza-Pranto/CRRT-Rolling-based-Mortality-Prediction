"""
eval/resampled_eval.py
======================
The Abalation_study evaluation protocol.

A model is trained once and scored once on the full test set (probabilities).
We then evaluate on resampled test subsets where:

  * ALL test positives are ALWAYS included (the positive set is fixed), and
  * negatives are randomly subsampled to a pos:neg ratio of 1:1 and 1:2,
  * repeated for N_ITER iterations, reporting mean +/- std of every metric.

Iteration i uses RandomState(seed0 + i). Because the seeds are fixed and the
positive set is constant, the negative draws are IDENTICAL across all models —
so model comparisons at each ratio are paired.

The classification threshold is the validation-derived F1-optimal threshold
(passed in); it is held fixed across all resamples.

This module reuses `compute_metrics` from the classical pipeline_utils.
"""

import os
import sys
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix, balanced_accuracy_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                        # Abalation_study/
from abl_config import RESAMPLE_RATIOS, N_ITER, SEED0             # noqa: E402


def compute_metrics(y_true, y_pred, y_prob):
    """Comprehensive classification metrics.

    Byte-identical to Machine_Clinical_Models/pipeline_utils.compute_metrics —
    vendored here so this central evaluation module is import-collision-free
    (every sibling pipeline ships its own top-level `config.py`).
    """
    has_both = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUROC":             roc_auc_score(y_true, y_prob) if has_both else np.nan,
        "AUPRC":             average_precision_score(y_true, y_prob) if has_both else np.nan,
        "F1":                f1_score(y_true, y_pred, zero_division=0),
        "F2":                fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "F3":                fbeta_score(y_true, y_pred, beta=3, zero_division=0),
        "F1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall":            recall_score(y_true, y_pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N_total":    int(len(y_true)),
        "N_positive": int(np.sum(y_true)),
        "N_negative": int(np.sum(1 - np.asarray(y_true))),
    }

# Metrics we summarize as mean/std across iterations (scalar, meaningful to average).
_SUMMARY_METRICS = ["AUROC", "AUPRC", "F1", "F2", "F3", "F1_macro", "F1_weighted",
                    "Precision", "Recall", "Balanced_Accuracy"]


def _draw_negatives(neg_idx, n_draw, rng):
    """Sample n_draw negatives without replacement (or all, if fewer exist)."""
    if n_draw >= len(neg_idx):
        return neg_idx
    return rng.choice(neg_idx, size=n_draw, replace=False)


def resample_ratio(y_true, y_prob, threshold, ratio,
                   n_iter=N_ITER, seed0=SEED0):
    """Resample one pos:neg ratio n_iter times; return (mean_dict, std_dict, raw_list).

    raw_list is the per-iteration metric dicts (for CIs / debugging).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.where(y_true == 1)[0]          # FIXED across iterations
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)

    raw = []
    for i in range(n_iter):
        rng = np.random.RandomState(seed0 + i)   # shared across models -> paired draws
        n_draw = min(ratio * n_pos, len(neg_idx))
        neg_s = _draw_negatives(neg_idx, n_draw, rng)
        idx = np.concatenate([pos_idx, neg_s])
        rng.shuffle(idx)
        yt, yp = y_true[idx], y_prob[idx]
        ypred = (yp >= threshold).astype(int)
        raw.append(compute_metrics(yt, ypred, yp))

    mean = {}
    std = {}
    for m in _SUMMARY_METRICS:
        vals = np.array([r[m] for r in raw], dtype=float)
        vals = vals[~np.isnan(vals)]
        mean[m] = float(vals.mean()) if len(vals) else float("nan")
        std[m] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    mean["n_pos"] = int(n_pos)
    mean["n_neg_per_iter"] = int(min(ratio * n_pos, len(neg_idx)))
    return mean, std, raw


def resampled_metrics(y_true, y_prob, threshold,
                      ratios=RESAMPLE_RATIOS, n_iter=N_ITER, seed0=SEED0):
    """Full protocol: per-ratio resampled mean/std + the full-test reference row.

    Returns a flat dict suitable for a results-table row, e.g.:
      full_test_AUROC, r1to1_AUROC_mean, r1to1_AUROC_std, r1to2_AUROC_mean, ...
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {}

    # Full (imbalanced) test reference, at the same fixed threshold.
    full = compute_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
    for m in _SUMMARY_METRICS:
        out[f"full_test_{m}"] = full[m]
    out["full_test_N_positive"] = full["N_positive"]
    out["full_test_N_negative"] = full["N_negative"]

    for ratio in ratios:
        mean, std, _ = resample_ratio(y_true, y_prob, threshold, ratio, n_iter, seed0)
        tag = f"r1to{ratio}"
        for m in _SUMMARY_METRICS:
            out[f"{tag}_{m}_mean"] = round(mean[m], 4)
            out[f"{tag}_{m}_std"] = round(std[m], 4)
        out[f"{tag}_n_pos"] = mean["n_pos"]
        out[f"{tag}_n_neg_per_iter"] = mean["n_neg_per_iter"]
    out["threshold"] = float(threshold)
    out["n_iter"] = int(n_iter)
    return out


# --- smoke test ---
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 2000
    y = (rng.rand(n) < 0.08).astype(int)              # ~8% positives, like the cohort
    # signal: positives get higher scores on average
    prob = np.clip(0.3 * y + 0.2 + 0.25 * rng.randn(n), 0, 1)
    thr = 0.5

    res = resampled_metrics(y, prob, thr)
    print("n_pos:", int(y.sum()), "n_neg:", int((1 - y).sum()))
    print("full_test_AUROC:", round(res["full_test_AUROC"], 4))
    for r in (1, 2):
        t = f"r1to{r}"
        print(f"{t}: AUROC {res[f'{t}_AUROC_mean']:.4f} +/- {res[f'{t}_AUROC_std']:.4f} "
              f"| F1 {res[f'{t}_F1_mean']:.4f} +/- {res[f'{t}_F1_std']:.4f} "
              f"| neg/iter={res[f'{t}_n_neg_per_iter']}")

    # invariants: positives fixed -> n_pos constant; std should be > 0 across iters.
    m1, s1, raw1 = resample_ratio(y, prob, thr, 1)
    assert all(r["N_positive"] == int(y.sum()) for r in raw1), "positives must be constant"
    assert s1["AUROC"] > 0, "AUROC std should be > 0 across resamples"
    # determinism: same seeds -> identical result
    m1b, _, _ = resample_ratio(y, prob, thr, 1)
    assert m1 == m1b, "protocol must be deterministic for fixed seeds"
    print("OK: positives fixed, std>0, deterministic.")
