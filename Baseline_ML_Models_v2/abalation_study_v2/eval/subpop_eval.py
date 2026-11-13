"""
eval/subpop_eval.py
===================
Subpopulation-based evaluation (alternative to negative resampling).

Point estimate  = AUROC on the FULL test set (all positives + all negatives, fixed
                  validation threshold) — the honest, reproducible number.
Uncertainty     = mean +/- std of AUROC ACROSS clinical subpopulations: the TEST
                  circuit-days are clustered into SUBPOP_N groups by Ward linkage on
                  the clinical features (model-independent, shared across all feature
                  sets), and AUROC is computed within each evaluable subgroup.

The subpopulation assignment is computed ONCE per (variant, peak, case) and cached,
so every model is scored on the identical subgroups.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, precision_score, recall_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import (feature_dir, RESULTS_DIR, KEY_COLS,            # noqa: E402
                        SUBPOP_N, SUBPOP_MIN_POS)
from resampled_eval import compute_metrics                            # noqa: E402  (full-test point metrics)

_META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
_ASSIGN_DIR = os.path.join(RESULTS_DIR, "subpop_assign")


def cluster_test_subpops(case, variant="current", peak="denoised", n_sub=SUBPOP_N):
    """Ward-cluster the TEST circuit-days on clinical features -> {(PID,Visit_no,Day): subpop}.

    Cached per (variant, peak, case) so all models share the identical subgroups.
    """
    os.makedirs(_ASSIGN_DIR, exist_ok=True)
    path = os.path.join(_ASSIGN_DIR, f"{variant}_{peak}_{case}.json")
    if os.path.exists(path):
        with open(path) as fh:
            recs = json.load(fh)["assign"]
        return {(r[0], r[1], r[2]): r[3] for r in recs}

    df = pd.read_csv(os.path.join(feature_dir(variant, peak, "clinical_only", False, case), "test.csv"))
    feats = [c for c in df.columns if c not in _META]
    X = df[feats].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    labels = fcluster(linkage(Xs, method="ward"), t=n_sub, criterion="maxclust") - 1   # 0..n_sub-1
    lut = {(int(r.PID), int(r.Visit_no), int(r.Day)): int(lab)
           for r, lab in zip(df[KEY_COLS].itertuples(index=False), labels)}
    with open(path, "w") as fh:
        json.dump({"variant": variant, "peak": peak, "case": case, "n_sub": int(n_sub),
                   "assign": [[k[0], k[1], k[2], v] for k, v in lut.items()]}, fh)
    return lut


_FULL = ["AUROC", "AUPRC", "F1", "F2", "F3", "F1_macro", "F1_weighted",
         "Precision", "Recall", "Balanced_Accuracy"]
_SUB  = ["AUROC", "F1", "F2", "F3", "Precision", "Recall"]


def _subgroup_metrics(ys, ps, pr):
    return {"AUROC": roc_auc_score(ys, ps),
            "F1": f1_score(ys, pr, zero_division=0),
            "F2": fbeta_score(ys, pr, beta=2, zero_division=0),
            "F3": fbeta_score(ys, pr, beta=3, zero_division=0),
            "Precision": precision_score(ys, pr, zero_division=0),
            "Recall": recall_score(ys, pr, zero_division=0)}


def subpop_metrics(keys_df, y_true, y_prob, threshold, lut,
                   n_sub=SUBPOP_N, min_pos=SUBPOP_MIN_POS, with_full=True):
    """Per-subpopulation mean/std (+ optional full-test point metrics). Flat results row.

    with_full=False omits full_test_* (used by the 'both' protocol, where resampled_metrics
    already supplies them).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}
    if with_full:
        full = compute_metrics(y_true, y_pred, y_prob)
        for m in _FULL:
            out[f"full_test_{m}"] = round(full[m], 4) if full[m] == full[m] else np.nan
        out["full_test_N_positive"] = full["N_positive"]
        out["full_test_N_negative"] = full["N_negative"]

    sub = np.array([lut.get((int(r[0]), int(r[1]), int(r[2])), -1)
                    for r in keys_df[KEY_COLS].to_numpy()])
    per = {m: [] for m in _SUB}
    n_eval = 0
    for s in range(n_sub):
        mask = sub == s
        npos = int(y_true[mask].sum())
        out[f"L2S{s}_n"] = int(mask.sum()); out[f"L2S{s}_npos"] = npos
        evaluable = npos >= min_pos and (y_true[mask] == 0).any()
        out[f"L2S{s}_status"] = "OK" if evaluable else "NE"
        if evaluable:
            mvals = _subgroup_metrics(y_true[mask], y_prob[mask], y_pred[mask])
            out[f"L2S{s}_AUROC"] = round(mvals["AUROC"], 4)
            for m in _SUB:
                per[m].append(mvals[m])
            n_eval += 1
        else:
            out[f"L2S{s}_AUROC"] = np.nan
    out["n_eval_subpops"] = n_eval
    for m in _SUB:
        vals = np.array(per[m], dtype=float)
        out[f"subpop_{m}_mean"] = round(float(vals.mean()), 4) if len(vals) else np.nan
        out[f"subpop_{m}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
    out["threshold"] = float(threshold)
    out["n_subpops"] = int(n_sub)
    return out


def eval_dispatch(keys_df, y_true, y_prob, threshold, case, variant, peak):
    """Protocol switch:
      "resampled" -> full-test + resampled 1:1/1:2 (AUROC/F1/F2/F3/Precision/Recall ... mean+/-std)
      "subpop"    -> full-test + per-subpopulation mean+/-std
      "both"      -> full-test + resampled 1:1/1:2 + per-subpopulation (default)
    """
    from abl_config import EVAL_PROTOCOL
    from resampled_eval import resampled_metrics
    if EVAL_PROTOCOL == "resampled":
        return resampled_metrics(y_true, y_prob, threshold)
    lut = cluster_test_subpops(case, variant, peak)
    if EVAL_PROTOCOL == "subpop":
        return subpop_metrics(keys_df, y_true, y_prob, threshold, lut)
    # "both": resampled supplies full_test_* + r1to1_*/r1to2_*; subpop adds subpop_* + L2S* only
    out = resampled_metrics(y_true, y_prob, threshold)
    out.update(subpop_metrics(keys_df, y_true, y_prob, threshold, lut, with_full=False))
    return out
