"""
ensemble.py — exp_3 Phase 3 gate G3
===================================
Stack the Phase-1 rich-classical models + the Phase-2 attention-CNN1D into one machine-only score.
Base models give VAL + TEST probabilities (computed leak-safe in their own phases). A logistic
meta-learner is fit on the VAL probabilities (yva) and applied to TEST — no test info touches the
stack. Also reports the plain probability average and the best single base model, so the ensemble's
value is explicit. Day-level test AUROC + patient bootstrap 95% CI + full panel. Gate: >= 0.65?

Inputs (row-aligned, both derived from the machine_only test order):
  features/results/classical_probs_<case>.npz   (val_/test_ per classical model + yva,yte,pid_*)
  dl/results/dl_probs_<case>.npz                (val,test + yva,yte,pid_*)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP3 = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")
CLASS_NPZ = os.path.join(_EXP3, "features", "results", "classical_probs_{case}.npz")
DL_NPZ = os.path.join(_EXP3, "dl", "results", "dl_probs_{case}.npz")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score, precision_recall_curve)

# classical models to consider (each available in base_ and rich_ variants); the top-K by VAL
# AUROC are kept for stacking, plus the DL ensemble.
CLASSICAL_MODELS = ["GradientBoosting", "RF", "LightGBM", "XGBoost", "LogisticRegression"]
TOP_K_CLASSICAL = 3
N_BOOTSTRAP = 1000
GATE = 0.65


def _opt_threshold(y, s):
    if np.sum(y) < 2:
        return 0.5
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))])


def _panel(y, s, thr):
    pred = (s >= thr).astype(int)
    return {"AUROC": round(roc_auc_score(y, s), 4), "AUPRC": round(average_precision_score(y, s), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "F2": round(fbeta_score(y, pred, beta=2, zero_division=0), 4),
            "F3": round(fbeta_score(y, pred, beta=3, zero_division=0), 4),
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(recall_score(y, pred, zero_division=0), 4)}


def _boot_ci(y, s, pid, n=N_BOOTSTRAP, seed=0):
    pid = np.asarray(pid); pats = np.unique(pid); rng = np.random.default_rng(seed)
    by = {p: np.where(pid == p)[0] for p in pats}; aucs = []
    for _ in range(n):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return (round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)) if aucs else (np.nan, np.nan)


def run(case="1day"):
    cz = np.load(CLASS_NPZ.format(case=case))
    dz = np.load(DL_NPZ.format(case=case))
    yte = cz["yte"].astype(int); yva = cz["yva"].astype(int); pid_te = cz["pid_te"]
    # row-alignment guard
    assert np.array_equal(cz["yte"], dz["yte"]), "test label order mismatch classical vs dl"
    assert np.array_equal(cz["pid_te"], dz["pid_te"]), "test pid order mismatch"
    assert np.array_equal(cz["yva"], dz["yva"]), "val label order mismatch"

    # candidate base models: every classical model in base_ and rich_ form + the DL ensemble.
    cands = {}                                                   # name -> (val_prob, test_prob)
    for fs in ("base", "rich"):
        for m in CLASSICAL_MODELS:
            vk, tk = f"val_{fs}_{m}", f"test_{fs}_{m}"
            if vk in cz.files:
                cands[f"cls:{fs}:{m}"] = (cz[vk], cz[tk])
    cands["dl:attn_cnn1d"] = (dz["val"], dz["test"])

    # rank candidates by VAL AUROC (leak-safe selection)
    val_auc = {nm: roc_auc_score(yva, v) for nm, (v, _) in cands.items()}
    ranked = sorted(val_auc, key=val_auc.get, reverse=True)
    # keep the DL model + the top-K classical by val (dedup model family so the stack is diverse)
    keep, seen_fam = [], set()
    for nm in ranked:
        fam = nm.split(":")[-1]
        if nm.startswith("dl:") or (fam not in seen_fam and len([k for k in keep if k.startswith("cls:")]) < TOP_K_CLASSICAL):
            keep.append(nm); seen_fam.add(fam)
    if "dl:attn_cnn1d" not in keep:
        keep.append("dl:attn_cnn1d")
    names = keep
    Vv = np.column_stack([cands[n][0] for n in names])
    Vt = np.column_stack([cands[n][1] for n in names])

    rows = []
    for j, nm in enumerate(names):
        thr = _opt_threshold(yva, Vv[:, j]); lo, hi = _boot_ci(yte, Vt[:, j], pid_te)
        rows.append({"variant": nm, "val_AUROC": round(val_auc[nm], 4),
                     "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte, Vt[:, j], thr)})

    avg_v = Vv.mean(1); avg_t = Vt.mean(1)
    thr = _opt_threshold(yva, avg_v); lo, hi = _boot_ci(yte, avg_t, pid_te)
    rows.append({"variant": "ensemble:mean", "val_AUROC": round(roc_auc_score(yva, avg_v), 4),
                 "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte, avg_t, thr)})

    meta = LogisticRegression(max_iter=5000, C=1.0).fit(Vv, yva)
    stk_t = meta.predict_proba(Vt)[:, 1]; stk_v = meta.predict_proba(Vv)[:, 1]
    thr = _opt_threshold(yva, stk_v); lo, hi = _boot_ci(yte, stk_t, pid_te)
    rows.append({"variant": "ensemble:logit_stack", "val_AUROC": round(roc_auc_score(yva, stk_v), 4),
                 "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte, stk_t, thr)})

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"ensemble_{case}.csv"), index=False)

    # FINAL pick = the variant with the highest VAL AUROC (leak-safe), report its TEST.
    final = df.sort_values("val_AUROC", ascending=False).iloc[0]
    print(df.to_string(index=False))
    print(f"\n==================== Phase 3/4 gate ({case}) ====================")
    print(f"  stacked base models: {names}")
    print(f"  meta-learner weights {dict(zip(names, np.round(meta.coef_[0], 3)))}")
    print(f"  FINAL (val-selected) variant: {final['variant']}  "
          f"val {final['val_AUROC']} -> TEST {final['AUROC']} {final['AUROC_CI95']} "
          f"| F2 {final['F2']} F3 {final['F3']}")
    status = "REACHED" if final["AUROC"] >= GATE else "NOT reached"
    print(f"  target {GATE}: {status} (machine-only {case} = {final['AUROC']})")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day")
    a = ap.parse_args()
    run(a.case)
