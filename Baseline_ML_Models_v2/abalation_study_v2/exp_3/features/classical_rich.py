"""
classical_rich.py — exp_3 Phase 1 gate G1
=========================================
Run the parent classical grid (mRMR top-30 + RandomizedSearchCV tuning) on the machine_only
table, for TWO feature sets through an identical pipeline:
  base : parent 133 features only
  rich : 133 + the new circuit-instability features (build_rich_features.py)
so the lift is attributable to the new features, not the pipeline. Multi-seed refit (mean +/- sd)
kills single-seed luck. Reports per-day full-test AUROC + patient bootstrap 95% CI, and for the
best rich model lists which NEW features mRMR selected.

Reuses: feature_select.mrmr_select, pipeline_utils.scale_features/find_optimal_threshold,
classical.build_model_registry + PARAM_GRIDS tuning. Leak-safe: impute/scale/mRMR/tune fit on
TRAIN only, threshold on VAL, AUROC on TEST. All outputs in exp_3/features/results.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_AS_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)                          # my fcfg first
from fcfg import (CASE, KEY_COLS, machine_table, rich_table, RESULTS_DIR,      # noqa: E402
                  BASELINE_1DAY_CLASSICAL, N_SEEDS, N_MRMR, N_BOOTSTRAP)

sys.path.insert(0, os.path.join(_AS_DIR, "models"))
from feature_select import mrmr_select                                         # noqa: E402
import classical as _cl                                                        # noqa: E402
build_model_registry = _cl.build_model_registry
PARAM_GRIDS, RANDOMIZED_SEARCH_ITER = _cl.PARAM_GRIDS, _cl.RANDOMIZED_SEARCH_ITER
CV_FOLDS, SCORING = _cl.CV_FOLDS, _cl.SCORING
from pipeline_utils import (scale_features, transform_with_preprocessor,       # noqa: E402
                            find_optimal_threshold)

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score)
from sklearn.base import clone

META_COLS = {"PID", "Visit_no", "Day", "Date", "Death_Date",
             "Mortality", "days_until_death", "label"}
MODELS = ["LightGBM", "RF", "XGBoost", "GradientBoosting", "LogisticRegression"]
N_JOBS = int(os.environ.get("ABL_NJOBS", min(8, (os.cpu_count() or 4))))


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


def _load(featset, case):
    tab = rich_table if featset == "rich" else machine_table
    tr = pd.read_csv(tab("train", case)); va = pd.read_csv(tab("val", case)); te = pd.read_csv(tab("test", case))
    feats = [c for c in tr.columns if c not in META_COLS]
    return tr, va, te, feats


def _xy(df, feats):
    return df[feats].to_numpy(float), df["label"].to_numpy(int)


def _tune(name, X, y):
    reg = build_model_registry()
    if name not in reg:
        return None
    grid = PARAM_GRIDS[name]
    n_iter = min(RANDOMIZED_SEARCH_ITER, int(np.prod([len(v) for v in grid.values()])))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    base = reg[name]
    s = RandomizedSearchCV(base.__class__(**base.get_params()), grid, n_iter=n_iter,
                           scoring=SCORING, cv=cv, n_jobs=N_JOBS, random_state=42,
                           refit=True, error_score=np.nan)
    s.fit(X, y)
    return s.best_estimator_


def _multiseed_probs(best, Xtr, ytr, Xva, Xte):
    """Refit best estimator with N_SEEDS random_states; return mean val & test probs across seeds."""
    vp, tp = [], []
    for sd in range(N_SEEDS):
        est = clone(best)
        try:
            est.set_params(random_state=sd)
        except ValueError:
            pass
        est.fit(Xtr, ytr)
        vp.append(est.predict_proba(Xva)[:, 1]); tp.append(est.predict_proba(Xte)[:, 1])
    return np.array(vp), np.array(tp)


def run_featset(featset, case=CASE):
    tr, va, te, feats = _load(featset, case)
    Xtr, ytr = _xy(tr, feats); Xva, yva = _xy(va, feats); Xte, yte = _xy(te, feats)
    pid_te = te["PID"].to_numpy()

    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
    Xtr_sc, Xte_sc, prep = scale_features(Xtr, Xte)
    Xva_sc = transform_with_preprocessor(prep, Xva)

    chosen = mrmr_select(Xtr_sc, ytr, list(feats), min(N_MRMR, Xtr_sc.shape[1]))
    cidx = [feats.index(c) for c in chosen]
    Xtr_s, Xva_s, Xte_s = Xtr_sc[:, cidx], Xva_sc[:, cidx], Xte_sc[:, cidx]

    rows = []; probs = {}
    for name in MODELS:
        t0 = time.time()
        best = _tune(name, Xtr_s, ytr)
        if best is None:
            continue
        vp, tp = _multiseed_probs(best, Xtr_s, ytr, Xva_s, Xte_s)
        seed_aucs = [roc_auc_score(yte, tp[i]) for i in range(len(tp))]
        mean_prob = tp.mean(0); mean_vprob = vp.mean(0)
        probs[name] = (mean_vprob, mean_prob)               # for the ensemble (val, test)
        thr = find_optimal_threshold(yva, mean_vprob)
        lo, hi = _boot_ci(yte, mean_prob, pid_te)
        row = {"featset": featset, "model": name, "n_features": len(feats), "n_selected": len(chosen),
               "test_AUROC_mean": round(float(np.mean(seed_aucs)), 4),
               "test_AUROC_sd": round(float(np.std(seed_aucs)), 4),
               "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte, mean_prob, thr),
               "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"[{featset}/{name}] test AUROC {row['test_AUROC_mean']}±{row['test_AUROC_sd']} "
              f"{row['AUROC_CI95']} | AUPRC {row['AUPRC']} F2 {row['F2']} ({row['elapsed_s']}s)")
    return rows, chosen, feats, probs, (yva, yte, te["PID"].to_numpy(), va["PID"].to_numpy())


def evaluate(case=CASE):
    all_rows = []
    rich_chosen = base_feats = None
    saved = {}
    for fs in ("base", "rich"):
        rows, chosen, feats, probs, meta = run_featset(fs, case)
        all_rows += rows
        if fs == "base":
            base_feats = set(feats)
        if fs == "rich":
            rich_chosen = chosen
        yva, yte, pid_te, pid_va = meta
        # save BOTH feature sets' probs (prefix base_/rich_) so the ensemble can pick the best
        saved.update({f"val_{fs}_{k}": v[0] for k, v in probs.items()})
        saved.update({f"test_{fs}_{k}": v[1] for k, v in probs.items()})
    np.savez(os.path.join(RESULTS_DIR, f"classical_probs_{case}.npz"),
             yva=yva, yte=yte, pid_te=pid_te, pid_va=pid_va, **saved)
    df = pd.DataFrame(all_rows).sort_values("test_AUROC_mean", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"classical_rich_{case}.csv")
    df.to_csv(out, index=False)

    new_selected = [c for c in (rich_chosen or []) if c not in base_feats]
    best_rich = df[df.featset == "rich"].iloc[0]
    best_base = df[df.featset == "base"].iloc[0]
    print("\n==================== Phase 1 gate G1 ====================")
    print(f"  baseline (project best 1day machine classical): {BASELINE_1DAY_CLASSICAL}")
    print(f"  base-133 best : {best_base['model']} {best_base['test_AUROC_mean']}±{best_base['test_AUROC_sd']} {best_base['AUROC_CI95']}")
    print(f"  rich-206 best : {best_rich['model']} {best_rich['test_AUROC_mean']}±{best_rich['test_AUROC_sd']} {best_rich['AUROC_CI95']}")
    print(f"  lift (rich-base): {round(best_rich['test_AUROC_mean'] - best_base['test_AUROC_mean'], 4)}")
    print(f"  NEW features mRMR selected ({len(new_selected)}/{len(rich_chosen or [])}): {new_selected}")
    print(f"  -> {out}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default=CASE)
    a = ap.parse_args()
    evaluate(a.case)
