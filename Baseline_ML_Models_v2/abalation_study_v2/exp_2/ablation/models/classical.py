"""
models/classical.py
====================
Classical 8-model benchmark for the Abalation_study, evaluated with the new
resampled protocol (eval/resampled_eval.py).

Reuses the existing hyperparameter grids + preprocessing from
Machine_Clinical_Models (config.PARAM_GRIDS, pipeline_utils.scale_features,
find_optimal_threshold). Trains each model ONCE, scores the full test set once,
then hands (y_test, y_prob, val-threshold) to resampled_metrics so every model
is judged on identical fixed-positive / resampled-negative test draws.

Feature sets x temporal scope map to existing Model_Ready tables:
  clinical_only  per-day -> Model_Ready/clinical_only/{case}
  clinical_only  agg     -> Model_Ready/clinical_agg_day/{case}
  machine_only   per-day -> Model_Ready/{case}/denoised
  machine_only   agg     -> Model_Ready/machine_agg_day/{case}
  combined       per-day -> Model_Ready/machine_clinical/{case}
  combined       agg     -> Model_Ready/machine_clinical_agg_day/{case}
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
sys.path.insert(0, _HERE)
from abl_config import (RANDOM_STATE, RESULTS_DIR, feature_dir, MRMR_ENABLED,  # noqa: E402
                        COMBINED_DYNAMIC, MACHINE_FWD_TOL, MACHINE_FWD_PATIENCE,
                        N_MRMR_MACHINE_MAX, KEY_COLS, is_machine_col, MACHINE_CLINICAL_DIR)
from feature_select import (get_or_compute_selection,                         # noqa: E402
                            get_or_compute_combined_ranking)
sys.path.insert(0, os.path.join(_ABL, "eval"))
from subpop_eval import eval_dispatch                                        # noqa: E402

# Reuse classical HP grids + preprocessing (Machine_Clinical_Models ships them).
# exp_2: use the config's absolute path (this file sits one level deeper than the parent's).
_MC = MACHINE_CLINICAL_DIR
sys.path.insert(0, _MC)
from config import (PARAM_GRIDS, PARAM_GRIDS_BALANCED,                       # noqa: E402
                    RANDOMIZED_SEARCH_ITER, CV_FOLDS, SCORING)
from pipeline_utils import (scale_features, transform_with_preprocessor,    # noqa: E402
                            find_optimal_threshold)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Union of meta columns across the various Model_Ready tables.
META_COLS = {"PID", "Visit_no", "Day", "Date", "Death_Date",
             "Mortality", "days_until_death", "label"}
TARGET = "label"

# Parallelism cap for the HP search. Estimators run SINGLE-THREADED (n_jobs=1) and
# the search parallelizes across this many workers — avoids the nested-parallelism
# explosion (cores x cores threads) that thrashes a shared node. Override with
# env ABL_NJOBS.
N_JOBS = int(os.environ.get("ABL_NJOBS", min(8, (os.cpu_count() or 4))))


def build_model_registry():
    """The 8 classical estimators (same set/order as the baseline pipelines).

    Tree models are SINGLE-THREADED (n_jobs=1); parallelism comes only from the
    RandomizedSearchCV pool (N_JOBS) so there is no nested oversubscription.
    """
    reg = {}
    if HAS_XGB:
        reg["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                           eval_metric="logloss", verbosity=0)
    if HAS_LGB:
        reg["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                             verbosity=-1, force_col_wise=True)
    reg["LogisticRegression"] = LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)
    reg["RF"] = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    reg["GradientBoosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)
    reg["KNN"] = KNeighborsClassifier()
    reg["DecisionTree"] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    reg["AdaBoost"] = AdaBoostClassifier(random_state=RANDOM_STATE)
    return reg


def _load(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    va = pd.read_csv(os.path.join(data_dir, "val.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    feats = [c for c in tr.columns if c not in META_COLS]
    return tr, va, te, feats


def _xy(df, feats):
    return df[feats].values.astype(float), df[TARGET].values.astype(int)


def _tune(model_name, registry, X, y, balanced):
    grid = (PARAM_GRIDS_BALANCED if balanced else PARAM_GRIDS)[model_name]
    n_iter = min(RANDOMIZED_SEARCH_ITER, int(np.prod([len(v) for v in grid.values()])))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base = registry[model_name]
    search = RandomizedSearchCV(base.__class__(**base.get_params()), grid,
                                n_iter=n_iter, scoring=SCORING, cv=cv, n_jobs=N_JOBS,
                                random_state=RANDOM_STATE, refit=True, error_score=np.nan)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _dynamic_combined_select(feats, X_tr, y_tr, X_va, y_va, eval_estimator, ranking_key):
    """Combined selection = fixed best clinical features + FORWARD-GREEDY machine features.

    Keep the clinical_only top-N fixed (the clinical floor), then walk the mRMR machine
    ranking and ADD each machine feature only if it raises VALIDATION AUROC by > tol — so
    combined-on-val >= clinical-on-val, pushing combined to outperform clinical-only.
    Returns (sel_idx, sel_names, machine_k, clinical_val_auroc, combined_val_auroc).
    """
    clin_top, mach_rank = get_or_compute_combined_ranking(ranking_key, feats, X_tr, y_tr)
    name_to_idx = {f: i for i, f in enumerate(feats)}

    def _val_auroc(cols):
        idx = [name_to_idx[c] for c in cols]
        est = eval_estimator.__class__(**eval_estimator.get_params())
        try:
            est.fit(X_tr[:, idx], y_tr)
            p = est.predict_proba(X_va[:, idx])[:, 1]
            return roc_auc_score(y_va, p) if len(np.unique(y_va)) > 1 else 0.0
        except Exception:
            return -1.0

    base = _val_auroc(clin_top) if clin_top else 0.0     # clinical-only floor (machine_k = 0)
    current = list(clin_top); best = base; k = 0; stall = 0
    for f in mach_rank[:N_MRMR_MACHINE_MAX]:             # mRMR-rank order
        a = _val_auroc(current + [f])
        if a > best + MACHINE_FWD_TOL:                   # keep only val-improving machine features
            current.append(f); best = a; k += 1; stall = 0
        else:
            stall += 1
            if stall >= MACHINE_FWD_PATIENCE:            # machine stopped helping -> stop
                break
    sel_idx = sorted(name_to_idx[c] for c in current)
    return sel_idx, [feats[i] for i in sel_idx], k, round(float(base), 4), round(float(best), 4)


SAVED_MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models", "classical")


def run_one(feature_set, case, agg, model_name, registry=None,
            variant="current", peak="denoised", data_dir=None, tag="", save_model=True):
    """Train one model, evaluate with the resampled protocol. Returns a results row.

    Resolves the train/val/test dir from the abalation layout via feature_dir(),
    unless an explicit `data_dir` is given (used by the LOO driver).
    Saves the fitted model (+ preprocessor + threshold) to results/saved_models/classical/.
    """
    registry = registry or build_model_registry()
    t0 = time.time()
    ddir = data_dir or feature_dir(variant, peak, feature_set, agg, case)
    tr, va, te, feats = _load(ddir)
    X_tr, y_tr = _xy(tr, feats); X_va, y_va = _xy(va, feats); X_te, y_te = _xy(te, feats)

    X_tr_sc, X_te_sc, prep = scale_features(X_tr, X_te)
    X_va_sc = transform_with_preprocessor(prep, X_va)

    # mRMR feature selection — computed ONCE per cell, cached + shared across all models
    scope = "agg" if agg else "per_day"
    machine_k = None; clin_val = comb_val = None
    if MRMR_ENABLED and feature_set == "combined" and COMBINED_DYNAMIC:
        # fixed best clinical features + forward-greedy machine features (per-model val-AUROC)
        rkey = f"{variant}_{peak}_combined_{scope}_{case}"
        sel_idx, sel_names, machine_k, clin_val, comb_val = _dynamic_combined_select(
            feats, X_tr_sc, y_tr, X_va_sc, y_va, registry[model_name], rkey)
        X_tr_sc = X_tr_sc[:, sel_idx]; X_va_sc = X_va_sc[:, sel_idx]; X_te_sc = X_te_sc[:, sel_idx]
    elif MRMR_ENABLED:
        sel_key = f"{variant}_{peak}_{feature_set}_{scope}_{case}"
        sel_idx, sel_names = get_or_compute_selection(sel_key, feats, X_tr_sc, y_tr, feature_set)
        X_tr_sc = X_tr_sc[:, sel_idx]; X_va_sc = X_va_sc[:, sel_idx]; X_te_sc = X_te_sc[:, sel_idx]
    else:
        sel_idx, sel_names = list(range(len(feats))), feats

    best, best_params, cv_auroc = _tune(model_name, registry, X_tr_sc, y_tr, balanced=False)
    best.fit(X_tr_sc, y_tr)

    thr = find_optimal_threshold(y_va, best.predict_proba(X_va_sc)[:, 1])
    y_prob_te = best.predict_proba(X_te_sc)[:, 1]

    scope = "agg" if agg else "per_day"
    n_machine_sel = sum(1 for f in sel_names if is_machine_col(f))
    row = {"family": "classical", "feature_set": feature_set, "case": case,
           "scope": scope, "model": model_name,
           "variant": tag or f"{variant}_{peak}", "n_features": len(feats),
           "n_selected": len(sel_names), "n_machine_selected": n_machine_sel,
           "machine_k": machine_k if machine_k is not None else "",
           "clinical_val_auroc": clin_val if clin_val is not None else "",
           "combined_val_auroc": comb_val if comb_val is not None else "",
           "cv_auroc": round(cv_auroc, 4)}
    row.update(eval_dispatch(te[KEY_COLS], y_te, y_prob_te, thr, case, variant, peak))

    if save_model:
        import joblib
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        mid = f"{tag or variant + '_' + peak}_{feature_set}_{scope}_{case}_{model_name}"
        mpath = os.path.join(SAVED_MODELS_DIR, mid + ".joblib")
        joblib.dump({"model": best, "preprocessor": prep, "threshold": thr,
                     "features": feats, "selected_idx": sel_idx, "selected_features": sel_names,
                     "best_params": best_params,
                     "feature_set": feature_set, "scope": scope, "case": case,
                     "variant": variant, "peak": peak}, mpath)
        row["model_path"] = mpath

    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def _done_keys(out):
    """Configs already present in the output CSV -> {(variant, feature_set, scope, case, model)}.

    The variant (e.g. 'current_denoised' vs 'current_noised') is part of the key so a
    different peak/variant is NEVER treated as already-done when sharing an output file.
    """
    if not out or not os.path.exists(out) or os.path.getsize(out) == 0:
        return set(), []
    prev = pd.read_csv(out)
    keys = {(getattr(r, "variant", ""), r.feature_set, r.scope, r.case, r.model)
            for r in prev.itertuples()}
    return keys, prev.to_dict("records")


def run_matrix(feature_sets, cases, scopes, models, variant="current", peak="denoised",
               tag="", out=None):
    """Run the config matrix with RESUME: skip configs already in `out`, and write
    `out` incrementally after every config so an interrupted run loses nothing."""
    registry = build_model_registry()
    vkey = tag or f"{variant}_{peak}"                # variant label stored on each row
    done, acc = _done_keys(out)
    if done:
        print(f"[resume] {len(done)} config(s) already done in {out} — skipping them.")
    for fs in feature_sets:
        for case in cases:
            for agg in scopes:                       # scopes: list of bools
                scope = "agg" if agg else "per_day"
                for m in models:
                    if m not in registry:
                        continue
                    if (vkey, fs, scope, case, m) in done:
                        print(f"[skip-done] {vkey}/{fs}/{scope}/{case}/{m}")
                        continue
                    try:
                        row = run_one(fs, case, agg, m, registry,
                                      variant=variant, peak=peak, tag=tag)
                        acc.append(row)
                        done.add((vkey, fs, scope, case, m))
                        if out:                       # checkpoint after every config
                            os.makedirs(os.path.dirname(out), exist_ok=True)
                            pd.DataFrame(acc).to_csv(out, index=False)
                        mk = f" k={row['machine_k']}" if row.get('machine_k') != "" else ""
                        print(f"[ok] {fs}/{scope}/{case}/{m}: "
                              f"full AUROC {row['full_test_AUROC']:.3f} "
                              f"| subpop {row['subpop_AUROC_mean']}±{row['subpop_AUROC_std']} "
                              f"(n={row['n_eval_subpops']}){mk}")
                    except Exception as e:
                        print(f"[err] {fs}/{scope}/{case}/{m}: {e}")
    return pd.DataFrame(acc)


# --- smoke test: one model, machine_only per-day, current denoised data ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="RF")
    ap.add_argument("--feature-set", default="machine_only")
    ap.add_argument("--case", default="1day")
    ap.add_argument("--agg", action="store_true")
    a = ap.parse_args()

    row = run_one(a.feature_set, a.case, a.agg, a.model)
    print(f"\n{a.feature_set}/{'agg' if a.agg else 'per_day'}/{a.case}/{a.model}")
    print(f"  full-test AUROC : {row['full_test_AUROC']:.4f}  (pos={row['full_test_N_positive']}, neg={row['full_test_N_negative']})")
    print(f"  subpop AUROC    : {row['subpop_AUROC_mean']} ± {row['subpop_AUROC_std']} "
          f"(evaluable subpops={row['n_eval_subpops']}/{row['n_subpops']})")
    if row.get("machine_k") != "":
        print(f"  machine_k={row['machine_k']}  clinical_val={row['clinical_val_auroc']}  combined_val={row['combined_val_auroc']}")
    print(f"  threshold={row['threshold']:.3f}  cv_auroc={row['cv_auroc']}  elapsed={row['elapsed_s']}s")
