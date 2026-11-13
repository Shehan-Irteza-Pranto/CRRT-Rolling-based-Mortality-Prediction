"""
bootstrap_table4.py — patient-clustered bootstrap 95% CIs for every Table IV cell (8 classical x 3 pipelines,
1-day, per-day, features 60 clinical / 40 machine / 40+60 combined).
Re-scores TEST from the fitted models saved by run_table7_updated.py (results/saved_models/classical/
current_denoised_<pipeline>_per_day_1day_<model>.joblib: model + train-fit preprocessor + selected_idx), checks
each point AUROC/AUPRC against results/table7_updated_1day.csv, then resamples the 196 test patients (all
their days) B=2000 times with seed 0 — the same protocol as exp_4/bootstrap_ci.py.
Writes results/bootstrap_table4_1day.csv, results/table4_test_probs_1day.csv (all 24 prob columns), BOOTSTRAP_TABLE4.md.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "models"))
import classical as CL                                                    # noqa: E402 (feature_dir, _load, prep utils)

RES = os.path.join(_HERE, "results"); SAVED = os.path.join(RES, "saved_models", "classical")
CASE, B, SEED = "1day", 2000, 0
KEY = CL.KEY_COLS
PIPES = [("clinical_only", "Clinical-only"), ("machine_only", "Machine-only"), ("combined", "Combined")]
MODELS = [("LogisticRegression", "LogReg"), ("DecisionTree", "DecisionTree"), ("RF", "RandomForest"),
          ("GradientBoosting", "GradBoost"), ("AdaBoost", "AdaBoost"), ("KNN", "KNN"),
          ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost")]


def test_probs():
    base, cols = None, []
    for fs, plabel in PIPES:
        tr, va, te, feats = CL._load(CL.feature_dir("current", "denoised", fs, False, CASE))
        X_te = te[feats].values.astype(float)
        d = te[KEY + [CL.TARGET]].rename(columns={CL.TARGET: "label"}).copy()
        for m, mlabel in MODELS:
            ck = joblib.load(os.path.join(SAVED, f"current_denoised_{fs}_per_day_{CASE}_{m}.joblib"))
            assert ck["features"] == feats, f"{fs}/{m}: feature list changed since the model was saved"
            Xs = CL.transform_with_preprocessor(ck["preprocessor"], X_te)[:, ck["selected_idx"]]
            d[f"{plabel}|{mlabel}"] = ck["model"].predict_proba(Xs)[:, 1]; cols.append(f"{plabel}|{mlabel}")
        if base is None:
            base = d
        else:
            d = d.rename(columns={"label": "label2"}); base = base.merge(d, on=KEY, how="inner")
            assert (base["label"] == base["label2"]).all(); base = base.drop(columns="label2")
    assert len(base) == 887 and base["label"].sum() == 62 and base["PID"].nunique() == 196, len(base)
    return base, cols


def main():
    df, cols = test_probs(); y = df["label"].to_numpy(int); pid = df["PID"].to_numpy()
    ref = pd.read_csv(os.path.join(RES, f"table7_updated_{CASE}.csv"))
    ref = {f"{r.pipeline}|{r.model}": (r.AUROC, r.AUPRC) for r in ref.itertuples()}
    print("point-estimate check vs table7_updated (AUROC/AUPRC):")
    bad = 0
    for c in cols:
        a, p = roc_auc_score(y, df[c]), average_precision_score(y, df[c]); ra, rp = ref[c]
        ok = abs(round(a, 3) - ra) < 1e-9 and abs(round(p, 3) - rp) < 1e-9
        bad += (not ok); print(f"  {c:28s} {a:.3f}/{p:.3f}  table {ra:.3f}/{rp:.3f}  {'ok' if ok else 'MISMATCH'}")
    print(f"  mismatches: {bad}")
    df.to_csv(os.path.join(RES, f"table4_test_probs_{CASE}.csv"), index=False)

    pats = np.unique(pid); by = {p: np.where(pid == p)[0] for p in pats}; rng = np.random.default_rng(SEED)
    A = {c: [] for c in cols}; P = {c: [] for c in cols}; used = 0
    for _ in range(B):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        used += 1
        for c in cols:
            A[c].append(roc_auc_score(y[idx], df[c].to_numpy()[idx])); P[c].append(average_precision_score(y[idx], df[c].to_numpy()[idx]))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    rows = []
    for c in cols:
        pl, ml = c.split("|"); la, ha = ci(A[c]); lp, hp = ci(P[c])
        rows.append({"pipeline": pl, "model": ml, "AUROC": round(roc_auc_score(y, df[c]), 3), "AUROC_lo": round(la, 3),
                     "AUROC_hi": round(ha, 3), "AUPRC": round(average_precision_score(y, df[c]), 3),
                     "AUPRC_lo": round(lp, 3), "AUPRC_hi": round(hp, 3), "AUROC_ci_excludes_0.5": la > 0.5})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, f"bootstrap_table4_{CASE}.csv"), index=False)
    md = [f"# Table IV with patient-clustered bootstrap 95% CIs (test: 887 days, 196 patients, 62 deaths; B={B}, {used} valid draws, seed {SEED})\n",
          "| pipeline | model | AUROC | 95% CI | AUPRC | 95% CI |", "|---|---|--:|--:|--:|--:|"]
    md += [f"| {r.pipeline} | {r.model} | {r.AUROC:.3f} | [{r.AUROC_lo:.3f}, {r.AUROC_hi:.3f}] | {r.AUPRC:.3f} | [{r.AUPRC_lo:.3f}, {r.AUPRC_hi:.3f}] |" for r in out.itertuples()]
    open(os.path.join(RES, "BOOTSTRAP_TABLE4.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nsaved -> {RES}/bootstrap_table4_{CASE}.csv, table4_test_probs_{CASE}.csv, BOOTSTRAP_TABLE4.md")


if __name__ == "__main__":
    main()
