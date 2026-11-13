"""
exp_2/make_shap.py
==================
SHAP for the BEST classical model per pipeline (highest full-test AUROC) in exp_2
(combined = fixed mRMR 20 machine + 40 clinical). Reloads the saved .joblib bundle,
explains the test set, and writes beeswarm + bar (top-15) per pipeline. For the combined
model, machine features are flagged so their rank among the clinical labs is visible.

Usage:  python make_shap.py [--peak denoised] [--top 15]
"""
import os, sys, glob, argparse, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from abl_config import RESULTS_DIR, feature_dir, is_machine_col      # noqa: E402

SAVED = os.path.join(RESULTS_DIR, "saved_models", "classical")
OUT   = os.path.join(RESULTS_DIR, "shap"); os.makedirs(OUT, exist_ok=True)
DPI   = 140
PIPES = ["clinical_only", "machine_only", "combined"]
TREE  = ("RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier",
         "XGBClassifier", "LGBMClassifier", "ExtraTreesClassifier")


def _apply_prep(prep, X):
    imp, sc, binmask = prep
    Xi = imp.transform(X); cont = ~binmask
    if cont.any():
        Xi[:, cont] = sc.transform(Xi[:, cont])
    return Xi


def _load_X(b, split):
    df = pd.read_csv(os.path.join(feature_dir(b["variant"], b["peak"], b["feature_set"],
                                              b["scope"] == "agg", b["case"]), f"{split}.csv"))
    X = _apply_prep(b["preprocessor"], df[b["features"]].to_numpy(float))[:, b["selected_idx"]]
    return pd.DataFrame(X, columns=b["selected_features"]), df["label"].to_numpy(int)


def best_per_pipeline(peak):
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"classical_current_{peak}.csv"))
    out = {}
    for fs in PIPES:
        s = df[(df.feature_set == fs) & df["full_test_AUROC"].notna()]
        if s.empty:
            continue
        r = s.loc[s["full_test_AUROC"].idxmax()]
        out[fs] = dict(case=r["case"], scope=r["scope"], model=r["model"],
                       auroc=r["full_test_AUROC"], machine_k=r.get("machine_k", ""))
    return out


def shap_one(fs, info, peak, top):
    mid = f"current_{peak}_{fs}_{info['scope']}_{info['case']}_{info['model']}"
    path = os.path.join(SAVED, mid + ".joblib")
    if not os.path.exists(path):
        print(f"  [skip] {fs}: {mid}.joblib not found"); return None
    b = joblib.load(path)
    Xte, yte = _load_X(b, "test"); Xtr, _ = _load_X(b, "train")
    mname = type(b["model"]).__name__
    if mname in TREE:
        sv = shap.TreeExplainer(b["model"])(Xte, check_additivity=False)
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
    else:                                          # AdaBoost / KNN / LogReg -> permutation
        bg = shap.utils.sample(Xtr, 60, random_state=0)
        sv = shap.PermutationExplainer(lambda x: b["model"].predict_proba(x)[:, 1], bg)(
            Xte, max_evals=2 * Xte.shape[1] + 1)
    sv.feature_names = list(Xte.columns)
    title = f"exp_2 SHAP — {fs} ({mname}/{info['scope']}/{info['case']}, AUROC {info['auroc']:.3f})"

    plt.figure(); shap.plots.beeswarm(sv, max_display=top, show=False)
    plt.title(title, fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"shap_beeswarm_{fs}.png"), dpi=DPI, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.bar(sv, max_display=top, show=False)
    plt.title(title, fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"shap_bar_{fs}.png"), dpi=DPI, bbox_inches="tight"); plt.close()

    mab = np.abs(sv.values).mean(0)
    tbl = pd.DataFrame({"feature": Xte.columns, "mean_abs_shap": mab,
                        "is_machine": [is_machine_col(c) for c in Xte.columns]})
    tbl = tbl.sort_values("mean_abs_shap", ascending=False).head(top)
    tbl.to_csv(os.path.join(OUT, f"shap_top{top}_{fs}.csv"), index=False)
    nmach = int(tbl["is_machine"].sum())
    print(f"  [{fs}] {mname}/{info['scope']}/{info['case']} (AUROC {info['auroc']:.3f}) "
          f"-> top-{top}: {nmach} machine feature(s); top5: {', '.join(tbl.feature.head(5))}")
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", default="denoised")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--out", default=None, help="output dir (default: RESULTS_DIR/shap)")
    a = ap.parse_args()
    global OUT, DPI
    DPI = a.dpi
    if a.out:
        OUT = a.out; os.makedirs(OUT, exist_ok=True)
    best = best_per_pipeline(a.peak)
    print("[exp_2 SHAP] best model per pipeline:")
    for fs, info in best.items():
        print(f"  {fs:<13} {info['model']}/{info['scope']}/{info['case']}  "
              f"AUROC={info['auroc']:.3f}  machine_k={info['machine_k']}")
    print("[exp_2 SHAP] computing ...")
    for fs, info in best.items():
        shap_one(fs, info, a.peak, a.top)
    print(f"[exp_2 SHAP] wrote plots + tables to {OUT}")


if __name__ == "__main__":
    main()
