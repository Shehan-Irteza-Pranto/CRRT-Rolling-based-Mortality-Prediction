"""
subpopulation_eval.py
=====================
Subpopulation distribution-shift evaluation (complement to the resampled bootstrap).

Cluster the TEST patients into distinct subpopulations using CLINICAL features
(Ward hierarchical clustering, cut to 4 = Level-2), then report each trained
classical model's AUROC WITHIN each subpopulation, and the mean ± std ACROSS the
four subpopulations. The std measures performance variability across distinctly
clustered patient subgroups; comparing combined vs clinical per subpopulation tests
whether machine pressure data adds signal within clinical-defined subgroups.

Visualizes the subpopulations with a dendrogram + 3-D t-SNE (as in the reference figure).
Reuses the already-trained saved classical models (results/saved_models/classical/).

Usage:  python subpopulation_eval.py --case 1day   (then 2day)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "models"))
from abl_config import (feature_dir, RESULTS_DIR, RANDOM_STATE,              # noqa: E402
                        DL_REDUCE, DL_REDUCE_ENABLED)
SAVED_DL = os.path.join(RESULTS_DIR, "saved_models", "dl")
DL_ENCODERS = ["cnn1d", "rnn", "transformer"]
DL_PIPELINES = ["machine_only", "combined"]

META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
KEYS = ["PID", "Visit_no", "Day"]
MODELS = ["XGBoost", "LightGBM", "LogisticRegression", "RF", "GradientBoosting",
          "KNN", "DecisionTree", "AdaBoost"]
PIPELINES = ["clinical_only", "machine_only", "combined"]
SCOPES = {"per_day": False, "agg": True}
N_SUB = 4                                   # Level-2 subpopulations
MIN_POS_EVAL = 10                           # min positive events for a subpop AUROC to be "evaluable" (EPV-style rule)
SAVED = os.path.join(RESULTS_DIR, "saved_models", "classical")


# --------------------------------------------------------------------------- clustering
def _feats(df):
    return [c for c in df.columns if c not in META]


def cluster_subpopulations(case, variant="current", peak="denoised"):
    """Ward clustering of the TEST circuit-days on clinical features -> 4 subpopulations."""
    path = os.path.join(feature_dir(variant, peak, "clinical_only", False, case), "test.csv")
    df = pd.read_csv(path)
    feats = _feats(df)
    X = df[feats].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))

    Z = linkage(Xs, method="ward")
    labels = fcluster(Z, t=N_SUB, criterion="maxclust")        # 1..N_SUB
    df_keys = df[KEYS].copy(); df_keys["subpop"] = labels - 1   # -> 0..N_SUB-1
    lut = {(int(r.PID), int(r.Visit_no), int(r.Day)): int(r.subpop) for r in df_keys.itertuples()}
    return Z, Xs, df_keys, lut


def _cut_height(Z, k):
    """Linkage distance that yields exactly k clusters (midpoint of the k-1 / k merges)."""
    d = Z[:, 2]
    return (d[-(k - 1)] + d[-k]) / 2.0 if k >= 2 else d[-1] * 1.1


def plot_dendrogram(Z, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(Z, no_labels=True, color_threshold=_cut_height(Z, N_SUB), ax=ax)
    for k, name, style in [(2, "Level 1 (2)", ":"), (N_SUB, f"Level 2 ({N_SUB})", "--")]:
        h = _cut_height(Z, k)
        ax.axhline(h, ls=style, c="gray"); ax.text(0.99, h, name, transform=ax.get_yaxis_transform(),
                                                    ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("height (Ward distance)"); ax.set_title("Hierarchical clustering of test patients (clinical features)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_tsne3d(Xs, subpops, out):
    emb = TSNE(n_components=3, random_state=RANDOM_STATE, init="pca",
               perplexity=30, learning_rate="auto").fit_transform(Xs)
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    for s in range(N_SUB):
        m = subpops == s
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=10, label=f"L2S{s}")
    ax.set_xlabel("t-SNE-0"); ax.set_ylabel("t-SNE-1"); ax.set_zlabel("t-SNE-2")
    ax.legend(title="Subpopulation (Level 2)", loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- per-model eval
def _apply_prep(prep, X):
    imp, sc, binmask = prep
    Xi = imp.transform(X)
    cont = ~binmask
    if cont.any():
        Xi[:, cont] = sc.transform(Xi[:, cont])
    return Xi


def _predict_test(bundle, case, variant, peak):
    """Reload a saved model bundle, predict proba on its test table. Returns (keys_df, y, prob)."""
    fs, scope = bundle["feature_set"], bundle["scope"]
    df = pd.read_csv(os.path.join(feature_dir(variant, peak, fs, scope == "agg", case), "test.csv"))
    X = df[bundle["features"]].to_numpy(dtype=float)
    Xp = _apply_prep(bundle["preprocessor"], X)[:, bundle["selected_idx"]]
    prob = bundle["model"].predict_proba(Xp)[:, 1]
    return df[KEYS], df["label"].to_numpy(int), prob


def eval_models(case, lut, variant="current", peak="denoised"):
    rows = []
    for fs in PIPELINES:
        for scope in SCOPES:
            for model in MODELS:
                mid = f"{variant}_{peak}_{fs}_{scope}_{case}_{model}"
                path = os.path.join(SAVED, mid + ".joblib")
                if not os.path.exists(path):
                    continue
                b = joblib.load(path)
                keys, y, prob = _predict_test(b, case, variant, peak)
                sub = np.array([lut.get((int(k.PID), int(k.Visit_no), int(k.Day)), -1)
                                for k in keys.itertuples()])
                rows.append(_subpop_row("classical", fs, scope, model, y, prob, sub))
    return pd.DataFrame(rows)


def _subpop_row(family, fs, scope, model, y, prob, sub):
    """Per-subpopulation AUROC row shared by classical + DL."""
    row = {"family": family, "feature_set": fs, "scope": scope, "model": model,
           "overall_AUROC": round(roc_auc_score(y, prob), 4)}
    aurocs = []
    for s in range(N_SUB):
        m = sub == s
        npos = int(y[m].sum())
        row[f"L2S{s}_n"] = int(m.sum()); row[f"L2S{s}_npos"] = npos
        evaluable = npos >= MIN_POS_EVAL and (y[m] == 0).any()
        row[f"L2S{s}_status"] = "OK" if evaluable else "NE"
        if evaluable:
            a = roc_auc_score(y[m], prob[m]); aurocs.append(a)
            row[f"L2S{s}_AUROC"] = round(a, 4)
        else:
            row[f"L2S{s}_AUROC"] = np.nan      # not evaluable: too few positive events
    row["n_eval_subpops"] = len(aurocs)
    row["subpop_mean"] = round(float(np.mean(aurocs)), 4) if aurocs else np.nan
    row["subpop_std"] = round(float(np.std(aurocs, ddof=1)), 4) if len(aurocs) > 1 else np.nan
    return row


# --------------------------------------------------------------------------- DL eval
@__import__("functools").lru_cache(maxsize=None)
def _dl_datasets(case, pipeline, scope, variant, peak, encoder):
    """Build (test_dataset, keys, y, dims) once per config; train rebuilt for the scalers."""
    import torch  # noqa: F401
    from dl_data import AggDataset
    spec = DL_REDUCE.get(encoder) if DL_REDUCE_ENABLED else None
    want_machine = False if pipeline == "combined" else None     # combined: machine via sequence
    tr = AggDataset(case, "train", pipeline, scope, variant, peak, reduce=spec,
                    want_machine=want_machine)
    te = AggDataset(case, "test", pipeline, scope, variant, peak, reduce=spec,
                    want_machine=want_machine, mach_scaler=tr.mach_scaler, clin_scaler=tr.clin_scaler)
    keys = np.array([[int(v) for v in k] for k in te.seq.keys])
    return te, keys, te.seq.y.numpy().astype(int), (te.machine_dim, te.clinical_dim, te.seq_feat_dim)


def _dl_predict(model, te, mach_dim, clin_dim):
    import torch
    from torch.utils.data import DataLoader
    model.eval(); probs = []
    with torch.no_grad():
        for x, mask, m, c, y in DataLoader(te, batch_size=128):
            kw = {}
            if mach_dim:
                kw["machine_summary"] = m
            if clin_dim:
                kw["clinical_summary"] = c
            probs.append(torch.sigmoid(model(x, mask, **kw)).numpy())
    return np.concatenate(probs)


def eval_dl_models(case, lut, variant="current", peak="denoised"):
    import glob as _glob
    import torch
    from dl_aggregated import build_agg_model
    rows = []
    for pipeline in DL_PIPELINES:
        for scope in SCOPES:
            for encoder in DL_ENCODERS:
                pts = sorted(_glob.glob(os.path.join(
                    SAVED_DL, f"{variant}_{peak}_{pipeline}_{scope}_{encoder}_{case}_s*.pt")))
                if not pts:
                    continue
                te, keys, y, (mach_dim, clin_dim, in_ch) = _dl_datasets(
                    case, pipeline, scope, variant, peak, encoder)
                probs = []
                for pt in pts:                                  # seed ensemble
                    ck = torch.load(pt, map_location="cpu", weights_only=False)
                    hp = ck["best_hp"]; sk = dict(hp.get("seq_kwargs") or {}); sk["in_ch"] = ck["in_ch"]
                    model = build_agg_model(pipeline, scope, seq_encoder=encoder,
                                            machine_summary_dim=ck["machine_dim"],
                                            clinical_summary_dim=ck["clinical_dim"],
                                            fusion_dim=hp.get("fusion_dim", 128),
                                            dropout=hp.get("dropout", 0.3), seq_kwargs=sk,
                                            clinical_hidden=tuple(ck.get("clinical_hidden", (256, 128))))
                    model.load_state_dict(ck["state_dict"])
                    probs.append(_dl_predict(model, te, mach_dim, clin_dim))
                prob = np.mean(probs, axis=0)
                sub = np.array([lut.get((int(a), int(b), int(d)), -1) for a, b, d in keys])
                rows.append(_subpop_row("deep", pipeline, scope, encoder, y, prob, sub))

    # clinical_only DL = MLP on tabular mRMR clinical features
    from dl_clinical_mlp import TabularClinicalDataset, MLP
    for scope in SCOPES:
        pts = sorted(_glob.glob(os.path.join(
            SAVED_DL, f"{variant}_{peak}_clinical_only_{scope}_mlp_{case}_s*.pt")))
        if not pts:
            continue
        tr = TabularClinicalDataset(case, "train", scope, variant, peak)
        te = TabularClinicalDataset(case, "test", scope, variant, peak, prep=tr.prep)
        y = te.y.numpy().astype(int); keys = te.keys
        probs = []
        for pt in pts:
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            m = MLP(ck["in_dim"], hidden=tuple(ck["hidden"]), dropout=ck["dropout"])
            m.load_state_dict(ck["state_dict"]); m.eval()
            with torch.no_grad():
                probs.append(torch.sigmoid(m(te.X)).numpy())
        prob = np.mean(probs, axis=0)
        sub = np.array([lut.get((int(k[0]), int(k[1]), int(k[2])), -1) for k in keys])
        rows.append(_subpop_row("deep", "clinical_only", scope, "mlp", y, prob, sub))
    return pd.DataFrame(rows)


def main():
    global MIN_POS_EVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1day")
    ap.add_argument("--variant", default="current")
    ap.add_argument("--peak", default="denoised")
    ap.add_argument("--min-pos", type=int, default=MIN_POS_EVAL,
                    help="min positive events for a subpop AUROC to be evaluable (else flagged NE)")
    a = ap.parse_args()
    MIN_POS_EVAL = a.min_pos

    out_dir = os.path.join(RESULTS_DIR, "subpopulation", a.case)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[subpop] clustering test patients (clinical features) -> {N_SUB} subpopulations")
    Z, Xs, df_keys, lut = cluster_subpopulations(a.case, a.variant, a.peak)
    df_keys.to_csv(os.path.join(out_dir, "subpop_assignments.csv"), index=False)
    print("  subpop sizes:", df_keys["subpop"].value_counts().sort_index().to_dict())

    plot_dendrogram(Z, os.path.join(out_dir, "dendrogram.png"))
    plot_tsne3d(Xs, df_keys["subpop"].to_numpy(), os.path.join(out_dir, "tsne3d.png"))
    print(f"  wrote dendrogram.png + tsne3d.png")

    res_c = eval_models(a.case, lut, a.variant, a.peak)
    print(f"[subpop] classical: {len(res_c)} models")
    res_d = eval_dl_models(a.case, lut, a.variant, a.peak)
    print(f"[subpop] deep: {len(res_d)} models")
    res = pd.concat([res_c, res_d], ignore_index=True)
    res.to_csv(os.path.join(out_dir, "per_subpop_auroc.csv"), index=False)
    print(f"[subpop] wrote per_subpop_auroc.csv ({len(res)} rows)\n")

    if res.empty or "feature_set" not in res.columns:
        print("[subpop] no trained models found in results/saved_models/ — run the benchmarks first:\n"
              "         python run_benchmark.py --variant current --peak denoised --scopes per_day agg\n"
              "         python run_dl_benchmark.py --gpus 0 1 --variant current --peak denoised --seeds 5\n"
              "         then re-run this script.")
        return

    # headline: best model per pipeline (across classical+DL) by subpop_mean, with across-subpop std
    print("=== best model per pipeline: AUROC mean ± std ACROSS the 4 subpopulations ===")
    for fs in PIPELINES:
        sub = res[(res.feature_set == fs) & res.subpop_mean.notna()]
        if sub.empty:
            continue
        b = sub.loc[sub.subpop_mean.idxmax()]
        print(f"  {fs:13} {b['family']}/{b['model']}/{b['scope']:8} "
              f"subpop {b['subpop_mean']:.3f} ± {b['subpop_std']:.3f}  (overall {b['overall_AUROC']:.3f})  "
              f"per-subpop=[{b['L2S0_AUROC']}, {b['L2S1_AUROC']}, {b['L2S2_AUROC']}, {b['L2S3_AUROC']}]")


if __name__ == "__main__":
    main()
