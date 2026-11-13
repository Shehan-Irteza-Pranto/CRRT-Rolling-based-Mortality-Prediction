"""
subpopulation_analysis.py
=========================================================================================
Sub-population robustness analysis of the BEST machine-only model (the exp_3 ensemble:mean
= rich-classical + attention-CNN1D, averaged) on the held-out TEST set.

The test circuit-days are clustered by **Ward hierarchical clustering on the clinical features**
(model-independent). The Ward dendrogram reveals a 2-level hierarchy: a dominant 2-way split
(Level 1) and a finer structure (Level 2); the number of sub-populations is read off the
dendrogram (not pre-fixed). We report the ensemble's AUROC / AUPRC within each Level-2
sub-population, describe each group's clinical phenotype, and visualize with a 3-D t-SNE and a
level-annotated dendrogram (matching the project's reference figures).

Outputs (paper_materials/subpop/):
  subpopulation_results_<case>.csv     per-subpopulation size / prevalence / AUROC / AUPRC
  subpopulation_phenotype_<case>.csv   cluster means of key clinical variables
  dendrogram_<case>.png                Ward dendrogram with Level-1 / Level-2 cut lines
  tsne_<case>.png                      3-D t-SNE: by sub-population and by outcome
=========================================================================================
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_AS_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _AS_DIR)
from abl_config import CLINICAL_ONLY, feature_dir, KEY_COLS                    # noqa: E402

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (roc_auc_score, average_precision_score, silhouette_score,
                             fbeta_score, recall_score, precision_recall_curve)
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

_META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
K_MAX = 8
MIN_POS_EVAL = 3            # a sub-pop needs >= this many deaths to report AUROC
MIN_CLUSTER_FRAC = 0.05     # "substantial" cluster threshold (fraction of the test set)
CLASSICAL_MODELS = ["GradientBoosting", "RF", "LightGBM", "XGBoost", "LogisticRegression"]
TOP_K_CLASSICAL = 3
PHENO_VARS = ["PTI_Age", "PTI_Gender_Male", "PTI_CrrtInitiationWeight_kg", "PTI_IcuBmi_kg_m2",
              "PTI_CharlsonCoMorbidityScore", "PTI_Apache2_total_partial",
              "LAB_LactateUs", "LAB_AlbuminUs", "LAB_INRUs", "LAB_PlateletsUs",
              "LAB_pHUs", "LAB_BicarbonateUs", "LAB_CreatinineUs", "LAB_UreaUs"]


def _ensemble_probs(case):
    """Reproduce the exp_3 ensemble:mean (val-ranked top-3 classical families + DL), aligned to
    the machine_only test order. Returns (val_prob, test_prob, yva, yte, pid_te)."""
    cz = np.load(os.path.join(_AS_DIR, "exp_3", "features", "results", f"classical_probs_{case}.npz"))
    dz = np.load(os.path.join(_AS_DIR, "exp_3", "dl", "results", f"dl_probs_{case}.npz"))
    yva, yte, pid_te = cz["yva"].astype(int), cz["yte"].astype(int), cz["pid_te"]
    assert np.array_equal(cz["yte"], dz["yte"]), "classical/DL test order mismatch"
    cands = {}
    for fs in ("base", "rich"):
        for m in CLASSICAL_MODELS:
            vk = f"val_{fs}_{m}"
            if vk in cz.files:
                cands[f"cls:{fs}:{m}"] = (cz[vk], cz[f"test_{fs}_{m}"])
    cands["dl:attn_cnn1d"] = (dz["val"], dz["test"])
    val_auc = {n: roc_auc_score(yva, v) for n, (v, _) in cands.items()}
    keep, seen = [], set()
    for n in sorted(val_auc, key=val_auc.get, reverse=True):
        fam = n.split(":")[-1]
        if n.startswith("dl:") or (fam not in seen and len([k for k in keep if k.startswith("cls:")]) < TOP_K_CLASSICAL):
            keep.append(n); seen.add(fam)
    if "dl:attn_cnn1d" not in keep:
        keep.append("dl:attn_cnn1d")
    Vv = np.column_stack([cands[n][0] for n in keep]).mean(1)
    Vt = np.column_stack([cands[n][1] for n in keep]).mean(1)
    print(f"  ensemble base models: {keep}")
    return Vv, Vt, yva, yte, pid_te


def _clinical_test(case, pid_te, yte):
    """Clinical feature matrix for the TEST circuit-days, row-aligned to the prob order."""
    df = pd.read_csv(os.path.join(CLINICAL_ONLY, case, "test.csv"))
    keys = pd.read_csv(os.path.join(feature_dir("current", "denoised", "machine_only", False, case),
                                    "test.csv"), usecols=KEY_COLS + ["label"])
    df = keys.merge(df, on=KEY_COLS, how="left", suffixes=("", "_clin"))
    assert len(df) == len(pid_te) and np.array_equal(df["PID"].to_numpy(), pid_te), "clinical align"
    feats = [c for c in df.columns if c not in _META and not c.endswith("_clin")]
    return df, feats


def _levels(Z, Xs, y):
    """Read the two coarsest dendrogram levels. Level 1 = the dominant split (largest merge-height
    gap). Level 2 = the next level that introduces a SUBSTANTIAL secondary group, i.e. the smallest
    K >= 3 whose partition has >= 3 substantial clusters (>= MIN_CLUSTER_FRAC of the test set);
    falls back to 4. Returns (K1, K2, silhouette_by_k, gaps)."""
    H = Z[::-1, 2][:K_MAX]
    gaps = np.round(H[:-1] - H[1:], 2)
    K1 = int(np.argmax(gaps)) + 2
    n = len(Xs); min_n = MIN_CLUSTER_FRAC * n
    sil, K2 = {}, None
    for k in range(2, K_MAX + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(lab)) != k:
            continue
        sil[k] = round(float(silhouette_score(Xs, lab)), 4)
        n_sub = int(sum((lab == c).sum() >= min_n for c in np.unique(lab)))
        if K2 is None and k >= 3 and n_sub >= 3:
            K2 = k
    return K1, (K2 or 4), sil, gaps


def _opt_threshold(y, s):
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))])


def _cut_height(Z, k):
    d = Z[:, 2]
    return (d[-(k - 1)] + d[-k]) / 2.0 if k >= 2 else d[-1] * 1.1


def run(case="1day"):
    outdir = os.path.join(_HERE, "subpop"); os.makedirs(outdir, exist_ok=True)
    print(f"[subpop] case={case}")
    vprob, tprob, yva, yte, pid_te = _ensemble_probs(case)
    thr = _opt_threshold(yva, vprob)
    overall_auroc = roc_auc_score(yte, tprob)
    print(f"  ensemble overall test AUROC {overall_auroc:.4f} (val-F1 threshold {thr:.3f})")

    df, feats = _clinical_test(case, pid_te, yte)
    X = df[feats].to_numpy(float)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))

    Z = linkage(Xs, method="ward")
    K1, K, sil, gaps = _levels(Z, Xs, yte)
    labels = fcluster(Z, t=K, criterion="maxclust") - 1
    print(f"  merge-height gaps (top): {list(gaps)}")
    print(f"  silhouette by K: {sil}")
    print(f"  dendrogram levels -> Level 1 = {K1} (dominant split), Level 2 = {K} (reported)")

    # ---- per-subpopulation metrics (Level-2 clusters; tiny/low-event ones flagged NE) ----
    pred = (tprob >= thr).astype(int)
    rows = []
    for s in range(K):
        m = labels == s
        n, npos = int(m.sum()), int(yte[m].sum())
        row = {"subpop": f"S{s}", "n_days": n, "n_patients": int(df.loc[m, "PID"].nunique()),
               "n_deaths": npos, "prevalence": round(npos / max(n, 1), 4)}
        if npos >= MIN_POS_EVAL and (yte[m] == 0).any():
            row.update({"AUROC": round(roc_auc_score(yte[m], tprob[m]), 4),
                        "AUPRC": round(average_precision_score(yte[m], tprob[m]), 4),
                        "F2": round(fbeta_score(yte[m], pred[m], beta=2, zero_division=0), 4),
                        "Recall": round(recall_score(yte[m], pred[m], zero_division=0), 4),
                        "status": "OK"})
        else:
            row.update({"AUROC": np.nan, "AUPRC": np.nan, "F2": np.nan, "Recall": np.nan,
                        "status": "NE (too few deaths)"})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"subpopulation_results_{case}.csv"), index=False)

    # ---- phenotype description ----
    pv = [v for v in PHENO_VARS if v in df.columns]
    pheno = df.assign(subpop=[f"S{s}" for s in labels]).groupby("subpop")[pv].mean().round(2)
    overall_mean = df[pv].mean().round(2); overall_mean.name = "ALL"
    pheno = pd.concat([pheno, overall_mean.to_frame().T])
    pheno.to_csv(os.path.join(outdir, f"subpopulation_phenotype_{case}.csv"))

    # ---- shared sub-population palette (SAME colors in dendrogram + t-SNE) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    SUBPOP_COLOR = {s: cycle[s % len(cycle)] for s in range(K)}     # S0..S{K-1} -> fixed color
    ABOVE_COLOR = "#b0b0b0"                                         # links spanning >1 sub-pop (above cut)
    auroc_by_s = {r.subpop: r.AUROC for r in res.itertuples()}      # for labels/legend

    # ---- dendrogram, links colored by sub-population (matches t-SNE), + Level cut lines ----
    n_obs = Z.shape[0] + 1
    node_cluster = {i: int(labels[i]) for i in range(n_obs)}        # leaf -> its Level-2 sub-pop
    for j in range(Z.shape[0]):                                     # internal node monochrome iff subtree in one sub-pop
        ca, cb = node_cluster.get(int(Z[j, 0])), node_cluster.get(int(Z[j, 1]))
        node_cluster[n_obs + j] = ca if (ca is not None and ca == cb) else None

    def _link_color(k):
        c = node_cluster.get(k)
        return SUBPOP_COLOR[c] if c is not None else ABOVE_COLOR

    fig, ax = plt.subplots(figsize=(9, 5))
    dn = dendrogram(Z, no_labels=True, link_color_func=_link_color, ax=ax)
    for k, name, style in [(K1, f"Level 1 ({K1})", ":"), (K, f"Level 2 ({K})", "--")]:
        h = _cut_height(Z, k)
        ax.axhline(h, ls=style, c="gray")
        ax.text(0.995, h, name, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    # colored S-labels centered under each sub-population's leaves (leaf p sits at x = 10*p + 5)
    xpos = {}
    for p, leaf in enumerate(dn["leaves"]):
        xpos.setdefault(int(labels[leaf]), []).append(10 * p + 5)
    order = sorted(xpos, key=lambda s: np.mean(xpos[s]))            # left-to-right order on the axis
    ax.set_xticks([np.mean(xpos[s]) for s in order])
    ax.set_xticklabels([f"S{s}\n(n={len(xpos[s])})" for s in order], fontsize=8)
    for t, s in zip(ax.get_xticklabels(), order):
        t.set_color(SUBPOP_COLOR[s])
    ax.tick_params(axis="x", length=0)
    legend_h = [Patch(color=SUBPOP_COLOR[s],
                      label=f"S{s}: n={len(xpos[s])}, AUROC="
                            + (f"{auroc_by_s.get(f'S{s}'):.3f}" if pd.notna(auroc_by_s.get(f"S{s}")) else "NE"))
                for s in range(K)]
    ax.legend(handles=legend_h, fontsize=7.5, loc="upper right", title="Level-2 sub-populations")
    ax.set_ylabel("Ward distance"); ax.set_xlabel("test circuit-days (grouped by sub-population)")
    ax.set_title(f"Hierarchical clustering of test patients — clinical features ({case})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"dendrogram_{case}.png"), dpi=130); plt.close(fig)

    # ---- 3-D t-SNE: by sub-population (same palette) and by outcome ----
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    emb = TSNE(n_components=3, perplexity=30, init="pca", learning_rate="auto",
               random_state=0).fit_transform(Xs)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    for s in range(K):
        m = labels == s
        ax1.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=12, alpha=0.7,
                    color=SUBPOP_COLOR[s], label=f"S{s} (n={int(m.sum())})")
    ax1.set_xlabel("t-SNE-0"); ax1.set_ylabel("t-SNE-1"); ax1.set_zlabel("t-SNE-2")
    ax1.set_title(f"Sub-populations (Level 2, K={K})"); ax1.legend(fontsize=8, loc="upper right")
    ax2 = fig.add_subplot(122, projection="3d")
    surv = yte == 0                                                 # neutral colors so "died" red ≠ any sub-pop
    ax2.scatter(emb[surv, 0], emb[surv, 1], emb[surv, 2], s=8, alpha=0.2, c="#c8c8c8", label="survived")
    ax2.scatter(emb[~surv, 0], emb[~surv, 1], emb[~surv, 2], s=34, alpha=0.95, c="black",
                marker="^", label="died")
    ax2.set_xlabel("t-SNE-0"); ax2.set_ylabel("t-SNE-1"); ax2.set_zlabel("t-SNE-2")
    ax2.set_title("Outcome"); ax2.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"tsne_{case}.png"), dpi=130); plt.close(fig)

    # ---- console summary ----
    ev = res[res.status == "OK"]
    print("\n=== per-sub-population results ===")
    print(res.to_string(index=False))
    print(f"\noverall test AUROC {overall_auroc:.4f} | evaluable {len(ev)}/{K} | "
          f"subpop AUROC mean±sd {ev.AUROC.mean():.4f}±{ev.AUROC.std(ddof=1):.4f} "
          f"(range {ev.AUROC.min():.3f}-{ev.AUROC.max():.3f})")
    print("\n=== phenotype (cluster means of key clinical variables) ===")
    print(pheno.to_string())
    print(f"\nsaved -> {outdir}/  (results csv, phenotype csv, dendrogram_{case}.png, tsne_{case}.png)")
    return res, pheno, K1, K, overall_auroc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day")
    run(ap.parse_args().case)
