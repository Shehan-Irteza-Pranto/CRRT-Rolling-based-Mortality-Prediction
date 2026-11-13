"""
subpopulation_analysis_combined.py
=========================================================================================
Tables X / XI for the REPORTED combined ensemble (tuned transformer fusion + top-3 classical,
forced logistic stack, 1-day, test AUROC 0.738) — the model behind exp_4/BEST_COMBINED_ARCHITECTURE.md.

Identical to paper_materials/subpopulation_analysis.py EXCEPT the per-circuit-day probabilities:
the sub-populations (S0..S3), sizes, deaths, prevalence and clinical phenotype are model-independent
(Ward hierarchical clustering on the clinical test features) and therefore match the old tables exactly;
only the AUROC / AUPRC / F2 (Table X) and the per-subpop AUROC (Table XI) change with the new model.

Predictions come from exp_4/results/combined_stack_{test,val}_probs_1day.csv (produced by
exp_4/regenerate_stack_test_probs.py). Everything else (clustering, threshold, phenotype, figures)
reuses the original module's helpers verbatim. Writes to paper_materials/subpop_combined/.
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
sys.path.insert(0, _HERE)
sys.path.insert(0, _AS_DIR)
import subpopulation_analysis as SA                                          # noqa: E402 (reuse helpers)
from abl_config import feature_dir, KEY_COLS                                 # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, fbeta_score, recall_score)
from scipy.cluster.hierarchy import linkage, fcluster

EXP4 = os.path.join(_AS_DIR, "exp_4")
MODEL_NAME = "combined ensemble (transformer + 3 classical, stack)"


def _stack_probs(case):
    """Return (vprob, tprob, yva, yte, df, feats) for the reported combined stack, with the clinical
    test matrix in the canonical machine_only order (identical clustering to the old tables)."""
    keyp = os.path.join(feature_dir("current", "denoised", "machine_only", False, case), "test.csv")
    keys = pd.read_csv(keyp, usecols=KEY_COLS + ["label"])
    pid_te = keys["PID"].to_numpy(); yte = keys["label"].to_numpy().astype(int)
    df, feats = SA._clinical_test(case, pid_te, yte)                          # canonical order + asserts

    test = pd.read_csv(os.path.join(EXP4, "results", f"combined_stack_test_probs_{case}.csv"))
    merged = df[KEY_COLS].merge(test[KEY_COLS + ["prob"]], on=KEY_COLS, how="left")
    assert merged["prob"].notna().all(), "stack test probs did not cover all test circuit-days"
    tprob = merged["prob"].to_numpy()
    val = pd.read_csv(os.path.join(EXP4, "results", f"combined_stack_val_probs_{case}.csv"))
    return val["prob"].to_numpy(), tprob, val["label"].to_numpy().astype(int), yte, df, feats


def run(case="1day"):
    outdir = os.path.join(_HERE, "subpop_combined"); os.makedirs(outdir, exist_ok=True)
    print(f"[subpop-combined] case={case}  model={MODEL_NAME}")
    vprob, tprob, yva, yte, df, feats = _stack_probs(case)
    thr = SA._opt_threshold(yva, vprob)
    overall_auroc = roc_auc_score(yte, tprob)
    print(f"  overall test AUROC {overall_auroc:.4f} (val-F1 threshold {thr:.3f})")

    X = df[feats].to_numpy(float)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    Z = linkage(Xs, method="ward")
    K1, K, sil, gaps = SA._levels(Z, Xs, yte)
    labels = fcluster(Z, t=K, criterion="maxclust") - 1
    print(f"  Level 1 = {K1}, Level 2 (reported) = {K}  | gaps {list(gaps)}")

    # ---- Table X: per-subpopulation discrimination ----
    pred = (tprob >= thr).astype(int)
    rows = []
    for s in range(K):
        m = labels == s
        n, npos = int(m.sum()), int(yte[m].sum())
        row = {"subpop": f"S{s}", "n_days": n, "n_patients": int(df.loc[m, "PID"].nunique()),
               "n_deaths": npos, "prevalence": round(npos / max(n, 1), 4)}
        if npos >= SA.MIN_POS_EVAL and (yte[m] == 0).any():
            row.update({"AUROC": round(roc_auc_score(yte[m], tprob[m]), 4),
                        "AUPRC": round(average_precision_score(yte[m], tprob[m]), 4),
                        "F2": round(fbeta_score(yte[m], pred[m], beta=2, zero_division=0), 4),
                        "Recall": round(recall_score(yte[m], pred[m], zero_division=0), 4), "status": "OK"})
        else:
            row.update({"AUROC": np.nan, "AUPRC": np.nan, "F2": np.nan, "Recall": np.nan,
                        "status": "NE (too few deaths)"})
        rows.append(row)
    # overall row
    orow = {"subpop": "Overall", "n_days": len(yte), "n_patients": int(df["PID"].nunique()),
            "n_deaths": int(yte.sum()), "prevalence": round(int(yte.sum()) / len(yte), 4),
            "AUROC": round(overall_auroc, 4), "AUPRC": round(average_precision_score(yte, tprob), 4),
            "F2": round(fbeta_score(yte, pred, beta=2, zero_division=0), 4),
            "Recall": round(recall_score(yte, pred, zero_division=0), 4), "status": "OK"}
    res = pd.DataFrame(rows + [orow])
    res.to_csv(os.path.join(outdir, f"subpopulation_results_{case}.csv"), index=False)

    # ---- Table XI: phenotype (cluster means) + per-subpop AUROC ----
    pv = [v for v in SA.PHENO_VARS if v in df.columns]
    pheno = df.assign(subpop=[f"S{s}" for s in labels]).groupby("subpop")[pv].mean().round(2)
    overall_mean = df[pv].mean().round(2); overall_mean.name = "ALL"
    pheno = pd.concat([pheno, overall_mean.to_frame().T])
    auroc_by_s = {r.subpop: r.AUROC for r in res.itertuples()}
    pheno["AUROC"] = [auroc_by_s.get(ix, auroc_by_s.get("Overall")) if ix != "ALL"
                      else auroc_by_s.get("Overall") for ix in pheno.index]
    pheno.to_csv(os.path.join(outdir, f"subpopulation_phenotype_{case}.csv"))

    _figures(outdir, case, Z, labels, K, K1, res, Xs, yte)

    print("\n=== TABLE X — discrimination within each clinical sub-population ===")
    print(res.to_string(index=False))
    print("\n=== TABLE XI — clinical phenotype (cluster means) + AUROC ===")
    print(pheno.to_string())
    print(f"\nsaved -> {outdir}/subpopulation_{{results,phenotype}}_{case}.csv "
          f"+ dendrogram_{case}.png + tsne_{case}.png")
    return res, pheno


def _figures(outdir, case, Z, labels, K, K1, res, Xs, yte):
    """Ward dendrogram (links colored by sub-population, Level cut lines, AUROC legend) + 3-D t-SNE.
    Same palette/logic as subpopulation_analysis.py; AUROC labels are the combined model's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.manifold import TSNE
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    SUBPOP_COLOR = {s: cycle[s % len(cycle)] for s in range(K)}
    ABOVE_COLOR = "#b0b0b0"
    auroc_by_s = {r.subpop: r.AUROC for r in res.itertuples()}

    # ---- dendrogram ----
    n_obs = Z.shape[0] + 1
    node_cluster = {i: int(labels[i]) for i in range(n_obs)}
    for j in range(Z.shape[0]):
        ca, cb = node_cluster.get(int(Z[j, 0])), node_cluster.get(int(Z[j, 1]))
        node_cluster[n_obs + j] = ca if (ca is not None and ca == cb) else None

    def _link_color(k):
        c = node_cluster.get(k)
        return SUBPOP_COLOR[c] if c is not None else ABOVE_COLOR

    # paper-sized fonts
    TITLE, LABEL, TICK, LEG, LEGT, LVL = 19, 17, 14, 16, 17, 15
    fig, ax = plt.subplots(figsize=(12, 7))
    dn = dendrogram(Z, no_labels=True, link_color_func=_link_color, ax=ax)
    for k, name, style in [(K1, f"Level 1 ({K1})", ":"), (K, f"Level 2 ({K})", "--")]:
        h = SA._cut_height(Z, k)
        ax.axhline(h, ls=style, c="gray", lw=1.6)
        ax.text(0.008, h, name, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=LVL)
    xpos = {}
    for p, leaf in enumerate(dn["leaves"]):
        xpos.setdefault(int(labels[leaf]), []).append(10 * p + 5)
    order = sorted(xpos, key=lambda s: np.mean(xpos[s]))
    ax.set_xticks([np.mean(xpos[s]) for s in order])
    ax.set_xticklabels([f"S{s}\n(n={len(xpos[s])})" for s in order], fontsize=TICK + 1, fontweight="bold")
    for t, s in zip(ax.get_xticklabels(), order):
        t.set_color(SUBPOP_COLOR[s])
    ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=TICK)
    legend_h = [Patch(color=SUBPOP_COLOR[s],
                      label=f"S{s}: n={len(xpos[s])}, AUROC="
                            + (f"{auroc_by_s.get(f'S{s}'):.3f}" if pd.notna(auroc_by_s.get(f"S{s}")) else "NE"))
                for s in range(K)]
    ax.legend(handles=legend_h, fontsize=LEG, loc="upper right", title="Level-2 sub-populations",
              title_fontsize=LEGT)
    ax.set_ylabel("Ward distance", fontsize=LABEL)
    ax.set_xlabel("test circuit-days (grouped by sub-population)", fontsize=LABEL)
    ax.set_title(f"Hierarchical clustering of test patients — clinical features ({case})\n"
                 f"combined ensemble (transformer + 3 classical)", fontsize=TITLE)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"dendrogram_{case}.png"), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)

    # ---- 3-D t-SNE ----
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    emb = TSNE(n_components=3, perplexity=30, init="pca", learning_rate="auto",
               random_state=0).fit_transform(Xs)
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    for s in range(K):
        m = labels == s
        ax1.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=20, alpha=0.75,
                    color=SUBPOP_COLOR[s], label=f"S{s} (n={int(m.sum())})")
    for ax3 in (ax1,):
        ax3.set_xlabel("t-SNE-0", fontsize=LABEL, labelpad=10)
        ax3.set_ylabel("t-SNE-1", fontsize=LABEL, labelpad=10)
        ax3.set_zlabel("t-SNE-2", fontsize=LABEL, labelpad=10)
        ax3.tick_params(labelsize=TICK - 2)
    ax1.set_title(f"Sub-populations (Level 2, K={K})", fontsize=TITLE)
    ax1.legend(fontsize=LEG, loc="upper right", markerscale=2)
    ax2 = fig.add_subplot(122, projection="3d")
    surv = yte == 0
    ax2.scatter(emb[surv, 0], emb[surv, 1], emb[surv, 2], s=12, alpha=0.2, c="#c8c8c8", label="survived")
    ax2.scatter(emb[~surv, 0], emb[~surv, 1], emb[~surv, 2], s=55, alpha=0.95, c="black",
                marker="^", label="died")
    ax2.set_xlabel("t-SNE-0", fontsize=LABEL, labelpad=10)
    ax2.set_ylabel("t-SNE-1", fontsize=LABEL, labelpad=10)
    ax2.set_zlabel("t-SNE-2", fontsize=LABEL, labelpad=10)
    ax2.tick_params(labelsize=TICK - 2)
    ax2.set_title("Outcome", fontsize=TITLE); ax2.legend(fontsize=LEG, loc="upper right", markerscale=1.6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"tsne_{case}.png"), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day")
    run(ap.parse_args().case)
