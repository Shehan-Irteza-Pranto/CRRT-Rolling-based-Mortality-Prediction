"""
sweep_nclin_dl.py — vary the DL CLINICAL MLP-branch feature count (N_CLIN_DL) from 10..80 for the
Windowed-Transformer Fusion Stack. The transformer-fusion DL is retrained (5 seeds) at each clinical-
branch size; the classical stack members are held at the best-features config (40 machine + 60 clinical)
and computed once. Reports validation AUROC (selection metric) + test AUROC (reference). Plots a line
graph -> figures/nclin_dl_sweep.png ; results/nclin_dl_sweep_1day.csv.  GPU + CPU.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stack_with_tuned_transformer as ST                 # noqa: E402
import ablation_ensemble_branch as ABL                    # noqa: E402
import sweep_mrmr_stack as SW                              # noqa: E402 (reuse _forced_stack_bestval)
CF = ST.CF
CASE = "1day"
SPLITS = ("train", "val", "test")
NCLIN_DL = [10, 20, 30, 40, 50, 60, 70, 80]
OUT = os.path.join(_HERE, "results", "nclin_dl_sweep_1day.csv")


def main():
    # classical stack members held at BEST-FEATURES (40 machine + 60 clinical) — independent of N_CLIN_DL
    CF.N_MACHINE, CF.N_CLINICAL = 40, 60
    print("classical members (combined 40x60 + clinical-only), computed once ...", flush=True)
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in SPLITS], "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in SPLITS], "clinical_only", per_seed=True)

    rows = []
    for k in NCLIN_DL:
        CF.N_CLIN_DL = k                                   # DL clinical MLP-branch size for this run
        data, clin_t = ABL._fusion_data(CASE)             # rebuilds the clinical tensor with mRMR-k
        yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
        print(f"[N_CLIN_DL={k}] clinical tensor dim = {clin_t['train'].shape[-1]}; training fusion ...", flush=True)
        tf_vp, tf_tp = ST._train_tf(data, clin_t)          # tuned transformer fusion, 5 seeds
        b = SW._forced_stack_bestval(tf_vp, tf_tp, comb, clin, yva, yte)
        rows.append({"n_clin_dl": k, "val_AUROC": round(b["val"], 4),
                     "test_AUROC": round(b["test"], 4), "sel_seed": b["seed"], "members": b["members"]})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  -> VAL {b['val']:.4f}  TEST {b['test']:.4f}  (seed {b['seed']}, {b['members']})", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nsaved -> {OUT}")
    _plot(df)


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(df.n_clin_dl, df.test_AUROC, "-o", color="#c0392b", lw=2.6, ms=8, label="test AUROC (held-out)")
    ax.plot(df.n_clin_dl, df.val_AUROC, "--s", color="#2a9d8f", lw=2.0, ms=7, label="validation AUROC (selection)")
    # mark the default (40) and the best-by-validation
    ax.axvline(40, color="#888", ls=":", lw=1.5)
    ax.text(40, ax.get_ylim()[0], " default (40)", fontsize=10, color="#555", va="bottom")
    bv = df.loc[df.val_AUROC.idxmax()]
    ax.scatter([bv.n_clin_dl], [bv.val_AUROC], s=180, facecolors="none", edgecolors="#2a9d8f", lw=2.5, zorder=5)
    for _, r in df.iterrows():
        ax.annotate(f"{r.test_AUROC:.3f}", (r.n_clin_dl, r.test_AUROC), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="#c0392b")
    ax.set_xlabel("N_CLIN_DL  —  clinical features in the DL MLP branch (mRMR)", fontsize=13)
    ax.set_ylabel("AUROC (1-day)", fontsize=14)
    ax.set_title("Windowed-Transformer Fusion Stack — DL clinical-branch size sweep\n"
                 "(classical members held at 40 machine + 60 clinical)", fontsize=12.5, fontweight="bold")
    ax.set_xticks(df.n_clin_dl); ax.grid(alpha=0.25); ax.legend(fontsize=12, frameon=False, loc="best")
    fig.tight_layout()
    out = os.path.join(_HERE, "figures", "nclin_dl_sweep.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
