"""
training/plots.py
=================
Per-rung diagnostic curves, generated from the artifacts train.py writes:

  * training_history/training_history_<tag>.csv  -> per-epoch train/val loss + val AUROC
  * predictions/<tag>_<case>_seed<seed>.npz      -> test (y_true, y_score) for ROC / PR

For every (tag, case) it emits two figures into results/curves/:
  <tag>_<case>_training_curves.png  — train loss, val loss, and val AUROC vs epoch
                                       (one faint line per seed + the across-seed mean)
  <tag>_<case>_test_curves.png      — ROC and precision-recall curves on the test set
                                       (per seed + mean AUROC / AP in the legend)

Usage
-----
    python plots.py                 # all rungs found under results/
    python plots.py --tag cnn1d     # just one rung
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HISTORY_DIR, PRED_DIR, CURVES_DIR


# ---------------------------------------------------------------- training curves

def _mean_over_seeds(series_by_seed):
    """Average a list of unequal-length 1D arrays over their common-length prefix.
    Seeds early-stop at different epochs, so we average up to the shortest run."""
    if not series_by_seed:
        return np.array([]), np.array([])
    n = min(len(s) for s in series_by_seed)
    stack = np.vstack([s[:n] for s in series_by_seed])
    return np.arange(1, n + 1), stack.mean(axis=0)


def plot_training_curves(hist, tag, case, outdir):
    g = hist[hist["case"] == case]
    if g.empty:
        return None
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4.5))
    train_series, val_series, auc_series = [], [], []
    for seed, sg in g.groupby("seed"):
        sg = sg.sort_values("epoch")
        ep = sg["epoch"].to_numpy()
        ax_loss.plot(ep, sg["train_loss"], color="tab:blue",  alpha=0.25, lw=1)
        ax_loss.plot(ep, sg["val_loss"],   color="tab:orange", alpha=0.25, lw=1)
        ax_auc.plot(ep, sg["val_auroc"],   color="tab:green",  alpha=0.25, lw=1)
        # mark the early-stop / best epoch
        best = sg[sg["best_so_far"] == 1]["epoch"]
        if len(best):
            ax_auc.axvline(best.max(), color="tab:green", alpha=0.08)
        train_series.append(sg["train_loss"].to_numpy())
        val_series.append(sg["val_loss"].to_numpy())
        auc_series.append(sg["val_auroc"].to_numpy())

    for series, color, label, ax in [
        (train_series, "tab:blue",  "train loss", ax_loss),
        (val_series,   "tab:orange", "val loss",  ax_loss),
        (auc_series,   "tab:green", "val AUROC",  ax_auc),
    ]:
        ex, ey = _mean_over_seeds(series)
        if len(ex):
            ax.plot(ex, ey, color=color, lw=2.5, label=f"{label} (mean)")

    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{tag} / {case} — training & validation loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)
    ax_auc.set_xlabel("epoch"); ax_auc.set_ylabel("validation AUROC")
    ax_auc.set_title(f"{tag} / {case} — validation AUROC")
    ax_auc.legend(); ax_auc.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, f"{tag}_{case}_training_curves.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ---------------------------------------------------------------- test ROC / PR curves

def plot_test_curves(pred_files, tag, case, outdir, split="balanced"):
    """ROC + PR curves on the test set. split='balanced' matches the reported
    balanced-test protocol; 'full' uses the real-prevalence test set."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.5))
    aucs, aps = [], []
    for f in sorted(pred_files):
        d = np.load(f, allow_pickle=True)
        y, s = d["y_true"], d["y_score"]
        if split == "balanced":
            idx = d["bal_idx"]; y, s = y[idx], s[idx]
        if len(np.unique(y)) < 2:
            continue
        seed = int(d["seed"]) if "seed" in d else 0
        fpr, tpr, _ = roc_curve(y, s); roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, s); ap = average_precision_score(y, s)
        aucs.append(roc_auc); aps.append(ap)
        ax_roc.plot(fpr, tpr, alpha=0.4, lw=1.2, label=f"seed {seed} (AUC {roc_auc:.3f})")
        ax_pr.plot(rec, prec, alpha=0.4, lw=1.2, label=f"seed {seed} (AP {ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title(f"{tag} / {case} — ROC ({split} test)\n"
                     f"AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}" if aucs else
                     f"{tag} / {case} — ROC ({split} test)")
    ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.3)
    ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")
    ax_pr.set_title(f"{tag} / {case} — precision-recall ({split} test)\n"
                    f"AP {np.mean(aps):.3f} ± {np.std(aps):.3f}" if aps else
                    f"{tag} / {case} — precision-recall ({split} test)")
    ax_pr.legend(fontsize=7); ax_pr.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, f"{tag}_{case}_test_curves.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ---------------------------------------------------------------- driver

def generate(tag_filter=None, split="balanced"):
    os.makedirs(CURVES_DIR, exist_ok=True)
    written = []

    # training curves from history CSVs
    for hist_csv in sorted(glob.glob(os.path.join(HISTORY_DIR, "training_history_*.csv"))):
        tag = os.path.basename(hist_csv)[len("training_history_"):-len(".csv")]
        if tag_filter and tag_filter not in tag:
            continue
        hist = pd.read_csv(hist_csv)
        for case in sorted(hist["case"].unique()):
            p = plot_training_curves(hist, tag, case, CURVES_DIR)
            if p:
                written.append(p)

    # test ROC/PR from prediction npz files, grouped by (tag, case)
    groups = {}
    for npz in glob.glob(os.path.join(PRED_DIR, "*.npz")):
        d = np.load(npz, allow_pickle=True)
        tag = str(d["tag"]); case = str(d["case"])
        if tag_filter and tag_filter not in tag:
            continue
        groups.setdefault((tag, case), []).append(npz)
    for (tag, case), files in sorted(groups.items()):
        p = plot_test_curves(files, tag, case, CURVES_DIR, split=split)
        if p:
            written.append(p)

    print(f"[plots] wrote {len(written)} figures to {CURVES_DIR}")
    for p in written:
        print(f"  {os.path.basename(p)}")
    return written


def _parse():
    ap = argparse.ArgumentParser(description="Generate per-rung training & test curves")
    ap.add_argument("--tag", default=None, help="only this rung (substring match, e.g. 'cnn1d')")
    ap.add_argument("--split", default="balanced", choices=["balanced", "full"],
                    help="test split for ROC/PR curves")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    generate(a.tag, a.split)
