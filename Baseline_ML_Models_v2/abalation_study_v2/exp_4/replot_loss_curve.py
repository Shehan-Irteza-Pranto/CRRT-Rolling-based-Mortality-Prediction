"""replot_loss_curve.py — mean±std train/val loss curves across the 5 seeds of the Windowed-Transformer
Fusion DL core, from results/loss_history_1day.csv (no retraining). Cleaner/representative than one noisy
seed. Marks the minimum mean-val-loss epoch. -> figures/loss_curve_windowed_transformer_fusion.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
h = pd.read_csv(os.path.join(_HERE, "results", "loss_history_1day.csv"))
MIN_SEEDS = 3                                              # only plot epochs backed by >= this many seeds

g = h.groupby("epoch")
n = g["seed"].count()
keep = n[n >= MIN_SEEDS].index.max()
e = np.arange(1, keep + 1)
tr_m = g["train_loss"].mean().loc[e]; tr_s = g["train_loss"].std().loc[e]
va_m = g["val_loss"].mean().loc[e];   va_s = g["val_loss"].std().loc[e]

# epoch used = per-seed early-stop (max val AUROC) epoch; the reported ensemble averages these
# checkpoints. val AUROC is noisy on the small val set, so members stop at scattered epochs — the
# MEDIAN is the representative "epoch used".
stop_ep = [int(h[h.seed == s].loc[h[h.seed == s]["val_auroc"].idxmax(), "epoch"])
           for s in sorted(h["seed"].unique())]
used = int(np.median(stop_ep))

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8, 5.4))
ax.plot(e, tr_m, color="#264653", lw=2.4, label="training loss (mean)")
ax.fill_between(e, tr_m - tr_s, tr_m + tr_s, color="#264653", alpha=0.15)
ax.plot(e, va_m, color="#e76f51", lw=2.4, label="validation loss (mean)")
ax.fill_between(e, va_m - va_s, va_m + va_s, color="#e76f51", alpha=0.15)
ax.axvline(used, color="#2a9d8f", ls="--", lw=2.0,
           label=f"checkpoint epoch (median early stop = {used})")
ax.set_xlabel("epoch", fontsize=15)
ax.set_ylabel("binary cross-entropy loss", fontsize=15)
ax.set_title("Windowed-Transformer Fusion (DL core) — loss curves (1-day)\n"
             "d_model 64 · 1 layer · 4 heads · ff 128 · dropout 0.4 · mean ± std over 5 seeds",
             fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.25); ax.legend(fontsize=12, frameon=False)
fig.text(0.5, -0.02, "Bands = ±1 std across seeds. Training uses AdamW (lr 3e-4, wd 5e-3), weighted "
         "sampler, grad-clip 5.0; early stopping on validation AUROC (patience 15).",
         ha="center", fontsize=8.5, color="#5b6472", style="italic")
fig.tight_layout()
out = os.path.join(_HERE, "figures", "loss_curve_windowed_transformer_fusion.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", out, "| epochs 1-%d (>=%d seeds) | per-seed stop epochs %s | median (used) %d"
      % (keep, MIN_SEEDS, stop_ep, used))
