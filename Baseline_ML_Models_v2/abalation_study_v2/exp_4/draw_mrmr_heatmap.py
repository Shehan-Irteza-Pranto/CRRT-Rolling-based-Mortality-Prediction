"""draw_mrmr_heatmap.py — paper-ready 4x3 heatmap of the grouped-mRMR sweep for the
Windowed-Transformer Fusion Stack. Reads results/mrmr_sweep_1day.csv.
  python draw_mrmr_heatmap.py --metric test   -> figures/mrmr_sweep_heatmap_test.png
  python draw_mrmr_heatmap.py --metric val    -> figures/mrmr_sweep_heatmap.png
NOTE: model selection is on VALIDATION AUROC; the test heatmap is a sensitivity view."""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, "figures"); os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--metric", choices=["val", "test"], default="test")
METRIC = ap.parse_args().metric
COL = "val_AUROC" if METRIC == "val" else "test_AUROC_ref"
LABEL = "validation AUROC" if METRIC == "val" else "test AUROC (held-out)"
OUTNAME = "mrmr_sweep_heatmap.png" if METRIC == "val" else "mrmr_sweep_heatmap_test.png"

df = pd.read_csv(os.path.join(_HERE, "results", "mrmr_sweep_1day.csv"))
N_MACH = sorted(df.n_machine.unique())            # rows (10,20,30,40)
N_CLIN = sorted(df.n_clinical.unique())           # cols (20,40,60)
M = np.full((len(N_MACH), len(N_CLIN)), np.nan)
for _, r in df.iterrows():
    M[N_MACH.index(r.n_machine), N_CLIN.index(r.n_clinical)] = r[COL]

best = df.loc[df[COL].idxmax()]
bi, bj = N_MACH.index(best.n_machine), N_CLIN.index(best.n_clinical)
ci, cj = N_MACH.index(20), N_CLIN.index(40)        # current model

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 6.4))
im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
lo, hi = np.nanmin(M), np.nanmax(M)
for i in range(len(N_MACH)):
    for j in range(len(N_CLIN)):
        v = M[i, j]
        norm = (v - lo) / (hi - lo + 1e-9) if not np.isnan(v) else 0
        ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—", ha="center", va="center", fontsize=15,
                color="white" if norm > 0.6 else "#1a1a1a",
                fontweight="bold" if (i == bi and j == bj) else "normal")

ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1, fill=False, edgecolor="#d1495b", lw=3.0))
ax.set_xticks(range(len(N_CLIN))); ax.set_xticklabels(N_CLIN, fontsize=14)
ax.set_yticks(range(len(N_MACH))); ax.set_yticklabels(N_MACH, fontsize=14)
ax.set_xlabel("n_clinical  (clinical features, mRMR)", fontsize=15)
ax.set_ylabel("n_machine  (machine features, mRMR)", fontsize=15)
ax.set_title(f"Windowed-Transformer Fusion Stack (1-day)\ngrouped-mRMR sweep — {LABEL}",
             fontsize=15, fontweight="bold", pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(LABEL, fontsize=13); cb.ax.tick_params(labelsize=12)
ax.plot([], [], color="#d1495b", lw=3.0, label=f"best ({int(best.n_machine)} × {int(best.n_clinical)}, {best[COL]:.3f})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=12)
fig.tight_layout()
out = os.path.join(FIG, OUTNAME)
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", out)
print(f"[{METRIC}] current 20x40 {M[ci,cj]:.4f} | best {int(best.n_machine)}x{int(best.n_clinical)} {best[COL]:.4f}")
