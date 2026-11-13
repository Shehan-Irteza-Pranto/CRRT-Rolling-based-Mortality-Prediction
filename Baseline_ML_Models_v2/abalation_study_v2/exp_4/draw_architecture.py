"""draw_architecture.py — figure-style block diagram of the best combined model (Transformer + 3-classical stack)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
C = {"in": "#e8eef7", "dl": "#fde3d0", "cls": "#e5efe0", "meta": "#e7dcf0", "out": "#f6d6da", "edge": "#5b6472"}


def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=C["edge"], lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, p1, p2, style="-|>", lw=1.4, color=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color or C["edge"], shrinkA=2, shrinkB=2, zorder=1))


fig, ax = plt.subplots(figsize=(12.5, 8.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")

ax.text(6.5, 8.7, "Best combined model — Transformer + 3-classical stacked ensemble  (1-day test AUROC 0.738)",
        ha="center", fontsize=12, fontweight="bold")
# section labels
ax.text(3.05, 8.15, "LEVEL-0  base learners", ha="center", fontsize=10, style="italic", color="#333")
ax.text(11.1, 8.15, "LEVEL-1  meta-learner", ha="center", fontsize=10, style="italic", color="#333")

# ---- inputs ----
box(ax, 0.2, 6.2, 2.3, 0.9, "machine sequence\n[144 × 30]\n(10-min windows)", C["in"], 8.2)
box(ax, 0.2, 4.9, 2.3, 0.8, "clinical\n[40 mRMR]", C["in"], 8.2)
box(ax, 0.2, 1.5, 2.3, 1.2, "tabular\n20 machine + 40 clinical\n(mRMR, impute+scale)", C["in"], 8.2)

# ---- (A) transformer-fusion DL ----
box(ax, 3.1, 6.2, 2.7, 0.9, "Transformer encoder\nLinear 30→64 · [CLS] · PE ·\n1×(4-head SA, FFN 128, GELU)", C["dl"], 7.6)
box(ax, 3.1, 4.9, 2.7, 0.8, "clinical MLP\n40→64 · BN · ReLU · Drop", C["dl"], 7.8)
box(ax, 6.25, 5.55, 2.0, 1.05, "late fusion head\nconcat[64+64]→64→1\nsigmoid", C["dl"], 8, bold=True)
ax.text(4.45, 7.2, "(A) Transformer-fusion deep network", ha="center", fontsize=8.5, style="italic", color="#8a4a20")

# ---- (B,C,D) classical ----
for i, (yy, lbl) in enumerate([(2.55, "classical #1"), (1.8, "classical #2"), (1.05, "classical #3")]):
    box(ax, 3.1, yy, 2.7, 0.6, f"{lbl}  (top-3 by val)\nGB / RF / LGBM / XGB / LR", C["cls"], 7.4)
ax.text(4.45, 3.2, "(B–D) top-3 tuned classical models", ha="center", fontsize=8.5, style="italic", color="#3d6b2e")

# ---- probabilities ----
for yy, t in [(6.0, "p_DL"), (2.85, "p₁"), (2.1, "p₂"), (1.35, "p₃")]:
    box(ax, 8.7, yy, 0.9, 0.5, t, "#ffffff", 9, bold=True)

# ---- meta ----
box(ax, 10.2, 3.2, 2.4, 1.5, "Logistic meta-learner\nfit on VALIDATION\n[p_DL, p₁, p₂, p₃]", C["meta"], 8.5, bold=True)
box(ax, 10.5, 1.3, 1.8, 0.9, "P(death)\n≥ max-F1-val\n→ label", C["out"], 8.5, bold=True)

# ---- arrows ----
arrow(ax, (2.5, 6.65), (3.1, 6.65))                 # seq -> transformer
arrow(ax, (2.5, 5.3), (3.1, 5.3))                   # clinical -> mlp
arrow(ax, (5.8, 6.5), (6.25, 6.2))                  # transformer -> fusion
arrow(ax, (5.8, 5.3), (6.25, 5.95))                 # mlp -> fusion
arrow(ax, (8.25, 6.05), (8.7, 6.25))                # fusion -> p_DL
for yy_src, yy_dst in [(2.85, 2.85), (2.1, 2.1), (1.35, 1.35)]:
    arrow(ax, (5.8, yy_dst + 0.3), (8.7, yy_dst + 0.25))
for yy in [1.5, 1.5]:
    arrow(ax, (2.5, 2.1), (3.1, 2.55)); arrow(ax, (2.5, 2.0), (3.1, 2.1)); arrow(ax, (2.5, 1.9), (3.1, 1.35))
# probs -> meta
for yy in [6.25, 3.1, 2.35, 1.6]:
    arrow(ax, (9.6, yy), (10.2, min(max(yy, 3.4), 4.5)) if yy > 5 else (10.2, 3.95), color="#999")
arrow(ax, (11.4, 3.2), (11.4, 2.2), lw=1.6)          # meta -> output

fig.tight_layout()
out = os.path.join(FIG, "best_combined_architecture.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", out)
