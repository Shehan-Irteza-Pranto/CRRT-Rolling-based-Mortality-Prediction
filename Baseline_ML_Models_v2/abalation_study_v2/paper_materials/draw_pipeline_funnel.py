"""draw_pipeline_funnel.py — data-attrition funnel figure for the CRRT signal-processing pipeline."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(FIG, exist_ok=True)

# (label, minute_rows, removed_note) for the minute-row funnel (stages 0-4)
STAGES = [
    ("Raw export", 6_603_507, "1,076 patients · 1,256 circuits"),
    ("Day numbering + cap (≤14)", 6_236_988, "−366,519 (Day > 14)"),
    ("Priming removal (RUN_TIME=0)", 6_215_753, "−21,235"),
    ("Downtime removal (no effluent)", 6_102_758, "−112,995  ·  −2 circuit-days"),
    ("Peak denoising (40-min, |z|>2)", 6_102_758, "0 dropped · 764,322 min (12.5%) smoothed"),
]
COH = [("Train", 3236, 201), ("Validation", 462, 30), ("Test", 887, 62)]
BLUE, ORANGE, GREEN, GREY = "#3d6b8e", "#e08a3c", "#4a8a5c", "#5b6472"

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1.15]})
fig.suptitle("CRRT signal-processing pipeline — data attrition", fontsize=14, fontweight="bold", y=0.98)

# ---- top: minute-row funnel (centered trapezoid bars) ----
maxw = max(s[1] for s in STAGES)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.4, len(STAGES) - 0.3); ax.axis("off")
ax.text(0, len(STAGES) - 0.35, "Minute-rows", ha="center", fontsize=10.5, style="italic", color="#333")
for i, (lbl, val, note) in enumerate(STAGES):
    y = len(STAGES) - 1 - i
    w = val / maxw
    ax.add_patch(FancyBboxPatch((-w, y - 0.32), 2 * w, 0.64, boxstyle="round,pad=0.005,rounding_size=0.02",
                                fc=BLUE, ec="white", lw=1.2, alpha=0.88))
    ax.text(0, y, f"{lbl}\n{val:,} rows", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.text(1.03, y, note, ha="left", va="center", fontsize=8.3, color=GREY)
    if i > 0:
        ax.annotate("", xy=(0, y + 0.34), xytext=(0, y + 0.66),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))

# ---- bottom: circuit-days -> cohort -> splits ----
ax2.set_xlim(0, 10); ax2.set_ylim(0, 3); ax2.axis("off")
ax2.text(5, 2.75, "Gap-aware reconstruction  →  Day ≤ 7 cap + mortality labels  →  model-ready cohort",
         ha="center", fontsize=9.5, style="italic", color="#333")

def cell(x, w, top, sub, fc):
    ax2.add_patch(FancyBboxPatch((x, 0.55), w, 1.35, boxstyle="round,pad=0.01,rounding_size=0.03",
                                 fc=fc, ec="white", lw=1.2, alpha=0.9))
    ax2.text(x + w / 2, 1.5, top, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax2.text(x + w / 2, 0.95, sub, ha="center", va="center", color="white", fontsize=8.2)

cell(0.2, 2.4, "5,696 circuit-days", "1,256 circuits · 1,076 patients\n(75.1% minutes valid/day)", GREY)
ax2.annotate("", xy=(2.95, 1.25), xytext=(2.65, 1.25), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
cell(3.0, 2.6, "Cohort: 4,585 circuit-days", "976 patients · 293 deaths (6.4%)", GREEN)
ax2.annotate("", xy=(5.95, 1.25), xytext=(5.65, 1.25), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
x = 6.05
for lbl, cd, dth in COH:
    w = 1.25
    cell(x, w, lbl, f"{cd:,} cd\n{dth} deaths", ORANGE)
    x += w + 0.05

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIG, "signal_pipeline_funnel.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", out)
