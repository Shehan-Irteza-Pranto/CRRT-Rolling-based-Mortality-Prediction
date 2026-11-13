"""draw_pipeline_pie.py — academic manuscript-ready donut of raw CRRT minute-row disposition (legend labels)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

RAW = 6_603_507
SLICES = [
    ("Retained (model-ready signal)",   6_102_758, "#3d6b8e"),
    ("Removed — Day > 14 cap",          366_519,   "#d98c4a"),
    ("Removed — Downtime (no effluent)", 112_995,  "#b5546a"),
    ("Removed — Priming (RUN_TIME = 0)", 21_235,   "#8a8f98"),
]
sizes = [s[1] for s in SLICES]; colors = [s[2] for s in SLICES]
pct = [100 * s / RAW for s in sizes]

fig, ax = plt.subplots(figsize=(9.6, 6.2))
wedges, _ = ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.8))
ax.set(aspect="equal")
# only the dominant slice labelled on-wedge
ang = np.deg2rad((wedges[0].theta1 + wedges[0].theta2) / 2)
ax.text(0.79 * np.cos(ang), 0.79 * np.sin(ang), f"{pct[0]:.1f}%", ha="center", va="center",
        color="white", fontsize=13, fontweight="bold")
# center total
ax.text(0, 0.10, f"{RAW:,}", ha="center", va="center", fontsize=15.5, fontweight="bold", color="#2a2f36")
ax.text(0, -0.16, "raw minute-rows", ha="center", va="center", fontsize=10.5, color="#5b6472")

# legend with counts + percentages
handles = [Patch(facecolor=c, edgecolor="white", label=f"{lbl}\n{s:,}  ({p:.1f}%)")
           for (lbl, s, c), p in zip(SLICES, pct)]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
          fontsize=10.3, handlelength=1.1, handleheight=1.1, labelspacing=1.1, borderaxespad=0)

ax.set_title("Disposition of raw CRRT pressure minute-rows", fontsize=13.5, fontweight="bold", pad=16)
fig.text(0.5, 0.02, "92.4% retained → 4,585 model-ready circuit-days (976 patients · 293 deaths). "
         "Peak denoising smooths 12.5% of retained minutes without removing rows.",
         ha="center", fontsize=8.8, color="#5b6472", style="italic")
fig.tight_layout(rect=[0, 0.035, 1, 1])
out = os.path.join(FIG, "signal_pipeline_pie.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", out)
