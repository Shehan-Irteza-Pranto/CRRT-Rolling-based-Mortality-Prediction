"""
make_interpretability_figs.py — cross-pipeline mRMR selection map
=================================================================
Reads the 8 mRMR selection caches (clinical / machine-base / machine-enriched / combined x 1day/2day)
and builds the "which features were selected where" picture for the interpretability section:
  * mrmr_selection_map.png  — two-panel selected/not heatmap (machine features | clinical features)
                              x the pipelines that use them, y-labels colored by feature family;
  * mrmr_selection_counts.csv — per feature: domain, family, how many pipelines selected it (stability);
  * mrmr_family_counts.csv    — family x pipeline count table (base-marginal / base-dynamic / enriched
                              groups for machine; LAB / PTI groups for clinical).
Output dir: paper_materials/interpretability/. Style matches subpopulation_analysis.py (Agg, tight_layout).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, _AS)
from abl_config import is_machine_col                                          # noqa: E402

OUT = os.path.join(_HERE, "interpretability"); os.makedirs(OUT, exist_ok=True)
FS_A = os.path.join(_AS, "results", "feature_selection")
FS_B = os.path.join(_AS, "exp_2", "results", "feature_selection")

# pipeline label -> selection json
PIPES = {
    "clinical\n1day":     os.path.join(FS_B, "current_denoised_clinical_only_per_day_1day.json"),
    "clinical\n2day":     os.path.join(FS_B, "current_denoised_clinical_only_per_day_2day.json"),
    "machine-base\n1day": os.path.join(FS_A, "current_denoised_machine_only_per_day_1day.json"),
    "machine-base\n2day": os.path.join(FS_A, "current_denoised_machine_only_per_day_2day.json"),
    "machine-rich\n1day": os.path.join(FS_A, "rich_enh_denoised_machine_only_per_day_1day.json"),
    "machine-rich\n2day": os.path.join(FS_A, "rich_enh_denoised_machine_only_per_day_2day.json"),
    "combined\n1day":     os.path.join(FS_B, "current_denoised_combined_per_day_1day.json"),
    "combined\n2day":     os.path.join(FS_B, "current_denoised_combined_per_day_2day.json"),
}
MACHINE_PIPES = ["machine-base\n1day", "machine-base\n2day", "machine-rich\n1day", "machine-rich\n2day",
                 "combined\n1day", "combined\n2day"]
CLIN_PIPES = ["clinical\n1day", "clinical\n2day", "combined\n1day", "combined\n2day"]

ENRICHED_SUFFIX = ("dom_period", "spec_entropy", "bandpow_lo", "bandpow_hi", "perm_entropy",
                   "cross_rate", "max_run_above", "eod_delta", "eod_slope")
DYNAMIC_SUFFIX = ("slope", "trend_r2", "mad", "max_jump", "ac1", "exc_frac")  # base within-day dynamics


def load_sel():
    return {p: set(json.load(open(pth))["selected"]) for p, pth in PIPES.items()}


def machine_family(f):
    if f.endswith(ENRICHED_SUFFIX):
        for grp, sfx in [("spectral", ("dom_period", "spec_entropy", "bandpow_lo", "bandpow_hi")),
                         ("complexity", ("perm_entropy",)),
                         ("excursion", ("cross_rate", "max_run_above")),
                         ("end-of-day", ("eod_delta", "eod_slope"))]:
            if f.endswith(sfx):
                return "enriched:" + grp
    if f.endswith(DYNAMIC_SUFFIX):
        return "base:dynamic"
    return "base:marginal"


def clin_family(f):
    if f.startswith("LAB_"):
        return "LAB"
    return "PTI"


FAMILY_COLOR = {
    "enriched:spectral": "#d95f02", "enriched:complexity": "#e6550d",
    "enriched:excursion": "#fdae6b", "enriched:end-of-day": "#a63603",
    "base:dynamic": "#1b9e77", "base:marginal": "#333333",
    "LAB": "#2166ac", "PTI": "#666666",
}


def panel(ax, feats, pipes, sel, fam_fn, title):
    mat = np.array([[1 if f in sel[p] else 0 for p in pipes] for f in feats])
    ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=1.4)
    ax.set_xticks(range(len(pipes))); ax.set_xticklabels([p.replace("\n", " ") for p in pipes],
                                                         rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=5.2)
    for tick, f in zip(ax.get_yticklabels(), feats):
        tick.set_color(FAMILY_COLOR.get(fam_fn(f), "black"))
    ax.set_xticks(np.arange(-.5, len(pipes), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=9)


def order_machine(feats):
    chan = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE", "TMP", "PFD", "ARD"]
    def key(f):
        c = next((i for i, ch in enumerate(chan) if f.startswith(ch)), len(chan))
        fam = machine_family(f)
        famrank = {"base:marginal": 0, "base:dynamic": 1}.get(fam, 2)  # enriched last within channel
        return (c, famrank, f)
    return sorted(feats, key=key)


def order_clin(feats):
    return sorted(feats, key=lambda f: (0 if f.startswith("LAB_") else 1, f))


def main():
    sel = load_sel()
    mach_union = order_machine({f for p in MACHINE_PIPES for f in sel[p] if is_machine_col(f)})
    clin_union = order_clin({f for p in CLIN_PIPES for f in sel[p] if not is_machine_col(f)})

    h = max(len(mach_union), len(clin_union)) * 0.14 + 1.6
    fig, (axm, axc) = plt.subplots(1, 2, figsize=(13, h))
    panel(axm, mach_union, MACHINE_PIPES, sel, machine_family,
          "Machine features (orange = enriched, green = base dynamics, grey = base marginal)")
    panel(axc, clin_union, CLIN_PIPES, sel, clin_family, "Clinical features (blue = LAB, grey = PTI)")
    fig.suptitle("mRMR feature selection across pipelines (shaded = selected)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "mrmr_selection_map.png"), dpi=300)
    plt.close(fig)

    # per-feature stability counts
    rows = []
    for f in mach_union:
        rows.append({"feature": f, "domain": "machine", "family": machine_family(f),
                     "n_selected": sum(f in sel[p] for p in MACHINE_PIPES), "n_pipelines": len(MACHINE_PIPES)})
    for f in clin_union:
        rows.append({"feature": f, "domain": "clinical", "family": clin_family(f),
                     "n_selected": sum(f in sel[p] for p in CLIN_PIPES), "n_pipelines": len(CLIN_PIPES)})
    counts = pd.DataFrame(rows).sort_values(["domain", "n_selected", "feature"], ascending=[True, False, True])
    counts.to_csv(os.path.join(OUT, "mrmr_selection_counts.csv"), index=False)

    # family x pipeline counts
    fam_rows = []
    for p in PIPES:
        for f in sel[p]:
            fam = machine_family(f) if is_machine_col(f) else clin_family(f)
            fam_rows.append({"pipeline": p.replace("\n", " "), "family": fam})
    fam = (pd.DataFrame(fam_rows).groupby(["pipeline", "family"]).size()
           .unstack(fill_value=0))
    fam.to_csv(os.path.join(OUT, "mrmr_family_counts.csv"))

    n_enr_rich1 = sum(machine_family(f).startswith("enriched") for f in sel["machine-rich\n1day"])
    n_enr_rich2 = sum(machine_family(f).startswith("enriched") for f in sel["machine-rich\n2day"])
    print(f"machine union {len(mach_union)} feats, clinical union {len(clin_union)} feats")
    print(f"enriched features kept: machine-rich 1day = {n_enr_rich1}/30, 2day = {n_enr_rich2}/30")
    print(f"wrote map + counts to {OUT}")


if __name__ == "__main__":
    main()
