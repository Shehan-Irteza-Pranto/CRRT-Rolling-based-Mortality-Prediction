"""
shap_best_combined.py — SHAP top-15 for the best COMBINED model's tabular backbone (1-day)
==========================================================================================
The best combined model is the Transformer + 3-classical ensemble (a logistic stack of a DL model + the
combined/clinical classical models). A stack mixing a sequence net with tabular models is not directly
feature-attributable, so we SHAP the **combined classical component** — the tabular model the stack is
built on: grouped mRMR (20 machine + 40 clinical) → tuned classifier. Reuses the exact combined table +
selection used by the ensemble. Writes 300-dpi beeswarm + bar + shap_top15_combined.csv to exp_4/.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                # noqa: E402 (paths + _combined_table/select/prep/CR)
from abl_config import is_machine_col                # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
CASE = "1day"
MODEL = "RF"          # best combined classical (RandomForest); the stack's tabular backbone
TOP = 15
FIG = os.path.join(_HERE, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_machine", type=int, default=20)
    ap.add_argument("--n_clinical", type=int, default=40)
    a = ap.parse_args()
    CF.N_MACHINE, CF.N_CLINICAL = a.n_machine, a.n_clinical    # grouped-mRMR counts for this config
    suffix = "" if (a.n_machine, a.n_clinical) == (20, 40) else f"_{a.n_machine}x{a.n_clinical}"
    tr = CF._combined_table(CASE, "train"); te = CF._combined_table(CASE, "test")
    feats = [c for c in tr.columns if c not in CF.META]
    Xtr, ytr = tr[feats].to_numpy(float), tr["label"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["label"].to_numpy(int)
    imp = CF.SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xte, prep = CF.scale_features(imp.transform(Xtr), imp.transform(Xte))
    idx, names = CF.select_columns(feats, Xtr, ytr, "combined", n_machine=CF.N_MACHINE, n_clinical=CF.N_CLINICAL)
    Xtr, Xte = Xtr[:, idx], Xte[:, idx]
    model = CF.CR._tune(MODEL, Xtr, ytr); model.fit(Xtr, ytr)
    auroc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"combined {MODEL}: test AUROC {auroc:.3f}  ({CF.N_MACHINE} machine + {CF.N_CLINICAL} clinical mRMR)", flush=True)

    Xte_df = pd.DataFrame(Xte, columns=names)
    sv = shap.TreeExplainer(model)(Xte_df, check_additivity=False)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    sv.feature_names = list(names)
    title = "SHAP — Top-15 Features for Enriched Combined Model (Windowed-Transformer Fusion Stack)"

    # keep ONLY the top-15 features so no "Sum of N other features" row is drawn
    mab = np.abs(sv.values).mean(0)
    top_idx = list(np.argsort(mab)[::-1][:TOP])
    sv_top = sv[:, top_idx]

    plt.figure(); shap.plots.beeswarm(sv_top, max_display=TOP, show=False)
    # highlight MACHINE features: box their y-axis labels in red
    from matplotlib.patches import Patch
    ax = plt.gca(); mach = {n for n in names if is_machine_col(n)}
    for t in ax.get_yticklabels():
        if t.get_text() in mach:
            t.set_color("#c0392b"); t.set_fontweight("bold")
            t.set_bbox(dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="#c0392b", lw=1.3))
    ax.legend(handles=[Patch(facecolor="none", edgecolor="#c0392b", lw=1.3, label="machine feature")],
              loc="lower right", fontsize=7, frameon=True)
    plt.title(title, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"shap_combined_beeswarm{suffix}.png"), dpi=300, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.bar(sv_top, max_display=TOP, show=False)
    plt.title(title, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"shap_combined_bar{suffix}.png"), dpi=300, bbox_inches="tight"); plt.close()

    tbl = pd.DataFrame({"feature": names, "mean_abs_shap": mab,
                        "is_machine": [is_machine_col(c) for c in names]}).sort_values(
        "mean_abs_shap", ascending=False).head(TOP)
    tbl.to_csv(os.path.join(OUT, f"shap_top15_combined{suffix}.csv"), index=False)
    print(f"\ntop-{TOP} features ({int(tbl.is_machine.sum())} machine / {TOP-int(tbl.is_machine.sum())} clinical):")
    print(tbl.to_string(index=False))
    print(f"\nsaved -> {FIG}/shap_combined_{{beeswarm,bar}}{suffix}.png + {OUT}/shap_top15_combined{suffix}.csv")


if __name__ == "__main__":
    main()
