"""
rerun_clinical_mlp_60.py — clinical-only DL MLP (Table IV 'Clinical MLP') with UPDATED clinical mRMR
count 40 -> 60. Non-destructive: patches the parent feature_select (n_clinical=60) + a fresh selection
cache dir, then retrains 5 seeds via dl_clinical_mlp.train_clinical_mlp (per_day, 1-day, search-iter 12).
Reports the best-val seed's full/1:1/1:2 metrics. CPU (MLP is tiny). Writes results/clinical_mlp_60feat_1day.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
for p in [_AS, os.path.join(_AS, "models"), os.path.join(_AS, "eval")]:
    sys.path.insert(0, p)

import feature_select as FS                                    # noqa: E402 (parent models/feature_select.py)
FS.select_columns.__defaults__ = (40, 60, 40)                  # n_clinical -> 60 (clinical_only path uses it)
FS.SELECTION_DIR = os.path.join(_AS, "results", "feature_selection_clin60")
os.makedirs(FS.SELECTION_DIR, exist_ok=True)

import dl_clinical_mlp as DM                                   # noqa: E402
OUT = os.path.join(_HERE, "results", "clinical_mlp_60feat_1day.csv")


def main():
    rows = []
    for s in range(5):
        print(f"\n==== clinical MLP seed {s} (clinical mRMR = 60) ====", flush=True)
        r = DM.train_clinical_mlp("per_day", "1day", s, search_iter=12, log_every=25)
        rows.append(r)
        print(f"  seed {s}: n_selected={r['n_selected']}  val {r['best_val_auroc']}  "
              f"full AUROC {r['full_test_AUROC']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_HERE, "results", "clinical_mlp_60feat_allseeds_1day.csv"), index=False)
    best = df.loc[df["best_val_auroc"].idxmax()]                # best-val seed (matches paper selection)
    out = {"model": "Clinical MLP (clinical=60)", "sel_seed": int(best["seed"]),
           "n_selected": int(best["n_selected"]),
           "AUROC": round(best["full_test_AUROC"], 3), "AUPRC": round(best["full_test_AUPRC"], 3),
           "F1": round(best["full_test_F1"], 3), "Precision": round(best["full_test_Precision"], 3),
           "Recall": round(best["full_test_Recall"], 3),
           "AUROC_1to1": round(best["r1to1_AUROC_mean"], 3), "AUPRC_1to1": round(best["r1to1_AUPRC_mean"], 3),
           "AUROC_1to2": round(best["r1to2_AUROC_mean"], 3), "AUPRC_1to2": round(best["r1to2_AUPRC_mean"], 3)}
    pd.DataFrame([out]).to_csv(OUT, index=False)
    print("\n=== Clinical MLP @ clinical mRMR-60 (best-val seed) ===")
    print(f"  n_selected={out['n_selected']}  seed={out['sel_seed']}")
    print(f"  Full  AUROC {out['AUROC']}  AUPRC {out['AUPRC']}  F1 {out['F1']}  P {out['Precision']}  R {out['Recall']}")
    print(f"  1:1   AUROC {out['AUROC_1to1']}  AUPRC {out['AUPRC_1to1']}")
    print(f"  1:2   AUROC {out['AUROC_1to2']}  AUPRC {out['AUPRC_1to2']}")
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
