"""
run_table7_updated.py — regenerate Table VII (8 classical models x 3 pipelines, 1-day, per-day,
current denoised) with the UPDATED feature list: clinical-only = 60 clinical, machine-only = 40 machine,
combined = 40 machine + 60 clinical (consistent 40/60 budget).

Non-destructive: patches feature_select's mRMR count defaults + uses a FRESH selection-cache dir + a NEW
output CSV, so the original config / cache / CSV are untouched. Writes:
  results/classical_current_denoised_upd4060.csv   (raw matrix)
  results/table7_updated_1day.csv                  (clean Table VII, one row per model/pipeline)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "models"))

import feature_select as FS                                     # noqa: E402
# updated mRMR counts: (n_machine, n_clinical, n_single) — used by select_columns' defaults
FS.select_columns.__defaults__ = (40, 60, 40)
FS.SELECTION_DIR = os.path.join(_HERE, "results", "feature_selection_upd4060")
os.makedirs(FS.SELECTION_DIR, exist_ok=True)

import pandas as pd                                             # noqa: E402
from classical import run_matrix, build_model_registry         # noqa: E402

RES = os.path.join(_HERE, "results")
RAW = os.path.join(RES, "classical_current_denoised_upd4060.csv")
CLEAN = os.path.join(RES, "table7_updated_1day.csv")

PIPE_LABEL = {"clinical_only": "Clinical-only", "machine_only": "Machine-only", "combined": "Combined"}
MODEL_LABEL = {"LogisticRegression": "LogReg", "DecisionTree": "DecisionTree", "RF": "RandomForest",
               "GradientBoosting": "GradBoost", "AdaBoost": "AdaBoost", "KNN": "KNN",
               "LightGBM": "LightGBM", "XGBoost": "XGBoost"}
MODEL_ORDER = ["LogReg", "DecisionTree", "RandomForest", "GradBoost", "AdaBoost", "KNN", "LightGBM", "XGBoost"]
PIPE_ORDER = ["Clinical-only", "Machine-only", "Combined"]


def main():
    models = list(build_model_registry().keys())
    print(f"running updated Table VII: {models}\n  clinical=60, machine=40, combined=40+60", flush=True)
    df = run_matrix(["clinical_only", "machine_only", "combined"], ["1day"], [False], models,
                    variant="current", peak="denoised", tag="", out=RAW)

    d = df[df["case"] == "1day"].copy()          # only per_day was run, so no scope filter needed
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "pipeline": PIPE_LABEL.get(r["feature_set"], r["feature_set"]),
            "model": MODEL_LABEL.get(r["model"], r["model"]),
            "n_selected": int(r.get("n_selected", 0)),
            "AUROC": round(r["full_test_AUROC"], 3), "AUPRC": round(r["full_test_AUPRC"], 3),
            "F1": round(r["full_test_F1"], 3), "Precision": round(r["full_test_Precision"], 3),
            "Recall": round(r["full_test_Recall"], 3),
            "AUROC_1to1": round(r["r1to1_AUROC_mean"], 3), "AUPRC_1to1": round(r["r1to1_AUPRC_mean"], 3),
            "AUROC_1to2": round(r["r1to2_AUROC_mean"], 3), "AUPRC_1to2": round(r["r1to2_AUPRC_mean"], 3)})
    out = pd.DataFrame(rows)
    out["pipeline"] = pd.Categorical(out["pipeline"], PIPE_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out = out.sort_values(["pipeline", "model"]).reset_index(drop=True)
    out.to_csv(CLEAN, index=False)
    print("\n=== UPDATED TABLE VII (1-day, per-day, current denoised; features 40/60) ===")
    print(out.to_string(index=False))
    print(f"\nsaved -> {CLEAN}\n         {RAW}")


if __name__ == "__main__":
    main()
