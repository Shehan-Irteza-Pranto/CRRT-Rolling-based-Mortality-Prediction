"""
export_pathA_bundles.py — bundle the best-val-seed benchmark DL models (MLP/CNN1D/RNN/Transformer)
==================================================================================================
Path A weights already exist (results/saved_models/dl/*.pt, per seed). This copies the BEST-val seed
checkpoint (weights + architecture config) + its validation threshold into a self-contained exp_4 bundle,
and extracts the fitted tabular preprocessor for the branches that have one (clinical MLP: imputer+scaler
+mRMR-40; combined seq models: clinical-branch imputer+scaler+mRMR-40). The reduced-machine-sequence
preprocessing is the repo's deterministic signal pipeline (cached; documented in LOAD_AND_PREDICT.md).
Runs in the parent 'models/' import context (NOT exp_3/dl) to avoid the dl_data name collision.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
for p in [os.path.join(os.path.dirname(_AS), "Deep_Learning_Models", "models"),
          os.path.join(os.path.dirname(_AS), "Machine_Clinical_Models"),
          _AS, os.path.join(_AS, "models")]:
    sys.path.insert(0, p)
from dl_clinical_mlp import TabularClinicalDataset      # noqa: E402 (models/)
from dl_data import AggDataset                          # noqa: E402 (models/)

SAVED = os.path.join(_AS, "results", "saved_models", "dl")
CSV = os.path.join(_AS, "results", "dl_current_denoised.csv")
MODELS_DIR = os.path.join(_HERE, "models"); os.makedirs(MODELS_DIR, exist_ok=True)
NAME2M = {"MLP": "mlp", "CNN1D": "cnn1d", "RNN (BiLSTM/GRU)": "rnn", "Transformer": "transformer"}
PIPE2FS = {"Clinical-only": "clinical_only", "Machine-only": "machine_only", "Combined": "combined"}
FS2SHORT = {"clinical_only": "clinical", "machine_only": "machine", "combined": "combined"}


def _clinical_prep(case):
    ds = TabularClinicalDataset(case, "train", "per_day", "current", "denoised", prep=None)
    return ds.prep                                       # (imputer, scaler, selection_idx)


def _combined_clin_prep(case):
    ds = AggDataset(case, "train", "combined", "per_day", "current", "denoised")
    return ds.clin_scaler                                # (scaler, imputer, sel) or None


def main():
    d = pd.read_csv(CSV)
    a = pd.read_csv(os.path.join(_HERE, "results", "pathA_bestseed.csv"))
    rows = []
    for _, r in a.iterrows():
        fs = PIPE2FS[r.pipeline]; m = NAME2M[r.model]; case = r.case; seed = int(r.sel_seed)
        pt = f"current_denoised_{fs}_per_day_{m}_{case}_s{seed}.pt"
        ckpt = torch.load(os.path.join(SAVED, pt), map_location="cpu", weights_only=False)
        row = d[(d.feature_set == fs) & (d.model == m) & (d.case == case) &
                (d.scope == "per_day") & (d.seed == seed)].iloc[0]
        prep, prep_kind = None, "sequence-only (repo signal pipeline; deterministic)"
        try:
            if fs == "clinical_only":
                prep = _clinical_prep(case); prep_kind = "tabular clinical (imputer, scaler, mRMR-40 idx)"
            elif fs == "combined":
                prep = _combined_clin_prep(case); prep_kind = "clinical branch (scaler, imputer, mRMR-40 sel)"
        except Exception as e:
            prep_kind = f"prep-extract-failed: {e}"
        out = f"pathA_{FS2SHORT[fs]}_{m}_{case}.joblib"
        joblib.dump({"kind": "pathA_benchmark", "pipeline": r.pipeline, "model": r.model, "case": case,
                     "feature_set": fs, "encoder": m, "sel_seed": seed,
                     "checkpoint": dict(ckpt), "threshold": float(row["threshold"]),
                     "tabular_prep": prep, "tabular_prep_kind": prep_kind,
                     "best_val_auroc": float(row["best_val_auroc"]),
                     "metrics": {"full_AUROC": float(r.full_AUROC), "full_Precision": float(r.full_Precision),
                                 "full_Recall": float(r.full_Recall)},
                     "source_pt": pt,
                     "loader": ("TabularClinicalDataset" if fs == "clinical_only" else "AggDataset")
                               + " in abalation_study_v2/models/ (rebuilds reduced sequence + tabular prep)"},
                    os.path.join(MODELS_DIR, out))
        rows.append({"model": r.model, "pipeline": r.pipeline, "case": case, "seed": seed,
                     "bundle": out, "prep": prep_kind})
        print(f"[ok] {case} {r.pipeline:13s} {r.model:16s} seed{seed} -> {out}  [{prep_kind}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(MODELS_DIR, "pathA_bundles_index.csv"), index=False)
    print(f"\nsaved 14 Path A bundles -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
