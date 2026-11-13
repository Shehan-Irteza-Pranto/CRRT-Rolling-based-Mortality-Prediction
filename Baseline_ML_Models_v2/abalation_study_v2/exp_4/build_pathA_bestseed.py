"""
build_pathA_bestseed.py — best-seed (by val AUROC) rows for the CNN1D/RNN/Transformer + clinical MLP
====================================================================================================
Path A of Table VI already stores every seed as its own row in results/dl_current_denoised.csv
(per_day scope reproduces the paper's seed-mean table). Best-seed = pick, per (pipeline, model, case),
the seed with the highest VALIDATION AUROC (best_val_auroc) — leak-safe model selection — and report
that seed's held-out test metrics. No retraining. Writes exp_4/results/pathA_bestseed.csv.
"""
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
SRC = os.path.join(_AS, "results", "dl_current_denoised.csv")
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCOPE = "per_day"                                   # matches the reported Table VI

# (pipeline label, feature_set, model) rows that Path A contributes to the table
CELLS = [
    ("Clinical-only", "clinical_only", "mlp"),
    ("Machine-only", "machine_only", "cnn1d"),
    ("Machine-only", "machine_only", "rnn"),
    ("Machine-only", "machine_only", "transformer"),
    ("Combined", "combined", "cnn1d"),
    ("Combined", "combined", "rnn"),
    ("Combined", "combined", "transformer"),
]
MODEL_NAME = {"mlp": "MLP", "cnn1d": "CNN1D", "rnn": "RNN (BiLSTM/GRU)", "transformer": "Transformer"}


def main():
    d = pd.read_csv(SRC)
    d = d[(d.scope == SCOPE)]
    rows = []
    for case in ("1day", "2day"):
        for pipe, fs, model in CELLS:
            s = d[(d.feature_set == fs) & (d.model == model) & (d.case == case)]
            if s.empty:
                print(f"[skip] {fs}/{model}/{case}: no rows"); continue
            best = s.loc[s["best_val_auroc"].idxmax()]                 # argmax VAL AUROC
            seedmean = s["full_test_AUROC"].mean()
            rows.append({
                "pipeline": pipe, "model": MODEL_NAME[model], "case": case,
                "full_AUROC": round(best["full_test_AUROC"], 3),
                "full_AUPRC": round(best["full_test_AUPRC"], 3),
                "full_F1": round(best["full_test_F1"], 3),
                "full_Precision": round(best["full_test_Precision"], 3),
                "full_Recall": round(best["full_test_Recall"], 3),
                "r1to1_AUROC": round(best["r1to1_AUROC_mean"], 3),
                "r1to1_AUPRC": round(best["r1to1_AUPRC_mean"], 3),
                "r1to2_AUROC": round(best["r1to2_AUROC_mean"], 3),
                "r1to2_AUPRC": round(best["r1to2_AUPRC_mean"], 3),
                "sel_seed": int(best["seed"]), "best_val_auroc": round(best["best_val_auroc"], 4),
                "seedmean_AUROC": round(seedmean, 3), "source": "pathA",
            })
            print(f"[ok] {case} {pipe:13s} {MODEL_NAME[model]:16s} seed{int(best['seed'])} "
                  f"val {best['best_val_auroc']:.3f} | test AUROC {best['full_test_AUROC']:.3f} "
                  f"(seed-mean {seedmean:.3f})  P {best['full_test_Precision']:.3f} R {best['full_test_Recall']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pathA_bestseed.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'pathA_bestseed.csv')}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
