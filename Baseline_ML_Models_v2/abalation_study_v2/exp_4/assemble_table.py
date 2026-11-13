"""
assemble_table.py — merge Path A (best-seed row-selection) + Path B (per-seed re-runs) into Table VI
====================================================================================================
Combines exp_4/results/{pathA_bestseed.csv, pathB_machine_bestseed.csv, pathB_combined_bestseed.csv}
into a single best-seed Table VI (values = best-of-5-seeds by VALIDATION AUROC; threshold = max-F1 on
validation; clinical features = mRMR top-40). Writes table_vi_bestseed.csv and TABLE_VI_bestseed.md in
the attached layout: Full (AUROC/AUPRC/F1/Prec/Rec) + Balanced 1:1 (AUROC/AUPRC) + 1:2 (AUROC/AUPRC),
grouped by horizon -> pipeline.
"""
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "results")

# row order within each horizon block
ORDER = [
    ("Clinical-only", ["MLP"]),
    ("Machine-only", ["CNN1D", "RNN (BiLSTM/GRU)", "Transformer", "Attention-CNN1D", "Ensemble"]),
    ("Combined", ["CNN1D", "RNN (BiLSTM/GRU)", "Transformer", "Attn-CNN fusion", "Ensemble/Stack"]),
]
FULL = ["full_AUROC", "full_AUPRC", "full_F1", "full_Precision", "full_Recall"]
BAL = ["r1to1_AUROC", "r1to1_AUPRC", "r1to2_AUROC", "r1to2_AUPRC"]


def _load():
    parts = []
    for f in ("pathA_bestseed.csv", "pathB_machine_bestseed.csv", "pathB_combined_bestseed.csv"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
        else:
            print(f"[warn] missing {f}")
    return pd.concat(parts, ignore_index=True)


def _fmt(v):
    return f"{v:.3f}" if pd.notna(v) else "--"


def main():
    df = _load()
    df.to_csv(os.path.join(OUT, "table_vi_bestseed.csv"), index=False)

    lines = []
    lines.append("# Table VI (best-seed) — DL benchmark, best-of-5-seeds by validation AUROC\n")
    lines.append("Values = the seed with the highest **validation** AUROC per cell (leak-safe model "
                 "selection); reported on the held-out test set. Threshold for F1/Precision/Recall = "
                 "**max-F1 on validation**. Clinical-only & combined use **mRMR top-40** clinical "
                 "features. Balanced 1:1 / 1:2 = mean over 100 resamples (positives kept, negatives "
                 "subsampled). Compare to the seed-averaged Table VI in the paper.\n")
    header = ("| Model | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | "
              "1:2 AUROC | 1:2 AUPRC |")
    sep = "|" + "---|" * 10
    for case, ctitle in (("1day", "One-day horizon"), ("2day", "Two-day horizon")):
        lines.append(f"\n### {ctitle}\n")
        # bold the best full-AUROC in this horizon
        cd = df[df.case == case]
        best_auroc = cd["full_AUROC"].max()
        lines.append(header); lines.append(sep)
        for pipe, models in ORDER:
            lines.append(f"| *{pipe}* | | | | | | | | | |")
            for m in models:
                r = cd[(cd.pipeline == pipe) & (cd.model == m)]
                if r.empty:
                    lines.append(f"| {m} | -- | -- | -- | -- | -- | -- | -- | -- | -- |"); continue
                r = r.iloc[0]
                a = f"**{_fmt(r.full_AUROC)}**" if r.full_AUROC == best_auroc else _fmt(r.full_AUROC)
                lines.append("| " + " | ".join([m, a, _fmt(r.full_AUPRC), _fmt(r.full_F1),
                             _fmt(r.full_Precision), _fmt(r.full_Recall), _fmt(r.r1to1_AUROC),
                             _fmt(r.r1to1_AUPRC), _fmt(r.r1to2_AUROC), _fmt(r.r1to2_AUPRC)]) + " |")
    lines.append("\n*Source: `exp_4/results/table_vi_bestseed.csv`. Path A (MLP/CNN1D/RNN/Transformer) "
                 "= best-val-seed rows from `results/dl_current_denoised.csv`; Path B (Attention-CNN1D, "
                 "Ensemble, Attn-CNN fusion, Ensemble/Stack) = per-seed re-runs "
                 "(`rerun_machine_bestseed.py`, `rerun_combined_bestseed.py`).*\n")
    md = "\n".join(lines)
    with open(os.path.join(_HERE, "TABLE_VI_bestseed.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\nsaved -> {os.path.join(OUT, 'table_vi_bestseed.csv')} + TABLE_VI_bestseed.md")


if __name__ == "__main__":
    main()
