"""tableV_rows1_4_40x60.py — Table V top block (incremental preprocessing on the Windowed-Transformer
combined stack) re-run with the 40/60 feature list. Wraps run_preproc_ablation with the classical
counts patched. Writes ../exp_4/results/tableV_rows1_4_40x60.csv."""
import os
import sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_preproc_ablation as RPA                         # noqa: E402
RPA.CF.N_MACHINE, RPA.CF.N_CLINICAL = 40, 60               # classical base learners at best features
OUT = os.path.join(_HERE, "..", "exp_4", "results", "tableV_rows1_4_40x60.csv")


def main():
    rows = []
    for name, tag, enriched in RPA.CONFIGS:
        rows.append(RPA.run_config(name, tag, enriched))
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    print("\n=== Table V top block (incremental preprocessing, 40/60) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
