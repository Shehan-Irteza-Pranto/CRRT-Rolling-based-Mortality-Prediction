"""
matrix_combined_1day.py — COMBINED pipeline: every encoder (standalone fusion + ensemble w/ 3 classical)
=======================================================================================================
Encoders under the SAME exp_3 recipe (5 seeds, best-val, window_agg 30-ch): CNN1D (mean pooling),
CNN-attention, RNN (BiGRU mean), patch-Transformer. Each reported as the standalone Fusion-DL and as the
ensemble = logistic stack of the fusion + top-3 combined/clinical classical models. 1-day.
Writes exp_4/results/matrix_combined_1day.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ablation_ensemble_branch as ABL                # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
CF = ABL.CF; MR = CF.MR
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)
CASE = "1day"
ENC = [("cnn1d", "CNN1D (mean pool)"), ("attn_cnn", "CNN-attention"),
       ("rnn", "RNN (BiGRU)"), ("transformer", "patch-Transformer")]


def main():
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")], "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)
    rows = []
    for enc, label in ENC:
        print(f"==== combined / {label} ====", flush=True)
        fv, ft, yva, yte = ABL.fusion_probs_encoder(CASE, enc)
        S = fv.shape[0]
        s_f = int(np.argmax([roc_auc_score(yva, fv[s]) for s in range(S)]))
        rows.append(ABL._metrics_row(label, "Fusion-DL", CASE, fv[s_f], ft[s_f], yva, yte, s_f))
        stack = []
        for s in range(S):
            c = {"dl:fusion": (fv[s], ft[s])}
            for m, d in comb.items():
                c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
            for m, d in clin.items():
                c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
            vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
            cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
            names = ["dl:fusion"] + cls3
            Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
            mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
            stack.append((mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]))
        s_s = int(np.argmax([roc_auc_score(yva, v) for v, _ in stack]))
        rows.append(ABL._metrics_row(label, "Ensemble (DL + 3 classical)", CASE, stack[s_s][0], stack[s_s][1], yva, yte, s_s))
        print(f"  {label}: Fusion-DL {rows[-2]['full_AUROC']} | Ensemble {rows[-1]['full_AUROC']}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "matrix_combined_1day.csv"), index=False)
    print(f"saved -> {os.path.join(OUT,'matrix_combined_1day.csv')}")


if __name__ == "__main__":
    main()
