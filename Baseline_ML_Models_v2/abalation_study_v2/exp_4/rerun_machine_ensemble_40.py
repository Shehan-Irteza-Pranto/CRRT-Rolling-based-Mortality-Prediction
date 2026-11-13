"""
rerun_machine_ensemble_40.py — machine-only Windowed-Transformer Ensemble (ours) with UPDATED machine
feature count (mRMR 30 -> 40). The tuned transformer DL is pure-sequence (independent of the machine
feature count), so we reuse the already-selected HP winner (from hp_transformer_machine_1day.csv) and
only recompute the classical machine members at mRMR-40, then re-form the forced logistic stack.
Non-destructive: writes results/machine_tuned_transformer_40feat_1day.csv. GPU + CPU.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import matrix_machine_1day as MM                          # noqa: E402
MM.CR.N_MRMR = 40                                         # <-- updated machine mRMR budget (was 30)
import machine_tuned_transformer_stack as MT             # noqa: E402 (reuse _train, _row)
CASE = "1day"
OUT = os.path.join(_HERE, "results", "machine_tuned_transformer_40feat_1day.csv")


def main():
    # winner transformer config (val-selected earlier; independent of the classical feature count)
    hp = pd.read_csv(os.path.join(_HERE, "results", "hp_transformer_machine_1day.csv"))
    w = hp.sort_values("val_AUROC_avg", ascending=False).iloc[0]
    win = (int(w.d_model), int(w.layers), int(w.heads), int(w.ff), float(w.dropout), float(w.wd), float(w.lr))
    print(f"winner transformer: dm{win[0]} L{win[1]} h{win[2]} ff{win[3]} do{win[4]} wd{win[5]:.0e} lr{win[6]:.0e}",
          flush=True)

    Xtr, mtr, ytr, _, _ = MT.TE._load("train", CASE)
    Xva, mva, yva_t, _, _ = MT.TE._load("val", CASE)
    Xte, mte, yte_t, _, _ = MT.TE._load("test", CASE)
    yva, yte = yva_t.numpy().astype(int), yte_t.numpy().astype(int)
    tensors = ((Xtr, mtr, ytr), (Xva, mva), (Xte, mte), yva, yte)

    print("retraining tuned transformer (5 seeds) + machine classical @ mRMR-40 + stack ...", flush=True)
    tf_vp, tf_tp = MT._train(win, tensors, 5)
    cls = MM._machine_classical(CASE)                     # now selects 40 machine features (CR.N_MRMR=40)

    stack, members = [], []
    for s in range(5):
        c = {"dl": (tf_vp[s], tf_tp[s])}
        for n, (vp, tp) in cls.items():
            c[n] = (vp[s], tp[s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        names = ["dl"] + cls3
        Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
        mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
        stack.append((mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1])); members.append(names)
    s_s = int(np.argmax([roc_auc_score(yva, v) for v, _ in stack]))
    row = MT._row("Windowed-Transformer Ensemble (ours, machine=40)", stack[s_s][0], stack[s_s][1], yva, yte, s_s)
    pd.DataFrame([row]).to_csv(OUT, index=False)
    # save per-day probabilities of the selected (best-val-seed) stack, keyed by PID/Visit_no/Day,
    # for the patient-clustered bootstrap (bootstrap_ci.py). Row order = machine_only feature table.
    _, va_tab, te_tab, _ = MM.CR._load("base", CASE)
    for split, tab, y, pr in (("val", va_tab, yva, stack[s_s][0]), ("test", te_tab, yte, stack[s_s][1])):
        assert len(tab) == len(y) and (tab["label"].to_numpy(int) == y).all(), f"{split}: row order mismatch"
        d = tab[["PID", "Visit_no", "Day"]].copy(); d["label"] = y; d["prob"] = pr
        d.to_csv(os.path.join(_HERE, "results", f"machine_ensemble_40feat_{split}_probs_{CASE}.csv"), index=False)
    print("  members:", members[s_s], flush=True)
    print("\n=== machine-only Windowed-Transformer Ensemble (ours) @ machine mRMR-40 ===")
    print(f"  Full  AUROC {row['full_AUROC']}  AUPRC {row['full_AUPRC']}  F1 {row['full_F1']}  "
          f"P {row['full_Precision']}  R {row['full_Recall']}")
    print(f"  1:1   AUROC {row['r1to1_AUROC']}  AUPRC {row['r1to1_AUPRC']}")
    print(f"  1:2   AUROC {row['r1to2_AUROC']}  AUPRC {row['r1to2_AUPRC']}")
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
