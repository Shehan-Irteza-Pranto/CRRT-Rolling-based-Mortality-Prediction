"""
regenerate_stack_test_probs.py — dump per-test-circuit-day probabilities of the reported combined
ensemble (tuned transformer fusion + top-3 classical, forced logistic stack, 1-day, AUROC 0.738).

That model's weights were never saved (stack_with_tuned_transformer.py only wrote a metrics CSV), so we
re-run its exact training (deterministic 5-seed) + forced-stack + best-val-seed selection and persist the
test probabilities aligned to (PID, Visit_no, Day, label). Needed for the sub-population analysis
(Tables X/XI). GPU. Writes results/combined_stack_test_probs_1day.csv.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stack_with_tuned_transformer as ST                 # noqa: E402
import ablation_ensemble_branch as ABL                    # noqa: E402
CF = ST.CF
CASE = "1day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_machine", type=int, default=20)
    ap.add_argument("--n_clinical", type=int, default=40)
    a = ap.parse_args()
    CF.N_MACHINE, CF.N_CLINICAL = a.n_machine, a.n_clinical          # grouped-mRMR counts (best-features config)
    suffix = "" if (a.n_machine, a.n_clinical) == (20, 40) else f"_{a.n_machine}x{a.n_clinical}"
    OUT = os.path.join(_HERE, "results", f"combined_stack_test_probs{suffix}_1day.csv")
    data, clin_t = ABL._fusion_data(CASE)
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
    keys = data["test"][3].reset_index(drop=True)          # PID, Visit_no, Day, label
    print(f"test circuit-days={len(keys)} pos={int(yte.sum())}", flush=True)

    print("training tuned transformer fusion (5 seeds)...", flush=True)
    tf_vp, tf_tp = ST._train_tf(data, clin_t)
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")],
                               "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)

    forced = []
    for s in range(ST.N_SEEDS):
        c = {"dl:tf_fusion": (tf_vp[s], tf_tp[s])}
        for m, d in comb.items():
            c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
        for m, d in clin.items():
            c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        forced.append(ST._stack(c, ["dl:tf_fusion"] + cls3, yva))
    sf = int(np.argmax([roc_auc_score(yva, v) for v, _ in forced]))
    p_val, p_test = forced[sf][0], forced[sf][1]
    au = roc_auc_score(yte, p_test)
    print(f"forced stack best-val seed={sf}  test AUROC={au:.4f}  "
          f"({CF.N_MACHINE} machine + {CF.N_CLINICAL} clinical)", flush=True)

    out = keys.copy()
    out["prob"] = p_test
    out.to_csv(OUT, index=False)
    # validation probs + labels (for the max-F1-on-val operating threshold)
    valdf = pd.DataFrame({"prob": p_val, "label": yva})
    valpath = os.path.join(_HERE, "results", f"combined_stack_val_probs{suffix}_1day.csv")
    valdf.to_csv(valpath, index=False)
    print(f"saved -> {OUT}  ({out.shape})\nsaved -> {valpath}  ({valdf.shape})", flush=True)


if __name__ == "__main__":
    main()
