"""
sweep_mrmr_stack.py — grouped-mRMR feature-count sweep for the Windowed-Transformer
Fusion Stack (combined 0.738 model), scored on VALIDATION AUROC only (test untouched).

Sweeps n_machine in {10,20,30,40} x n_clinical in {20,40,60} — the grouped-mRMR counts
that build the classical stack members' tabular vector (panel C). The tuned transformer
fusion DL (sequence + mRMR-40 clinical branch) and the clinical-only classical candidates
are held fixed, so the DL is trained ONCE (5 seeds) and reused across all 12 cells; only
the combined-classical members + the forced logistic stack are recomputed per cell.

Selection metric = validation AUROC of the forced stack (dl:tf_fusion + top-3 classical),
best-of-5-seeds by validation AUROC — exactly how the reported model is chosen. Test AUROC
is recorded for reference only (never used to pick anything). Writes results/mrmr_sweep_1day.csv.
GPU + CPU (classical tuning).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stack_with_tuned_transformer as ST                 # noqa: E402 (imports ABL, CF, WIN, _train_tf, _stack)
import ablation_ensemble_branch as ABL                    # noqa: E402
CF = ST.CF
CASE = "1day"
N_MACH = [10, 20, 30, 40, 50, 60]
N_CLIN = [20, 40, 60, 80, 100]
SPLITS = ("train", "val", "test")
OUT = os.path.join(_HERE, "results", "mrmr_sweep_1day.csv")


def _forced_stack_bestval(tf_vp, tf_tp, comb, clin, yva, yte):
    """Per seed: stack dl:tf_fusion + top-3 classical (by val AUROC among all classical
    candidates); pick the seed with the best VALIDATION stack AUROC."""
    best = {"val": -1.0, "test": np.nan, "seed": -1, "members": None}
    for s in range(ST.N_SEEDS):
        c = {"dl:tf_fusion": (tf_vp[s], tf_tp[s])}
        for m, d in comb.items():
            c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
        for m, d in clin.items():
            c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        names = ["dl:tf_fusion"] + cls3
        sv, st_ = ST._stack(c, names, yva)
        v = roc_auc_score(yva, sv)
        if v > best["val"]:
            best = {"val": v, "test": roc_auc_score(yte, st_), "seed": s,
                    "members": "+".join(n.split(":")[-1] for n in cls3)}
    return best


def main():
    print("building fusion data ...", flush=True)
    data, clin_t = ABL._fusion_data(CASE)
    yva = data["val"][4].astype(int); yte = data["test"][4].astype(int)

    print("training tuned transformer fusion ONCE (5 seeds, reused across all cells) ...", flush=True)
    tf_vp, tf_tp = ST._train_tf(data, clin_t)

    print("clinical-only classical candidates (fixed) ...", flush=True)
    clin_tables = [pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv")) for s in SPLITS]
    clin = CF._classical_probs(clin_tables, "clinical_only", per_seed=True)
    comb_tables = [CF._combined_table(CASE, s) for s in SPLITS]

    rows = []
    for nm in N_MACH:
        for nc in N_CLIN:
            CF.N_MACHINE, CF.N_CLINICAL = nm, nc          # override grouped-mRMR counts
            comb = CF._classical_probs(comb_tables, "combined", per_seed=True)
            b = _forced_stack_bestval(tf_vp, tf_tp, comb, clin, yva, yte)
            rows.append({"n_machine": nm, "n_clinical": nc, "n_total": nm + nc,
                         "val_AUROC": round(b["val"], 4), "test_AUROC_ref": round(b["test"], 4),
                         "sel_seed": b["seed"], "members": b["members"]})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  n_machine={nm:2d} n_clinical={nc:2d} -> VAL AUROC {b['val']:.4f} "
                  f"(test-ref {b['test']:.4f}, seed {b['seed']}, {b['members']})", flush=True)
    print(f"\nsaved -> {OUT}")
    df = pd.DataFrame(rows)
    best = df.loc[df.val_AUROC.idxmax()]
    print(f"best by VALIDATION AUROC: n_machine={int(best.n_machine)} n_clinical={int(best.n_clinical)} "
          f"val {best.val_AUROC} (current 20x40 val {float(df[(df.n_machine==20)&(df.n_clinical==40)].val_AUROC.iloc[0])})")


if __name__ == "__main__":
    main()
