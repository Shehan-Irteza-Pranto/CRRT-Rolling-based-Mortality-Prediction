"""
rerun_combined_bestseed.py — best-seed (by val AUROC) for the COMBINED 'ours' DL rows
=====================================================================================
Re-runs the combined CNN-fusion + logistic Stack PER SEED (exp_3 averages them) and selects, per model,
the seed with the highest VALIDATION AUROC — then reports that seed's full / 1:1 / 1:2 metrics using the
max-F1-on-val threshold. Produces the two 'ours' combined rows:
  * Attn-CNN fusion  = the CNN-fusion DL (attention-CNN machine branch + mRMR-40 clinical MLP branch)
  * Ensemble/Stack   = logistic meta-learner over the top-4-by-val components (STACK:top4)
Reuses exp_3/results/combined_fusion_ensemble.py (imported as CF) via its per_seed hooks. GPU.
Writes exp_4/results/pathB_combined_bestseed.csv (+ per-seed probs).
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                              # noqa: E402 (sets up its own sys.path)
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402
MR = CF.MR
OUT = os.path.join(_HERE, "results"); os.makedirs(os.path.join(OUT, "probs"), exist_ok=True)
MET = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]


def _seed_cands(s, fv, ft, comb, clin):
    cands = {"dl:cnn_fusion": (fv[s], ft[s])}
    for m, (vp, tp) in comb.items():
        cands[f"cls:combined:{m}"] = (vp[s], tp[s])
    for m, (vp, tp) in clin.items():
        cands[f"cls:clinical:{m}"] = (vp[s], tp[s])
    return cands


def _stack_top4(cands, yva):
    names = list(cands)
    vauc = {n: roc_auc_score(yva, cands[n][0]) for n in names}
    top = sorted(vauc, key=vauc.get, reverse=True)[:4]
    Lv = np.column_stack([cands[n][0] for n in top]); Lt = np.column_stack([cands[n][1] for n in top])
    meta = LogisticRegression(max_iter=5000).fit(Lv, yva)
    return meta.predict_proba(Lv)[:, 1], meta.predict_proba(Lt)[:, 1]


def _pick_and_score(label, pipe, case, seed_probs, yva, yte):
    """seed_probs: list of (vprob, tprob). Pick argmax val AUROC; return the table row."""
    vaucs = [roc_auc_score(yva, vp) for vp, _ in seed_probs]
    s = int(np.argmax(vaucs))
    vp, tp = seed_probs[s]
    rows = MR._rows_for_variant(label, case, vp, tp, yva, yte)      # full / 1:1 / 1:2 @ max-F1-val thr
    by = {r["protocol"]: r for r in rows}
    out = {"pipeline": pipe, "model": label, "case": case, "sel_seed": s,
           "best_val_auroc": round(float(vaucs[s]), 4), "source": "pathB",
           "seedmean_AUROC": round(float(np.mean([roc_auc_score(yte, tp2) for _, tp2 in seed_probs])), 3)}
    out["full_AUROC"] = round(by["full"]["AUROC"], 3); out["full_AUPRC"] = round(by["full"]["AUPRC"], 3)
    out["full_F1"] = round(by["full"]["F1"], 3); out["full_Precision"] = round(by["full"]["Precision"], 3)
    out["full_Recall"] = round(by["full"]["Recall"], 3)
    out["r1to1_AUROC"] = round(by["1:1"]["AUROC"], 3); out["r1to1_AUPRC"] = round(by["1:1"]["AUPRC"], 3)
    out["r1to2_AUROC"] = round(by["1:2"]["AUROC"], 3); out["r1to2_AUPRC"] = round(by["1:2"]["AUPRC"], 3)
    print(f"[ok] {case} {label:18s} best-val seed {s} (val {vaucs[s]:.3f}) | "
          f"full AUROC {out['full_AUROC']} P {out['full_Precision']} R {out['full_Recall']} "
          f"(seed-mean test AUROC {out['seedmean_AUROC']})", flush=True)
    return out


def main():
    rows = []
    for case in ("1day", "2day"):
        print(f"==== combined {case} ====", flush=True)
        fv, ft, yva, yte, pid_te = CF._fusion_probs(case, per_seed=True)          # [S,n]
        comb = CF._classical_probs([CF._combined_table(case, s) for s in ("train", "val", "test")],
                                   "combined", per_seed=True)
        clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, case, f"{s}.csv"))
                                    for s in ("train", "val", "test")], "clinical_only", per_seed=True)
        S = fv.shape[0]
        np.savez(os.path.join(OUT, "probs", f"combined_fusion_perseed_{case}.npz"),
                 fv=fv, ft=ft, yva=yva, yte=yte)
        fusion_seeds = [(fv[s], ft[s]) for s in range(S)]
        stack_seeds = [_stack_top4(_seed_cands(s, fv, ft, comb, clin), yva) for s in range(S)]
        rows.append(_pick_and_score("Attn-CNN fusion", "Combined", case, fusion_seeds, yva, yte))
        rows.append(_pick_and_score("Ensemble/Stack", "Combined", case, stack_seeds, yva, yte))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "pathB_combined_bestseed.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'pathB_combined_bestseed.csv')}")


if __name__ == "__main__":
    main()
