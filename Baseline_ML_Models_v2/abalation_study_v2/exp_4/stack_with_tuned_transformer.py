"""
stack_with_tuned_transformer.py — does the HP-tuned transformer fusion earn a Stack slot & beat 0.740?
=====================================================================================================
Adds the tuned transformer fusion (winner config) to the combined Stack candidate pool alongside the
combined-classical + clinical-classical models, re-runs the top-4-by-validation selection per seed,
best-val-seed select, and reports: (a) whether the transformer enters the top-4, (b) the resulting Stack
AUROC / metrics vs the current 0.740, and (c) a 'forced-in' stack (tf + top-3 classical). GPU.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ablation_ensemble_branch as ABL                # noqa: E402
import hp_search_transformer as HPT                   # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
CF = ABL.CF; MR = CF.MR
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIN = dict(d_model=64, layers=1, heads=4, ff=128, dropout=0.4, wd=5e-3, lr=3e-4)
N_SEEDS, EPOCHS, PATIENCE, CASE = 5, 120, 15, "1day"
OUT = os.path.join(_HERE, "results")


def _train_tf(data, clin):
    Xtr, mtr, ctr, ytr = data["train"][0], data["train"][1], clin["train"], data["train"][2]
    Xva, mva, cva = data["val"][0], data["val"][1], clin["val"]
    Xte, mte, cte = data["test"][0], data["test"][1], clin["test"]
    yva = data["val"][4].astype(int)
    vps, tps = [], []
    for sd in range(N_SEEDS):
        torch.manual_seed(sd); np.random.seed(sd)
        model = HPT.TFusion(Xtr.shape[-1], ctr.shape[-1], WIN["d_model"], WIN["layers"], WIN["heads"],
                            WIN["ff"], WIN["dropout"]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=WIN["lr"], weight_decay=WIN["wd"])
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, mtr, ctr, ytr), batch_size=CF.BATCH, sampler=ABL._sampler(ytr))
        best = (-1, None, 0)
        for ep in range(EPOCHS):
            model.train()
            for xb, mb, cb, yb in loader:
                opt.zero_grad(); loss = lossf(model(xb.to(DEV), mb.to(DEV), cb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, ABL._infer(model, Xva, mva, cva))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= PATIENCE:
                break
        model.load_state_dict(best[1])
        vps.append(ABL._infer(model, Xva, mva, cva)); tps.append(ABL._infer(model, Xte, mte, cte))
        print(f"  tf seed {sd}: test {roc_auc_score(data['test'][4].astype(int), tps[-1]):.4f}", flush=True)
    return np.array(vps), np.array(tps)


def _stack(c, names, yva):
    Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
    mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
    return mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]


def _row(name, case, vp, tp, yva, yte, extra=None):
    by = {r["protocol"]: r for r in MR._rows_for_variant(name, case, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    d = {"variant": name, "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"),
         "full_F1": g("full", "F1"), "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
         "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
         "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}
    if extra:
        d.update(extra)
    return d


def main():
    data, clin_t = ABL._fusion_data(CASE)
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
    print("training tuned transformer fusion (5 seeds)...", flush=True)
    tf_vp, tf_tp = _train_tf(data, clin_t)
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")], "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)

    top4_seeds, forced_seeds, in_top4, top4_names = [], [], [], []
    for s in range(N_SEEDS):
        c = {"dl:tf_fusion": (tf_vp[s], tf_tp[s])}
        for m, d in comb.items():
            c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
        for m, d in clin.items():
            c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        t4 = sorted(vauc, key=vauc.get, reverse=True)[:4]
        top4_seeds.append(_stack(c, t4, yva)); in_top4.append("dl:tf_fusion" in t4); top4_names.append(t4)
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        forced_seeds.append(_stack(c, ["dl:tf_fusion"] + cls3, yva))
    s4 = int(np.argmax([roc_auc_score(yva, v) for v, _ in top4_seeds]))
    sf = int(np.argmax([roc_auc_score(yva, v) for v, _ in forced_seeds]))
    sd_tf = int(np.argmax([roc_auc_score(yva, tf_vp[s]) for s in range(N_SEEDS)]))

    rows = [
        _row("tuned-transformer fusion (best-val seed)", CASE, tf_vp[sd_tf], tf_tp[sd_tf], yva, yte),
        _row("Stack top-4 (tf as candidate)", CASE, top4_seeds[s4][0], top4_seeds[s4][1], yva, yte,
             {"tf_in_top4": bool(in_top4[s4]), "members": "+".join(n.split(":")[-1] for n in top4_names[s4])}),
        _row("Stack tf+top-3-classical (forced)", CASE, forced_seeds[sf][0], forced_seeds[sf][1], yva, yte),
    ]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "stack_tuned_transformer_1day.csv"), index=False)
    print("\n================ tuned transformer in the Stack (1-day) ================")
    for r in rows:
        extra = f"  [tf_in_top4={r.get('tf_in_top4')}, members={r.get('members')}]" if "tf_in_top4" in r else ""
        print(f"  {r['variant']:44s} AUROC {r['full_AUROC']}  AUPRC {r['full_AUPRC']}  "
              f"P {r['full_Precision']} R {r['full_Recall']}{extra}")
    print(f"\n  (reference: current best Stack top-4 = 0.740; it excluded the DL fusion)")
    print(f"saved -> {os.path.join(OUT, 'stack_tuned_transformer_1day.csv')}")


if __name__ == "__main__":
    main()
