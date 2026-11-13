"""
machine_tuned_transformer_stack.py — MACHINE-ONLY tuned patch-Transformer + top-3 classical ensemble
====================================================================================================
Parallels the combined 0.738 result on the machine-only side. (1) HP-search the machine-only transformer
(encoder-only, no clinical branch) over the same grid, 3-seed seed-averaged, select on VALIDATION AUROC.
(2) Retrain the winner 5 seeds and build the ensemble = logistic stack of the tuned-transformer DL +
top-3 base/rich machine classical (forced-in), best-val seed. Reuses matrix_machine machinery. GPU.
Writes exp_4/results/machine_tuned_transformer_1day.csv.
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
import matrix_machine_1day as MM                      # noqa: E402 (TE, CR, MR, build_seq_encoder, _machine_classical, _sampler)
from hp_search_transformer import CONFIGS, _boot_ci   # noqa: E402 (same 12-config grid)
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
TE = MM.TE
DEV = MM.DEV
OUT = os.path.join(_HERE, "results")
CASE = "1day"
N_SEEDS_HP, EPOCHS, PATIENCE = 3, 120, 15


class MachineTransformer(nn.Module):
    def __init__(self, in_ch, d_model, layers, heads, ff, dropout):
        super().__init__()
        self.seq_enc, dim = MM.build_seq_encoder("transformer", in_ch=in_ch, d_model=d_model,
                                                 n_heads=heads, n_layers=layers, ff=ff, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(dim, MM.CH), nn.ReLU(), nn.Dropout(dropout), nn.Linear(MM.CH, 1))

    def forward(self, x, mask):
        return self.head(self.seq_enc(x, mask)).squeeze(-1)


def _train(cfg, tensors, n_seeds):
    d_model, layers, heads, ff, dropout, wd, lr = cfg
    (Xtr, mtr, ytr), (Xva, mva), (Xte, mte), yva, yte = tensors
    vps, tps = [], []
    for sd in range(n_seeds):
        TE._set_seed(sd)
        model = MachineTransformer(Xtr.shape[-1], d_model, layers, heads, ff, dropout).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, mtr, ytr), batch_size=TE.BATCH, sampler=MM._sampler(ytr))
        best = (-1, None, 0)
        for ep in range(EPOCHS):
            model.train()
            for xb, mb, yb in loader:
                opt.zero_grad(); loss = lossf(model(xb.to(DEV), mb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, TE._infer(model, Xva, mva, DEV))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= PATIENCE:
                break
        model.load_state_dict(best[1])
        vps.append(TE._infer(model, Xva, mva, DEV)); tps.append(TE._infer(model, Xte, mte, DEV))
    return np.array(vps), np.array(tps)


def _row(model, vp, tp, yva, yte, seed=None):
    by = {r["protocol"]: r for r in MM.MR._rows_for_variant(model, CASE, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    return {"variant": model, "sel_seed": seed, "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"),
            "full_F1": g("full", "F1"), "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
            "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
            "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}


def main():
    Xtr, mtr, ytr, _, _ = TE._load("train", CASE)
    Xva, mva, yva_t, _, _ = TE._load("val", CASE)
    Xte, mte, yte_t, _, _ = TE._load("test", CASE)
    yva, yte = yva_t.numpy().astype(int), yte_t.numpy().astype(int)
    tensors = ((Xtr, mtr, ytr), (Xva, mva), (Xte, mte), yva, yte)

    # ---- 1. HP search (val-selected, seed-averaged) ----
    print("== HP search: machine-only transformer ==", flush=True)
    hp_rows, best = [], None
    for i, cfg in enumerate(CONFIGS):
        vps, tps = _train(cfg, tensors, N_SEEDS_HP)
        vmean, tmean = vps.mean(0), tps.mean(0)
        va, te = roc_auc_score(yva, vmean), roc_auc_score(yte, tmean)
        hp_rows.append({"cfg": i, **dict(zip(["d_model","layers","heads","ff","dropout","wd","lr"], cfg)),
                        "val_AUROC_avg": round(va, 4), "test_AUROC_avg": round(te, 4)})
        print(f"  [{i:2d}] dm{cfg[0]} L{cfg[1]} do{cfg[4]} wd{cfg[5]:.0e} lr{cfg[6]:.0e} | val {va:.4f} test-avg {te:.4f}", flush=True)
        if best is None or va > best[1]:
            best = (cfg, va)
    pd.DataFrame(hp_rows).sort_values("val_AUROC_avg", ascending=False).to_csv(
        os.path.join(OUT, "hp_transformer_machine_1day.csv"), index=False)
    win = best[0]
    print(f"  winner (best val): dm{win[0]} L{win[1]} h{win[2]} ff{win[3]} do{win[4]} wd{win[5]:.0e} lr{win[6]:.0e}", flush=True)

    # ---- 2. retrain winner 5 seeds + build ensemble (tuned-tf + top-3 machine classical) ----
    print("== retrain winner (5 seeds) + machine classical + stack ==", flush=True)
    tf_vp, tf_tp = _train(win, tensors, 5)
    cls = MM._machine_classical(CASE)                 # {cls:base/rich:model: ([5,n]v,[5,n]t)}
    rows = []
    s_d = int(np.argmax([roc_auc_score(yva, tf_vp[s]) for s in range(5)]))
    rows.append(_row("tuned-transformer DL (machine)", tf_vp[s_d], tf_tp[s_d], yva, yte, s_d))
    stack = []
    for s in range(5):
        c = {"dl": (tf_vp[s], tf_tp[s])}
        for n, (vp, tp) in cls.items():
            c[n] = (vp[s], tp[s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        names = ["dl"] + cls3
        Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
        mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
        stack.append((mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]))
    s_s = int(np.argmax([roc_auc_score(yva, v) for v, _ in stack]))
    rows.append(_row("Ensemble: tuned-transformer + 3 classical", stack[s_s][0], stack[s_s][1], yva, yte, s_s))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "machine_tuned_transformer_1day.csv"), index=False)
    print("\n=== machine-only tuned transformer ===")
    for r in rows:
        print(f"  {r['variant']:42s} AUROC {r['full_AUROC']}  P {r['full_Precision']} R {r['full_Recall']}")
    print(f"saved -> {os.path.join(OUT,'machine_tuned_transformer_1day.csv')}")


if __name__ == "__main__":
    main()
