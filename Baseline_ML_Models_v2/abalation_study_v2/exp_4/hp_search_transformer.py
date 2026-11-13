"""
hp_search_transformer.py — val-selected HP search for the COMBINED transformer fusion (generalization)
=====================================================================================================
Search transformer arch + regularization (d_model, depth, heads, ff, dropout, weight_decay, lr) for the
combined fusion (patch/window transformer machine branch + mRMR-40 clinical branch). Each config trained
with 3 seeds and SEED-AVERAGED; the winner is picked by **validation** AUROC (leak-safe, favors
generalizing configs), and its **full-test** AUROC is reported with a patient-bootstrap 95% CI. Reuses the
ablation's data + fusion machinery. GPU.  Writes exp_4/results/hp_transformer_1day.csv.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ablation_ensemble_branch as ABL                # noqa: E402 (reuse _fusion_data/_sampler/_infer/CF/build_seq_encoder)
CF = ABL.CF
from fusion import build_seq_encoder                  # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)
N_SEEDS = 3
EPOCHS, PATIENCE = 120, 15

# (d_model, layers, heads, ff, dropout, weight_decay, lr) — curated, favouring small + regularized
CONFIGS = [
    (64, 2, 4, 128, 0.30, 1e-3, 3e-4),   # exp_4 baseline (ref, test 0.724 best-seed)
    (32, 1, 2, 64,  0.30, 1e-3, 3e-4),   # tiny
    (32, 2, 2, 64,  0.40, 5e-3, 3e-4),   # tiny + reg
    (48, 1, 4, 96,  0.40, 5e-3, 3e-4),
    (48, 2, 4, 96,  0.50, 5e-3, 1e-4),
    (64, 1, 4, 128, 0.40, 5e-3, 3e-4),
    (64, 2, 4, 128, 0.50, 1e-2, 3e-4),   # strong dropout + wd
    (64, 2, 8, 128, 0.40, 5e-3, 1e-4),
    (32, 2, 4, 64,  0.50, 1e-2, 3e-4),
    (48, 2, 2, 128, 0.40, 1e-2, 3e-4),
    (96, 2, 4, 128, 0.50, 1e-2, 1e-4),   # larger but heavily regularized
    (64, 1, 2, 64,  0.50, 5e-3, 3e-4),
]


class TFusion(nn.Module):
    def __init__(self, in_ch, clin_dim, d_model, layers, heads, ff, dropout):
        super().__init__()
        self.seq_enc, seq_dim = build_seq_encoder(
            "transformer", in_ch=in_ch, d_model=d_model, n_heads=heads, n_layers=layers, ff=ff, dropout=dropout)
        self.clin = nn.Sequential(nn.Linear(clin_dim, CF.CH), nn.BatchNorm1d(CF.CH), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(seq_dim + CF.CH, CF.CH), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(CF.CH, 1))

    def forward(self, x, mask, clin):
        return self.head(torch.cat([self.seq_enc(x, mask), self.clin(clin)], 1)).squeeze(-1)


def _boot_ci(y, s, pid, n=1000, seed=0):
    pid = np.asarray(pid); pats = np.unique(pid); rng = np.random.default_rng(seed)
    by = {p: np.where(pid == p)[0] for p in pats}; aucs = []
    for _ in range(n):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)


def train_config(cfg, tensors):
    d_model, layers, heads, ff, dropout, wd, lr = cfg
    (Xtr, mtr, ctr, ytr), (Xva, mva, cva), (Xte, mte, cte), yva, yte = tensors
    vps, tps, seed_te = [], [], []
    for sd in range(N_SEEDS):
        torch.manual_seed(sd); np.random.seed(sd)
        model = TFusion(Xtr.shape[-1], ctr.shape[-1], d_model, layers, heads, ff, dropout).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
        seed_te.append(roc_auc_score(yte, tps[-1]))
    vmean = np.mean(vps, 0); tmean = np.mean(tps, 0)
    return {"val_AUROC_avg": round(roc_auc_score(yva, vmean), 4),
            "test_AUROC_avg": round(roc_auc_score(yte, tmean), 4),
            "test_AUROC_seedmax": round(max(seed_te), 4), "test_prob_avg": tmean}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day"); a = ap.parse_args()
    case = a.case
    data, clin = ABL._fusion_data(case)
    tensors = ((data["train"][0], data["train"][1], clin["train"], data["train"][2]),
               (data["val"][0], data["val"][1], clin["val"]),
               (data["test"][0], data["test"][1], clin["test"]),
               data["val"][4].astype(int), data["test"][4].astype(int))
    yte = tensors[4]; pid_te = data["test"][3]["PID"].to_numpy()
    rows = []
    for i, cfg in enumerate(CONFIGS):
        r = train_config(cfg, tensors)
        lo, hi = _boot_ci(yte, r.pop("test_prob_avg"), pid_te)
        row = {"cfg": i, "d_model": cfg[0], "layers": cfg[1], "heads": cfg[2], "ff": cfg[3],
               "dropout": cfg[4], "weight_decay": cfg[5], "lr": cfg[6],
               "val_AUROC_avg": r["val_AUROC_avg"], "test_AUROC_avg": r["test_AUROC_avg"],
               "test_AUROC_seedmax": r["test_AUROC_seedmax"], "test_CI95": f"[{lo}, {hi}]"}
        rows.append(row)
        pd.DataFrame(rows).sort_values("val_AUROC_avg", ascending=False).to_csv(
            os.path.join(OUT, f"hp_transformer_{case}.csv"), index=False)
        print(f"[{i:2d}] dm{cfg[0]} L{cfg[1]} h{cfg[2]} ff{cfg[3]} do{cfg[4]} wd{cfg[5]:.0e} lr{cfg[6]:.0e} "
              f"| val {row['val_AUROC_avg']} test-avg {row['test_AUROC_avg']} "
              f"(seedmax {row['test_AUROC_seedmax']}) CI {row['test_CI95']}", flush=True)
    df = pd.DataFrame(rows).sort_values("val_AUROC_avg", ascending=False)
    best = df.iloc[0]
    print("\n=== best by VALIDATION AUROC (seed-averaged) ===")
    print(f"  cfg d_model={best.d_model} layers={best.layers} heads={best.heads} ff={best.ff} "
          f"dropout={best.dropout} wd={best.weight_decay} lr={best.lr}")
    print(f"  val {best.val_AUROC_avg} | FULL-TEST AUROC {best.test_AUROC_avg}  CI {best.test_CI95} "
          f"(baseline exp_4 transformer best-seed 0.724)")
    print(f"\nsaved -> {os.path.join(OUT, f'hp_transformer_{case}.csv')}")


if __name__ == "__main__":
    main()
