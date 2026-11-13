"""
train_eval.py — train the attention CNN1D (5 seeds) + day-level test eval (exp_3 Phase 2 gate G2)
================================================================================================
Per-day samples (one circuit-day = one sample). Balanced minibatches (WeightedRandomSampler), BCE,
AdamW, early-stop on val AUROC. Train N_SEEDS models, AVERAGE their val/test probabilities
(ensemble), report day-level test AUROC + patient bootstrap CI + full panel. Saves the averaged
val/test probabilities to results/dl_probs_<case>.npz for the Phase-3 ensemble. Saves a train/val
loss+AUROC curve for the first seed.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dlcfg import (CASE, DATA_DIR, RESULTS_DIR, LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS,   # noqa: E402
                   EARLY_STOP_PATIENCE, N_SEEDS, GPUS, BASELINE_1DAY_CNN, N_BOOTSTRAP)
from model import build_model                                                  # noqa: E402
from dl_data import load_reduced                                               # noqa: E402

import torch                                                                   # noqa: E402
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler  # noqa: E402
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score, precision_recall_curve)


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _load(split, case):
    Xr, m, y, pid, day = load_reduced(split, case)
    return (torch.from_numpy(Xr).float(), torch.from_numpy(m).bool(),
            torch.from_numpy(y).float(), pid, day)


@torch.no_grad()
def _infer(model, X, m, device, bs=256):
    model.eval(); ps = []
    for i in range(0, len(X), bs):
        ps.append(torch.sigmoid(model(X[i:i+bs].to(device), m[i:i+bs].to(device))).cpu().numpy())
    return np.concatenate(ps)


def _auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.0


def train_one(case, seed, device, curve=False):
    _set_seed(seed)
    Xtr, mtr, ytr, *_ = _load("train", case)
    Xva, mva, yva, pid_va, _ = _load("val", case)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.BCEWithLogitsLoss()

    yv = ytr.numpy(); npos = yv.sum(); nneg = len(yv) - npos
    w = np.where(yv == 1, 1.0 / max(npos, 1), 1.0 / max(nneg, 1))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(yv), replacement=True)
    loader = DataLoader(TensorDataset(Xtr, mtr, ytr), batch_size=BATCH, sampler=sampler)

    history, best = [], (-1.0, None, 0)
    yva_np = yva.numpy()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); ls, n = 0.0, 0
        for xb, mb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(device), mb.to(device)), yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ls += float(loss.item()) * len(yb); n += len(yb)
        va = _auroc(yva_np, _infer(model, Xva, mva, device))
        if curve:
            tr_au = _auroc(ytr.numpy(), _infer(model, Xtr, mtr, device))
            history.append({"epoch": epoch, "train_loss": ls / max(n, 1), "train_auroc": tr_au,
                            "val_auroc": va})
        if va > best[0]:
            best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
        else:
            best = (best[0], best[1], best[2] + 1)
        if best[2] >= EARLY_STOP_PATIENCE:
            break
    model.load_state_dict(best[1])
    if curve:
        _curve(history, case)
    return model, best[0]


def _curve(history, case):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    a1.plot(ep, [h["train_loss"] for h in history], "C0-o", ms=3, label="train loss")
    a1.set_ylabel("loss"); a1.legend(fontsize=8); a1.set_title(f"exp3 attn-CNN1D {case} (seed0)", fontsize=9)
    a2.plot(ep, [h["train_auroc"] for h in history], "C0-o", ms=3, label="train AUROC")
    a2.plot(ep, [h["val_auroc"] for h in history], "C1-s", ms=3, label="val AUROC")
    a2.set_ylabel("AUROC"); a2.set_xlabel("epoch"); a2.legend(fontsize=8)
    cd = os.path.join(RESULTS_DIR, "curves"); os.makedirs(cd, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(cd, f"attn_cnn1d_{case}.png"), dpi=110); plt.close(fig)


def _opt_threshold(y, s):
    if np.sum(y) < 2:
        return 0.5
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))])


def _panel(y, s, thr):
    pred = (s >= thr).astype(int)
    return {"AUROC": round(roc_auc_score(y, s), 4), "AUPRC": round(average_precision_score(y, s), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "F2": round(fbeta_score(y, pred, beta=2, zero_division=0), 4),
            "F3": round(fbeta_score(y, pred, beta=3, zero_division=0), 4),
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(recall_score(y, pred, zero_division=0), 4)}


def _boot_ci(y, s, pid, n=N_BOOTSTRAP, seed=0):
    pid = np.asarray(pid); pats = np.unique(pid); rng = np.random.default_rng(seed)
    by = {p: np.where(pid == p)[0] for p in pats}; aucs = []
    for _ in range(n):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return (round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)) if aucs else (np.nan, np.nan)


def run(case=CASE):
    device = torch.device(f"cuda:{GPUS[0]}" if torch.cuda.is_available() else "cpu")
    Xva, mva, yva, pid_va, _ = _load("val", case)
    Xte, mte, yte, pid_te, _ = _load("test", case)
    yva_np, yte_np = yva.numpy().astype(int), yte.numpy().astype(int)

    val_probs, test_probs, seed_aucs = [], [], []
    for sd in range(N_SEEDS):
        model, vbest = train_one(case, sd, device, curve=(sd == 0))
        vp = _infer(model, Xva, mva, device); tp = _infer(model, Xte, mte, device)
        val_probs.append(vp); test_probs.append(tp)
        seed_aucs.append(_auroc(yte_np, tp))
        print(f"[attn-cnn1d] seed {sd}: val_best={vbest:.4f} test={seed_aucs[-1]:.4f}", flush=True)

    vmean = np.mean(val_probs, 0); tmean = np.mean(test_probs, 0)
    thr = _opt_threshold(yva_np, vmean)
    lo, hi = _boot_ci(yte_np, tmean, pid_te)
    row = {"model": "attn_cnn1d_ens", "case": case, "n_seeds": N_SEEDS,
           "test_AUROC_mean_seed": round(float(np.mean(seed_aucs)), 4),
           "test_AUROC_sd_seed": round(float(np.std(seed_aucs)), 4),
           "test_AUROC_ens": round(_auroc(yte_np, tmean), 4),
           "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte_np, tmean, thr)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([row]).to_csv(os.path.join(RESULTS_DIR, f"dl_eval_{case}.csv"), index=False)
    np.savez(os.path.join(RESULTS_DIR, f"dl_probs_{case}.npz"),
             val=vmean, test=tmean, yva=yva_np, yte=yte_np, pid_te=pid_te, pid_va=pid_va)

    print("\n==================== Phase 2 gate G2 ====================")
    print(f"  baseline (parent plain CNN1D 1day): {BASELINE_1DAY_CNN}")
    print(f"  attn-CNN1D 5-seed: per-seed {row['test_AUROC_mean_seed']}±{row['test_AUROC_sd_seed']} | "
          f"ENSEMBLE {row['test_AUROC_ens']} {row['AUROC_CI95']}")
    print(f"  AUPRC {row['AUPRC']} | F2 {row['F2']} F3 {row['F3']} | P {row['Precision']} R {row['Recall']}")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default=CASE)
    a = ap.parse_args()
    run(a.case)
