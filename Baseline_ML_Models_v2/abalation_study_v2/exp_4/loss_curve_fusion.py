"""
loss_curve_fusion.py — training/validation loss curves for the DL core of the Windowed-Transformer
Fusion Stack (the tuned transformer-fusion network; the classical members + logistic meta-learner have
no epoch-wise loss). Retrains the exact WIN config (5 seeds, same protocol as the reported model) and
records per-epoch TRAIN and VAL binary-cross-entropy (both on the full split, for comparability) plus
val AUROC (the early-stopping signal). Plots the best-val seed. GPU.
Writes results/loss_history_1day.csv + figures/loss_curve_windowed_transformer_fusion.png.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stack_with_tuned_transformer as ST                 # noqa: E402
import hp_search_transformer as HPT                       # noqa: E402
import ablation_ensemble_branch as ABL                    # noqa: E402
CF = ST.CF; DEV = ST.DEV; WIN = ST.WIN
CASE = "1day"
N_SEEDS, EPOCHS, PATIENCE = 5, 120, 15
FIG = os.path.join(_HERE, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)


def _logits(model, X, m, c, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs].to(DEV), m[i:i + bs].to(DEV), c[i:i + bs].to(DEV)).cpu())
    return torch.cat(out)


def train_seed(sd, data, clin):
    Xtr, mtr, ctr, ytr = data["train"][0], data["train"][1], clin["train"], data["train"][2]
    Xva, mva, cva = data["val"][0], data["val"][1], clin["val"]
    yva = data["val"][4].astype(int)
    torch.manual_seed(sd); np.random.seed(sd)
    model = HPT.TFusion(Xtr.shape[-1], ctr.shape[-1], WIN["d_model"], WIN["layers"],
                        WIN["heads"], WIN["ff"], WIN["dropout"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=WIN["lr"], weight_decay=WIN["wd"])
    lossf = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, mtr, ctr, ytr), batch_size=CF.BATCH, sampler=ABL._sampler(ytr))
    hist = []
    best = (-1.0, 0)                                       # (best val AUROC, patience)
    best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, mb, cb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(DEV), mb.to(DEV), cb.to(DEV)), yb.to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        # full-split losses (eval mode) for comparable curves
        tr_loss = float(lossf(_logits(model, Xtr, mtr, ctr), ytr))
        va_logit = _logits(model, Xva, mva, cva)
        va_loss = float(lossf(va_logit, torch.from_numpy(yva.astype(np.float32))))
        va_auc = roc_auc_score(yva, torch.sigmoid(va_logit).numpy())
        hist.append({"seed": sd, "epoch": ep + 1, "train_loss": tr_loss,
                     "val_loss": va_loss, "val_auroc": va_auc})
        if va_auc > best[0]:
            best = (va_auc, 0); best_ep = ep + 1
        else:
            best = (best[0], best[1] + 1)
        if best[1] >= PATIENCE:
            break
    return pd.DataFrame(hist), best[0], best_ep


def main():
    data, clin = ABL._fusion_data(CASE)
    all_hist, seed_best = [], []
    for sd in range(N_SEEDS):
        h, bva, bep = train_seed(sd, data, clin)
        all_hist.append(h); seed_best.append((sd, bva, bep))
        print(f"seed {sd}: best val AUROC {bva:.4f} @ epoch {bep} ({len(h)} epochs run)", flush=True)
    hist = pd.concat(all_hist, ignore_index=True)
    hist.to_csv(os.path.join(OUT, "loss_history_1day.csv"), index=False)

    # representative = best-val seed (how the reported model is chosen)
    bsd, bva, bep = max(seed_best, key=lambda t: t[1])
    h = hist[hist.seed == bsd]
    print(f"\nplotting best-val seed {bsd} (val AUROC {bva:.4f}, early-stop epoch {bep})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(h.epoch, h.train_loss, color="#264653", lw=2.2, label="training loss")
    ax.plot(h.epoch, h.val_loss, color="#e76f51", lw=2.2, label="validation loss")
    ax.axvline(bep, color="#2a9d8f", ls="--", lw=1.8, label=f"early stop (epoch {bep})")
    ax.set_xlabel("epoch", fontsize=15)
    ax.set_ylabel("binary cross-entropy loss", fontsize=15)
    ax.set_title("Windowed-Transformer Fusion (DL core) — loss curves (1-day)\n"
                 f"d_model 64 · 1 layer · 4 heads · ff 128 · dropout 0.4 · best-val seed {bsd}",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG, "loss_curve_windowed_transformer_fusion.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", out, "+ results/loss_history_1day.csv")


if __name__ == "__main__":
    main()
