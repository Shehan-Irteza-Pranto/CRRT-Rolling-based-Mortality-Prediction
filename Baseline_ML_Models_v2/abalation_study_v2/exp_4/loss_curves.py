"""
loss_curves.py — train/val loss curves for the two ensemble DL cores (Day-1, best-val seed)
===========================================================================================
The machine Ensemble's DL core = Attention-CNN1D; the combined Ensemble/Stack's DL core = Attn-CNN
fusion. Neither logged per-epoch loss, so we retrain each once (1-day, best-val seed = 1) with the SAME
recipe/early-stop as the saved bundle, logging per-epoch train_loss + val_loss (BCE) + val AUROC.
Writes exp_4/figures/loss_curve_{attn_cnn1d,fusion}_1day.png (300 dpi) + history CSVs. GPU.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                # noqa: E402 (sets paths; FusionModel + helpers)
sys.path.insert(0, os.path.join(_AS, "exp_3", "dl"))
import train_eval as TE                              # noqa: E402
from model import build_model                        # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

FIG = os.path.join(_HERE, "figures"); os.makedirs(FIG, exist_ok=True)
CUR = os.path.join(_HERE, "results", "curves"); os.makedirs(CUR, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASE = "1day"
SEED = 1                                             # best-val Day-1 seed for both cores
BCE = nn.BCEWithLogitsLoss()


def _logits(model, tensors, clin=None, bs=256):
    model.eval(); out = []
    X, m = tensors[0], tensors[1]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            if clin is None:
                lg = model(X[i:i+bs].to(DEV), m[i:i+bs].to(DEV))
            else:
                lg = model(X[i:i+bs].to(DEV), m[i:i+bs].to(DEV), clin[i:i+bs].to(DEV))
            out.append(lg.cpu())
    return torch.cat(out)


def _sampler(y):
    yv = y.numpy(); npos = yv.sum(); nneg = len(yv) - npos
    w = np.where(yv == 1, 1.0 / max(npos, 1), 1.0 / max(nneg, 1))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(yv), replacement=True)


def train_attn():
    TE._set_seed(SEED)
    Xtr, mtr, ytr, *_ = TE._load("train", CASE)
    Xva, mva, yva, *_ = TE._load("val", CASE)
    yva_np = yva.numpy()
    model = build_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=TE.LR, weight_decay=TE.WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(Xtr, mtr, ytr), batch_size=TE.BATCH, sampler=_sampler(ytr))
    hist, best = [], (-1.0, -1)
    for ep in range(1, TE.MAX_EPOCHS + 1):
        model.train(); ls, n = 0.0, 0
        for xb, mb, yb in loader:
            opt.zero_grad(); loss = BCE(model(xb.to(DEV), mb.to(DEV)), yb.to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ls += float(loss) * len(yb); n += len(yb)
        vlog = _logits(model, (Xva, mva))
        vloss = float(BCE(vlog, yva)); vau = roc_auc_score(yva_np, torch.sigmoid(vlog).numpy())
        hist.append({"epoch": ep, "train_loss": ls / n, "val_loss": vloss, "val_auroc": vau})
        if vau > best[0]:
            best = (vau, ep)
        elif ep - best[1] >= TE.EARLY_STOP_PATIENCE:
            break
    return pd.DataFrame(hist), best[1]


def train_fusion():
    torch.manual_seed(SEED); np.random.seed(SEED)
    # rebuild fusion tensors exactly as combined_fusion_ensemble._fusion_probs
    data = {}
    for split in ("train", "val"):
        Xr, m, y, pid, day = CF.DD.load_reduced(split, CASE)
        keys = CF._machine_keys(CASE, split)
        data[split] = [torch.from_numpy(Xr).float(), torch.from_numpy(m).bool(),
                       torch.from_numpy(y.astype(np.float32)), keys, y]
    Xc_tr, cl = CF._clinical_aligned(CASE, "train", data["train"][3])
    imp = CF.SimpleImputer(strategy="median").fit(Xc_tr)
    sc = CF.StandardScaler().fit(imp.transform(Xc_tr))
    ytr_int = data["train"][4].astype(int)
    chosen = CF.mrmr_select(sc.transform(imp.transform(Xc_tr)), ytr_int, cl, min(CF.N_CLIN_DL, len(cl)))
    cidx = [cl.index(c) for c in chosen]
    clin = {}
    for split in ("train", "val"):
        Xc, _ = CF._clinical_aligned(CASE, split, data[split][3])
        clin[split] = torch.from_numpy(sc.transform(imp.transform(Xc))[:, cidx].astype(np.float32))
    Xtr, mtr, ytr = data["train"][0], data["train"][1], data["train"][2]
    Xva, mva, yva = data["val"][0], data["val"][1], data["val"][2]
    yva_np = data["val"][4].astype(int)
    ctr, cva = clin["train"], clin["val"]
    model = CF.FusionModel(Xtr.shape[-1], ctr.shape[-1], pooling="attention").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CF.LR, weight_decay=CF.WD)
    loader = DataLoader(TensorDataset(Xtr, mtr, ctr, ytr), batch_size=CF.BATCH, sampler=_sampler(ytr))
    hist, best = [], (-1.0, -1)
    for ep in range(1, CF.EPOCHS + 1):
        model.train(); ls, n = 0.0, 0
        for xb, mb, cb, yb in loader:
            opt.zero_grad(); loss = BCE(model(xb.to(DEV), mb.to(DEV), cb.to(DEV)), yb.to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ls += float(loss) * len(yb); n += len(yb)
        vlog = _logits(model, (Xva, mva), clin=cva)
        vloss = float(BCE(vlog, yva)); vau = roc_auc_score(yva_np, torch.sigmoid(vlog).numpy())
        hist.append({"epoch": ep, "train_loss": ls / n, "val_loss": vloss, "val_auroc": vau})
        if vau > best[0]:
            best = (vau, ep)
        elif ep - best[1] >= CF.PATIENCE:
            break
    return pd.DataFrame(hist), best[1]


def plot(df, best_ep, title, out):
    fig, axL = plt.subplots(figsize=(7, 4.5))
    axL.plot(df.epoch, df.train_loss, "C0-o", ms=3, lw=1.6, label="train loss")
    axL.plot(df.epoch, df.val_loss, "C3-^", ms=3, lw=1.6, label="val loss")
    axL.set_xlabel("epoch"); axL.set_ylabel("BCE loss"); axL.grid(alpha=0.25)
    axL.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best-val epoch ({best_ep})")
    axR = axL.twinx()
    axR.plot(df.epoch, df.val_auroc, "C2:s", ms=2.5, lw=1.2, alpha=0.6, label="val AUROC")
    axR.set_ylabel("val AUROC", color="C2"); axR.tick_params(axis="y", colors="C2")
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    axL.set_title(title, fontsize=10); fig.tight_layout()
    fig.savefig(out, dpi=300); plt.close(fig); print(f"  saved {out}", flush=True)


def main():
    print(f"device={DEV}", flush=True)
    for name, tag, fn in [("Attention-CNN1D (machine Ensemble core)", "attn_cnn1d", train_attn),
                          ("Attn-CNN fusion (combined Stack core)", "fusion", train_fusion)]:
        print(f"== training {tag} (1day, seed {SEED}) ==", flush=True)
        df, best_ep = fn()
        df.to_csv(os.path.join(CUR, f"{tag}_1day_history.csv"), index=False)
        plot(df, best_ep, f"{name} — 1-day (seed {SEED})",
             os.path.join(FIG, f"loss_curve_{tag}_1day.png"))
    print("done")


if __name__ == "__main__":
    main()
