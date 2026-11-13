"""
matrix_machine_1day.py — MACHINE-ONLY pipeline: every encoder (standalone + ensemble w/ 3 classical)
====================================================================================================
Encoders under the SAME exp_3 recipe (5 seeds, best-val, window_agg 30-ch): CNN1D (mean pooling),
CNN-attention (exp_3 AttnCNN1D), RNN (BiGRU mean), patch-Transformer. Machine-only DL = sequence encoder
+ head (no clinical branch). Each reported as standalone DL and as the ensemble = logistic stack of the
DL + top-3 base/rich machine classical models. 1-day. Writes exp_4/results/matrix_machine_1day.csv.
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
for p in [os.path.join(os.path.dirname(_AS), "Deep_Learning_Models", "models"),
          os.path.join(os.path.dirname(_AS), "Machine_Clinical_Models"),
          _AS, os.path.join(_AS, "models"),
          os.path.join(_AS, "exp_3", "features"), os.path.join(_AS, "exp_3", "results"),
          os.path.join(_AS, "exp_3", "dl")]:
    sys.path.insert(0, p)
import train_eval as TE                              # noqa: E402 (exp_3/dl -> AttnCNN1D + reduced seq)
import classical_rich as CR                          # noqa: E402
import make_resampled_variants as MR                 # noqa: E402
from fusion import build_seq_encoder                 # noqa: E402 (Deep_Learning_Models/models)
from sklearn.base import clone                       # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)
CASE = "1day"; CH, DROP = 64, 0.3
SEQ_KW = {"cnn1d": {"channels": 64, "n_blocks": 3, "kernel": 5, "dropout": DROP},
          "rnn": {"hidden": 128, "layers": 1, "cell": "gru", "dropout": DROP},
          "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 2, "ff": 128, "dropout": DROP}}
ENC = [("cnn1d", "CNN1D (mean pool)"), ("attn_cnn", "CNN-attention"),
       ("rnn", "RNN (BiGRU)"), ("transformer", "patch-Transformer")]


class MachineEncoderModel(nn.Module):
    def __init__(self, enc, in_ch):
        super().__init__()
        self.seq_enc, dim = build_seq_encoder(enc, in_ch=in_ch, **SEQ_KW[enc])
        self.head = nn.Sequential(nn.Linear(dim, CH), nn.ReLU(), nn.Dropout(DROP), nn.Linear(CH, 1))

    def forward(self, x, mask):
        return self.head(self.seq_enc(x, mask)).squeeze(-1)


def _sampler(y):
    yv = y.numpy(); w = np.where(yv == 1, 1/max(yv.sum(), 1), 1/max((yv == 0).sum(), 1))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(yv), True)


def _dl_probs(enc, case):
    Xva, mva, yva_t, _, _ = TE._load("val", case)
    Xte, mte, yte_t, _, _ = TE._load("test", case)
    yva, yte = yva_t.numpy().astype(int), yte_t.numpy().astype(int)
    if enc == "attn_cnn":                            # reuse exp_3 attention-CNN1D
        vps, tps = [], []
        for sd in range(TE.N_SEEDS):
            model, _ = TE.train_one(case, sd, DEV)
            vps.append(TE._infer(model, Xva, mva, DEV)); tps.append(TE._infer(model, Xte, mte, DEV))
            print(f"    attn seed {sd}: test {roc_auc_score(yte, tps[-1]):.4f}", flush=True)
        return np.array(vps), np.array(tps), yva, yte
    Xtr, mtr, ytr, _, _ = TE._load("train", case)
    vps, tps = [], []
    for sd in range(TE.N_SEEDS):
        TE._set_seed(sd)
        model = MachineEncoderModel(enc, Xtr.shape[-1]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=TE.LR, weight_decay=TE.WEIGHT_DECAY)
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, mtr, ytr), batch_size=TE.BATCH, sampler=_sampler(ytr))
        best = (-1, None, 0)
        for ep in range(TE.MAX_EPOCHS):
            model.train()
            for xb, mb, yb in loader:
                opt.zero_grad(); loss = lossf(model(xb.to(DEV), mb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, TE._infer(model, Xva, mva, DEV))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= TE.EARLY_STOP_PATIENCE:
                break
        model.load_state_dict(best[1])
        vps.append(TE._infer(model, Xva, mva, DEV)); tps.append(TE._infer(model, Xte, mte, DEV))
        print(f"    {enc} seed {sd}: test {roc_auc_score(yte, tps[-1]):.4f}", flush=True)
    return np.array(vps), np.array(tps), yva, yte


def _machine_classical(case):
    out = {}
    for fs in ("base", "rich"):
        tr, va, te, feats = CR._load(fs, case)
        Xtr, ytr = CR._xy(tr, feats); Xva, _ = CR._xy(va, feats); Xte, _ = CR._xy(te, feats)
        imp = CR.SimpleImputer(strategy="median").fit(Xtr)
        Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
        Xtr_sc, Xte_sc, prep = CR.scale_features(Xtr, Xte); Xva_sc = CR.transform_with_preprocessor(prep, Xva)
        chosen = CR.mrmr_select(Xtr_sc, ytr, list(feats), min(CR.N_MRMR, Xtr_sc.shape[1]))
        ci = [feats.index(c) for c in chosen]
        Xtr_s, Xva_s, Xte_s = Xtr_sc[:, ci], Xva_sc[:, ci], Xte_sc[:, ci]
        for m in CR.MODELS:
            best = CR._tune(m, Xtr_s, ytr)
            if best is None:
                continue
            vp, tp = CR._multiseed_probs(best, Xtr_s, ytr, Xva_s, Xte_s)
            out[f"cls:{fs}:{m}"] = (vp, tp)
    return out


def _row(model, comp, vp, tp, yva, yte, seed):
    by = {r["protocol"]: r for r in MR._rows_for_variant(comp, CASE, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    return {"branch": model, "component": comp, "sel_seed": seed,
            "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"), "full_F1": g("full", "F1"),
            "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
            "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
            "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}


def main():
    cls = _machine_classical(CASE)
    rows = []
    for enc, label in ENC:
        print(f"==== machine-only / {label} ====", flush=True)
        dv, dt, yva, yte = _dl_probs(enc, CASE)
        S = dv.shape[0]
        s_d = int(np.argmax([roc_auc_score(yva, dv[s]) for s in range(S)]))
        rows.append(_row(label, "DL", dv[s_d], dt[s_d], yva, yte, s_d))
        stack = []
        for s in range(S):
            c = {"dl": (dv[s], dt[s])}
            for n, (vp, tp) in cls.items():
                c[n] = (vp[s], tp[s])
            vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
            cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
            names = ["dl"] + cls3
            Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
            mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
            stack.append((mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]))
        s_s = int(np.argmax([roc_auc_score(yva, v) for v, _ in stack]))
        rows.append(_row(label, "Ensemble (DL + 3 classical)", stack[s_s][0], stack[s_s][1], yva, yte, s_s))
        print(f"  {label}: DL {rows[-2]['full_AUROC']} | Ensemble {rows[-1]['full_AUROC']}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "matrix_machine_1day.csv"), index=False)
    print(f"saved -> {os.path.join(OUT,'matrix_machine_1day.csv')}")


if __name__ == "__main__":
    main()
