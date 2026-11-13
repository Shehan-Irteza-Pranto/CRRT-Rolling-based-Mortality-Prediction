"""
ablation_ensemble_branch.py — swap the machine BRANCH of the best ensemble (combined Ensemble/Stack)
====================================================================================================
The best model is the combined Ensemble/Stack, whose DL component is a CNN-fusion: an attention-CNN1D
machine branch + a clinical-MLP branch. This ablation asks: what if the machine branch is a different
sequence encoder? We rebuild the fusion with three machine encoders — attention-CNN1D (current, baseline),
RNN (BiGRU), and a patch-Transformer — keeping the SAME clinical branch (mRMR-40), training recipe, and
stacking, and re-select the best-of-5-seeds by validation AUROC. Everything else (combined-classical +
clinical-classical stack members, meta-learner) is identical, so the delta isolates the encoder.
Writes exp_4/results/ablation_branch_<case>.csv. GPU.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                # noqa: E402 (paths + FusionModel/helpers/classical/stack)
from fusion import build_seq_encoder                 # noqa: E402 (Deep_Learning_Models/models: cnn1d/rnn/transformer)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
MR = CF.MR
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)

# encoder configs for the swapped machine branch (native pooling per encoder)
SEQ_KW = {"rnn": {"hidden": 128, "layers": 1, "cell": "gru", "dropout": CF.DROP},
          "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 2, "ff": 128, "dropout": CF.DROP},
          "cnn1d": {"channels": 64, "n_blocks": 3, "kernel": 5, "dropout": CF.DROP}}  # masked-mean pooling


class GeneralFusion(nn.Module):
    """Machine sequence encoder (rnn/transformer, native pooling) + clinical-MLP branch, late-fused."""
    def __init__(self, encoder_type, in_ch, clin_dim):
        super().__init__()
        self.seq_enc, seq_dim = build_seq_encoder(encoder_type, in_ch=in_ch, **SEQ_KW[encoder_type])
        self.clin = nn.Sequential(nn.Linear(clin_dim, CF.CH), nn.BatchNorm1d(CF.CH), nn.ReLU(),
                                  nn.Dropout(CF.DROP))
        self.head = nn.Sequential(nn.Linear(seq_dim + CF.CH, CF.CH), nn.ReLU(), nn.Dropout(CF.DROP),
                                  nn.Linear(CF.CH, 1))

    def forward(self, x, mask, clin):
        return self.head(torch.cat([self.seq_enc(x, mask), self.clin(clin)], 1)).squeeze(-1)


def _fusion_data(case):
    """Reduced machine sequence + mRMR-40 clinical branch tensors (train/val/test) — mirrors CF._fusion_probs."""
    data = {}
    for split in ("train", "val", "test"):
        Xr, m, y, pid, day = CF.DD.load_reduced(split, case)
        keys = CF._machine_keys(case, split)
        data[split] = [torch.from_numpy(Xr).float(), torch.from_numpy(m).bool(),
                       torch.from_numpy(y.astype(np.float32)), keys, y]
    Xc_tr, cl = CF._clinical_aligned(case, "train", data["train"][3])
    imp = CF.SimpleImputer(strategy="median").fit(Xc_tr)
    sc = CF.StandardScaler().fit(imp.transform(Xc_tr))
    ytr = data["train"][4].astype(int)
    chosen = CF.mrmr_select(sc.transform(imp.transform(Xc_tr)), ytr, cl, min(CF.N_CLIN_DL, len(cl)))
    cidx = [cl.index(c) for c in chosen]
    clin = {}
    for split in ("train", "val", "test"):
        Xc, _ = CF._clinical_aligned(case, split, data[split][3])
        clin[split] = torch.from_numpy(sc.transform(imp.transform(Xc))[:, cidx].astype(np.float32))
    return data, clin


def _sampler(y):
    yv = y.numpy(); w = np.where(yv == 1, 1/max(yv.sum(), 1), 1/max((yv == 0).sum(), 1))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(yv), True)


def _infer(model, X, m, c, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i+bs].to(DEV), m[i:i+bs].to(DEV),
                                            c[i:i+bs].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def fusion_probs_encoder(case, encoder_type):
    """Per-seed (val,test) probs for the fusion with the given machine encoder. attn_cnn reuses exp_3."""
    if encoder_type == "attn_cnn":
        fv, ft, yva, yte, pid = CF._fusion_probs(case, per_seed=True)   # current attention-CNN fusion
        return fv, ft, yva, yte
    data, clin = _fusion_data(case)
    Xtr, mtr, ytr = data["train"][0], data["train"][1], data["train"][2]
    Xva, mva = data["val"][0], data["val"][1]; Xte, mte = data["test"][0], data["test"][1]
    ctr, cva, cte = clin["train"], clin["val"], clin["test"]
    yva = data["val"][4].astype(int); yte = data["test"][4].astype(int)
    in_ch, clin_dim = Xtr.shape[-1], ctr.shape[-1]
    vps, tps = [], []
    for sd in range(CF.N_SEEDS_DL):
        torch.manual_seed(sd); np.random.seed(sd)
        model = GeneralFusion(encoder_type, in_ch, clin_dim).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=CF.LR, weight_decay=CF.WD)
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, mtr, ctr, ytr), batch_size=CF.BATCH, sampler=_sampler(ytr))
        best = (-1, None, 0)
        for ep in range(CF.EPOCHS):
            model.train()
            for xb, mb, cb, yb in loader:
                opt.zero_grad(); loss = lossf(model(xb.to(DEV), mb.to(DEV), cb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, _infer(model, Xva, mva, cva))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= CF.PATIENCE:
                break
        model.load_state_dict(best[1])
        vps.append(_infer(model, Xva, mva, cva)); tps.append(_infer(model, Xte, mte, cte))
        print(f"    [{encoder_type} {case}] seed {sd}: test {roc_auc_score(yte, tps[-1]):.4f}", flush=True)
    return np.array(vps), np.array(tps), yva, yte


def _metrics_row(branch, comp, case, vp, tp, yva, yte, seed):
    by = {r["protocol"]: r for r in MR._rows_for_variant(comp, case, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    return {"branch": branch, "component": comp, "case": case, "sel_seed": seed,
            "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"), "full_F1": g("full", "F1"),
            "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
            "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
            "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day"); a = ap.parse_args()
    case = a.case
    # classical stack members (identical across branches) — compute once
    comb = CF._classical_probs([CF._combined_table(case, s) for s in ("train", "val", "test")],
                               "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, case, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)
    rows = []
    for branch, label in [("attn_cnn", "CNN-attention (current)"), ("rnn", "RNN (BiGRU)"),
                          ("transformer", "patch-Transformer")]:
        print(f"==== branch = {label} ====", flush=True)
        fv, ft, yva, yte = fusion_probs_encoder(case, branch)
        S = fv.shape[0]
        # Fusion-DL alone: best-val seed
        s_f = int(np.argmax([roc_auc_score(yva, fv[s]) for s in range(S)]))
        rows.append(_metrics_row(label, "Fusion-DL", case, fv[s_f], ft[s_f], yva, yte, s_f))
        # Two stacks per seed: (a) top-4 by val (the current best method — may drop the DL fusion),
        # (b) fusion + top-3 classical (forces the swapped branch INTO the ensemble).
        top4, forced, fusion_in_top4 = [], [], []
        for s in range(S):
            c = {"dl:fusion": (fv[s], ft[s])}
            for m, d in comb.items():                       # d = (vp[S,n], tp[S,n])
                c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
            for m, d in clin.items():
                c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
            vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}

            def _stack(names):
                Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
                mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
                return mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]
            t4 = sorted(vauc, key=vauc.get, reverse=True)[:4]
            top4.append(_stack(t4)); fusion_in_top4.append("dl:fusion" in t4)
            cls_ranked = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
            forced.append(_stack(["dl:fusion"] + cls_ranked))
        s4 = int(np.argmax([roc_auc_score(yva, v) for v, _ in top4]))
        sf = int(np.argmax([roc_auc_score(yva, v) for v, _ in forced]))
        r4 = _metrics_row(label, "Stack top-4 (best method)", case, top4[s4][0], top4[s4][1], yva, yte, s4)
        r4["fusion_in_top4"] = bool(fusion_in_top4[s4]); rows.append(r4)
        rows.append(_metrics_row(label, "Stack fusion+top3-classical", case, forced[sf][0], forced[sf][1], yva, yte, sf))
        print(f"  {label}: Fusion-DL {rows[-3]['full_AUROC']} | Stack-top4 {rows[-2]['full_AUROC']} "
              f"(fusion_in_top4={fusion_in_top4[s4]}) | Stack-forced-fusion {rows[-1]['full_AUROC']} "
              f"(P {rows[-1]['full_Precision']} R {rows[-1]['full_Recall']})", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"ablation_branch_{case}.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, f'ablation_branch_{case}.csv')}")


if __name__ == "__main__":
    main()
