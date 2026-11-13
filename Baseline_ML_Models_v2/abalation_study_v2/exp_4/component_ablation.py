"""
component_ablation.py — ablate the deployed combined model along its three components:
  +/- machine   : the Transformer encoder (machine sequence) branch of the DL fusion
  +/- clinical  : the clinical MLP branch of the DL fusion
  +/- classical : the top-3 classical base learners in the logistic stack

The DL fusion (machine+clinical) is trained ONCE (tuned WIN config, 5 seeds, classical at best-features
40 machine + 60 clinical). Each branch is ablated at inference by zeroing its embedding before the fusion
head (deployed-model branch ablation). The classical base learners are toggled in/out of the logistic
stack. Best-of-5-seeds by validation AUROC; metrics (full + 1:1 + 1:2) via MR._rows_for_variant.
Writes results/component_ablation_1day.csv.  GPU + CPU.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "exp_3", "results"))
import stack_with_tuned_transformer as ST                 # noqa: E402
import ablation_ensemble_branch as ABL                    # noqa: E402
from fusion import build_seq_encoder                      # noqa: E402
import make_resampled_variants as MR                      # noqa: E402
CF = ST.CF; DEV = ST.DEV; WIN = ST.WIN
CASE = "1day"; N_SEEDS = 5; EPOCHS, PATIENCE = 120, 15
CF.N_MACHINE, CF.N_CLINICAL = 40, 60                       # classical base learners at best features
OUT = os.path.join(_HERE, "results", "component_ablation_1day.csv")


class TFusionDrop(nn.Module):
    """Tuned transformer fusion with per-branch drop (zero a branch embedding before the head)."""
    def __init__(self, in_ch, clin_dim):
        super().__init__()
        self.seq_enc, seq_dim = build_seq_encoder("transformer", in_ch=in_ch, d_model=WIN["d_model"],
                                                  n_heads=WIN["heads"], n_layers=WIN["layers"],
                                                  ff=WIN["ff"], dropout=WIN["dropout"])
        self.clin = nn.Sequential(nn.Linear(clin_dim, CF.CH), nn.BatchNorm1d(CF.CH), nn.ReLU(),
                                  nn.Dropout(WIN["dropout"]))
        self.head = nn.Sequential(nn.Linear(seq_dim + CF.CH, CF.CH), nn.ReLU(), nn.Dropout(WIN["dropout"]),
                                  nn.Linear(CF.CH, 1))

    def forward(self, x, mask, clin, drop=None):
        s = self.seq_enc(x, mask); c = self.clin(clin)
        if drop == "machine":
            s = torch.zeros_like(s)
        if drop == "clinical":
            c = torch.zeros_like(c)
        return self.head(torch.cat([s, c], 1)).squeeze(-1)


def _infer(model, X, m, c, drop, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i+bs].to(DEV), m[i:i+bs].to(DEV),
                                           c[i:i+bs].to(DEV), drop=drop)).cpu().numpy())
    return np.concatenate(out)


def train_fusion(data, clin):
    Xtr, mtr, ctr, ytr = data["train"][0], data["train"][1], clin["train"], data["train"][2]
    Xva, mva, cva = data["val"][0], data["val"][1], clin["val"]
    Xte, mte, cte = data["test"][0], data["test"][1], clin["test"]
    yva = data["val"][4].astype(int)
    P = {"fusion": ([], []), "machine": ([], []), "clinical": ([], [])}   # drop=None / drop clinical / drop machine
    for sd in range(N_SEEDS):
        torch.manual_seed(sd); np.random.seed(sd)
        model = TFusionDrop(Xtr.shape[-1], ctr.shape[-1]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=WIN["lr"], weight_decay=WIN["wd"])
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, mtr, ctr, ytr), batch_size=CF.BATCH, sampler=ABL._sampler(ytr))
        best = (-1, None, 0)
        for ep in range(EPOCHS):
            model.train()
            for xb, mb, cb, yb in loader:
                opt.zero_grad(); loss = lossf(model(xb.to(DEV), mb.to(DEV), cb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, _infer(model, Xva, mva, cva, None))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= PATIENCE:
                break
        model.load_state_dict(best[1])
        for name, drop in [("fusion", None), ("machine", "clinical"), ("clinical", "machine")]:
            P[name][0].append(_infer(model, Xva, mva, cva, drop))
            P[name][1].append(_infer(model, Xte, mte, cte, drop))
        print(f"  seed {sd}: fusion val {roc_auc_score(yva, P['fusion'][0][-1]):.4f}", flush=True)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in P.items()}, yva


def _best_stack(dl_key, use_K, dl, comb, yva, yte):
    """Best-of-5-seeds forced stack for a component config -> (val_probs, test_probs)."""
    best = (-1.0, None, None, None, None)
    for s in range(N_SEEDS):
        c = {}
        if dl_key:
            c["dl"] = (dl[dl_key][0][s], dl[dl_key][1][s])
        if use_K:
            cls = {f"cls:{m}": (d[0][s], d[1][s]) for m, d in comb.items()}
            top3 = sorted(cls, key=lambda n: roc_auc_score(yva, cls[n][0]), reverse=True)[:3]
            for n in top3:
                c[n] = cls[n]
        names = list(c)
        if len(names) == 1:
            vp, tp = c[names[0]]
        else:
            Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
            mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
            vp, tp = mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]
        v = roc_auc_score(yva, vp)
        if v > best[0]:
            best = (v, vp, tp, s, names)
    return best[1], best[2], best[3], best[4]


def main():
    data, clin = ABL._fusion_data(CASE)
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
    print("training deployed fusion (5 seeds, drop-capable) ...", flush=True)
    dl, _ = train_fusion(data, clin)
    print("classical base learners (combined 40x60) ...", flush=True)
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")],
                               "combined", per_seed=True)

    # (machine, clinical, classical) -> (dl_key, use_K)
    CONFIGS = [
        (1, 1, 1, "fusion", True),    # full deployed model
        (1, 1, 0, "fusion", False),   # DL fusion only
        (1, 0, 1, "machine", True),   # machine branch + classical
        (0, 1, 1, "clinical", True),  # clinical branch + classical
        (0, 0, 1, None, True),        # classical only
        (1, 0, 0, "machine", False),  # machine branch only
        (0, 1, 0, "clinical", False),  # clinical branch only
    ]
    rows = []
    for M, C, K, dl_key, useK in CONFIGS:
        vp, tp, sel, members = _best_stack(dl_key, useK, dl, comb, yva, yte)
        by = {r["protocol"]: r for r in MR._rows_for_variant("cfg", CASE, vp, tp, yva, yte)}
        g = lambda pr, k: round(by[pr][k], 3)
        rows.append({"machine": "+" if M else "−", "clinical": "+" if C else "−",
                     "classical": "+" if K else "−",
                     "AUROC": g("full", "AUROC"), "AUPRC": g("full", "AUPRC"), "F1": g("full", "F1"),
                     "Precision": g("full", "Precision"), "Recall": g("full", "Recall"),
                     "AUROC_1to1": g("1:1", "AUROC"), "AUPRC_1to1": g("1:1", "AUPRC"),
                     "AUROC_1to2": g("1:2", "AUROC"), "AUPRC_1to2": g("1:2", "AUPRC"),
                     "sel_seed": sel, "members": "+".join(n.split(":")[-1] for n in members)})
        # per-day probabilities (PID/Visit_no/Day order of the machine table) for bootstrap_ci.py
        slug = f"M{M}C{C}K{K}"
        for split, y, pr in (("val", yva, vp), ("test", yte, tp)):
            keys = data[split][3][["PID", "Visit_no", "Day"]].copy(); keys["label"] = y; keys["prob"] = pr
            keys.to_csv(os.path.join(_HERE, "results", f"component_ablation_{slug}_{split}_probs_{CASE}.csv"), index=False)
        print(f"  M{rows[-1]['machine']} C{rows[-1]['clinical']} K{rows[-1]['classical']} -> "
              f"AUROC {rows[-1]['AUROC']}  [{rows[-1]['members']}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\n=== component ablation (deployed combined model) ===")
    print(out.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
