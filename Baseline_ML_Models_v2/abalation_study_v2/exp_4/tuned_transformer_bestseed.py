"""
tuned_transformer_bestseed.py — best-seed full-test AUROC for the HP-search winner transformer fusion.
Trains the winning config (d_model64/L1/h4/ff128/dropout0.4/wd5e-3/lr3e-4) with 5 seeds, records per-seed
(val, test) AUROC, and reports the BEST-VALIDATION seed's full-test AUROC + patient-bootstrap CI (the
leak-safe 'best seed' number), alongside best-test seed (reference) and the seed-average.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ablation_ensemble_branch as ABL                # noqa: E402
import hp_search_transformer as HPT                   # noqa: E402 (TFusion, _boot_ci)
from sklearn.metrics import roc_auc_score             # noqa: E402
CF = ABL.CF
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIN = dict(d_model=64, layers=1, heads=4, ff=128, dropout=0.4, wd=5e-3, lr=3e-4)
N_SEEDS, EPOCHS, PATIENCE, CASE = 5, 120, 15, "1day"


def main():
    data, clin = ABL._fusion_data(CASE)
    Xtr, mtr, ctr, ytr = data["train"][0], data["train"][1], clin["train"], data["train"][2]
    Xva, mva, cva = data["val"][0], data["val"][1], clin["val"]
    Xte, mte, cte = data["test"][0], data["test"][1], clin["test"]
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
    pid_te = data["test"][3]["PID"].to_numpy()

    rows, tps = [], []
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
        vp = ABL._infer(model, Xva, mva, cva); tp = ABL._infer(model, Xte, mte, cte); tps.append(tp)
        va_auc, te_auc = roc_auc_score(yva, vp), roc_auc_score(yte, tp)
        lo, hi = HPT._boot_ci(yte, tp, pid_te)
        rows.append({"seed": sd, "val_AUROC": round(va_auc, 4), "test_AUROC": round(te_auc, 4),
                     "test_CI95": f"[{lo}, {hi}]"})
        print(f"  seed {sd}: val {va_auc:.4f} | test {te_auc:.4f}  CI [{lo}, {hi}]", flush=True)

    bv = max(rows, key=lambda r: r["val_AUROC"])       # best-VALIDATION seed (leak-safe)
    bt = max(rows, key=lambda r: r["test_AUROC"])       # best-TEST seed (cherry-pick, reference only)
    tmean = np.mean(tps, 0); avg = roc_auc_score(yte, tmean); alo, ahi = HPT._boot_ci(yte, tmean, pid_te)
    print("\n================ tuned transformer fusion (winner config), 1-day ================")
    print(f"  BEST-VAL seed  = s{bv['seed']}: val {bv['val_AUROC']} -> FULL-TEST AUROC {bv['test_AUROC']}  CI {bv['test_CI95']}   <-- the legit best-seed number")
    print(f"  best-TEST seed = s{bt['seed']}: test {bt['test_AUROC']} (cherry-picked on test; not valid to report)")
    print(f"  seed-averaged  : full-test AUROC {round(avg,4)}  CI [{alo}, {ahi}]")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(_HERE, "results", "tuned_transformer_bestseed_1day.csv"), index=False)


if __name__ == "__main__":
    main()
