"""
models/dl_clinical_mlp.py
=========================
Clinical-only DL model: an MLP on the mRMR-selected clinical features (top-40), so
the DL family covers all three pipelines (clinical / machine / combined). Standalone
(does not import the sequence dl_train) — same rigor as the other DL models:
balanced minibatches, weighted/plain BCE, AdamW + LR scheduler, early-stop on val
AUROC, randomized HP search (keep best by val AUROC), resampled 1:1/1:2 eval,
per-epoch train/val-loss curve, and model saving.

CLI mirrors dl_train.py:
    python dl_clinical_mlp.py --scope agg --case 1day --seed 0 --search-iter 12 --out row.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ABL, "eval"))
from abl_config import feature_dir, RESULTS_DIR, DL_DIR, KEY_COLS           # noqa: E402
from subpop_eval import eval_dispatch                                       # noqa: E402
from feature_select import get_or_compute_selection                        # noqa: E402

sys.path.insert(0, DL_DIR)
sys.path.insert(0, os.path.join(DL_DIR, "models"))
from config import (MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, WEIGHT_DECAY, GRAD_CLIP,   # noqa: E402
                    LR_SCHED_FACTOR, LR_SCHED_PATIENCE, MIN_LR, USE_LR_SCHEDULER)
from mlp import MLP                                                         # noqa: E402

import torch                                                               # noqa: E402
import torch.nn as nn                                                      # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler    # noqa: E402
from sklearn.impute import SimpleImputer                                   # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402
from sklearn.metrics import roc_auc_score, precision_recall_curve         # noqa: E402

META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
KEYS = ["PID", "Visit_no", "Day"]
HIDDENS = [(128,), (256, 128), (256, 128, 64)]


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _opt_threshold(y, p):
    if np.sum(y) < 2:
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0.0)
    return float(th[int(np.argmax(f1))])


class TabularClinicalDataset(Dataset):
    """Clinical table -> mRMR top-40 -> impute+scale (train-fit, reused). Returns (X, y)."""
    def __init__(self, case, split, scope, variant, peak, prep=None):
        df = pd.read_csv(os.path.join(feature_dir(variant, peak, "clinical_only", scope == "agg", case),
                                      f"{split}.csv"))
        feats = [c for c in df.columns if c not in META]
        X = df[feats].to_numpy(dtype=float)
        self.keys = df[KEYS].to_numpy()
        self.y = torch.from_numpy(df["label"].to_numpy(np.float32))
        if prep is None:                                   # TRAIN: fit impute+scale, then mRMR
            imp = SimpleImputer(strategy="median").fit(X); Xi = imp.transform(X)
            sc = StandardScaler().fit(Xi); Xs = sc.transform(Xi)
            key = f"{variant}_{peak}_clinical_only_{scope}_{case}"
            sel, _ = get_or_compute_selection(key, feats, Xs, df["label"].to_numpy(int), "clinical_only")
            prep = (imp, sc, sel)
        else:
            imp, sc, sel = prep
            Xs = sc.transform(imp.transform(X))
        self.X = torch.from_numpy(Xs[:, sel].astype(np.float32))
        self.prep, self.in_dim = prep, self.X.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mlp_hp(n_iter, seed):
    rng = np.random.RandomState(seed)
    lrs, wds, drs, bss = [3e-4, 1e-3, 1e-4], [1e-3, 1e-4], [0.2, 0.3, 0.5], [32, 64]
    out = []
    for _ in range(max(1, n_iter)):
        out.append({"lr": lrs[rng.randint(3)], "weight_decay": wds[rng.randint(2)],
                    "dropout": drs[rng.randint(3)], "batch_size": bss[rng.randint(2)],
                    "hidden": HIDDENS[rng.randint(len(HIDDENS))]})
    return out


@torch.no_grad()
def _eval(model, ds, loss_fn, device):
    model.eval(); ps, ys, ls, n = [], [], 0.0, 0
    for X, y in DataLoader(ds, batch_size=256):
        logit = model(X.to(device)); ls += float(loss_fn(logit, y.to(device)).item()) * y.size(0); n += y.size(0)
        ps.append(torch.sigmoid(logit).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys), ls / max(n, 1)


def _fit(tr, va, hp, device, label="", hp_idx=0, history=None, log_every=1):
    model = MLP(tr.in_dim, hidden=hp["hidden"], dropout=hp["dropout"]).to(device)
    y = tr.y.numpy(); npos = y.sum(); nneg = len(y) - npos
    w = np.where(y == 1, 1.0 / max(npos, 1), 1.0 / max(nneg, 1))            # ~50/50 batches
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    loader = DataLoader(tr, batch_size=hp["batch_size"], sampler=sampler)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=LR_SCHED_FACTOR,
             patience=LR_SCHED_PATIENCE, min_lr=MIN_LR) if USE_LR_SCHEDULER else None)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); ls, n = 0.0, 0
        for X, yb in loader:
            opt.zero_grad(); logit = model(X.to(device)); loss = loss_fn(logit, yb.to(device))
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); ls += float(loss.item()) * yb.size(0); n += yb.size(0)
        vp, vy, vl = _eval(model, va, loss_fn, device)
        auroc = roc_auc_score(vy, vp) if len(np.unique(vy)) > 1 else 0.0
        if sched: sched.step(auroc)
        improved = auroc > best
        if improved:
            best, patience = auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if history is not None:
            history.append({"hp": hp_idx, "epoch": epoch, "train_loss": round(ls / max(n, 1), 5),
                            "val_loss": round(float(vl), 5), "val_auroc": round(float(auroc), 5)})
        if epoch % log_every == 0 or improved or patience >= EARLY_STOP_PATIENCE:
            print(f"[{label}] epoch {epoch:3d}: train_loss={ls/max(n,1):.4f}  val_loss={vl:.4f}  "
                  f"val_auroc={auroc:.4f}{'  *best' if improved else ''}", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            break
    return best, best_state


def _save_curve(history, best_j, mid, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [h for h in history if h["hp"] == best_j]
    if not rows:
        return ""
    ep = [r["epoch"] for r in rows]
    fig, (axL, axA) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axL.plot(ep, [r["train_loss"] for r in rows], "C0-o", ms=3, label="train loss")
    axL.plot(ep, [r["val_loss"] for r in rows], "C3-^", ms=3, label="val loss")
    axL.set_ylabel("loss"); axL.legend(fontsize=8); axL.set_title(title, fontsize=9)
    axA.plot(ep, [r["val_auroc"] for r in rows], "C1-s", ms=3, label="val AUROC")
    axA.set_ylabel("val AUROC"); axA.set_xlabel("epoch"); axA.legend(fontsize=8)
    fig.tight_layout(); cdir = os.path.join(RESULTS_DIR, "curves"); os.makedirs(cdir, exist_ok=True)
    p = os.path.join(cdir, mid + ".png"); fig.savefig(p, dpi=110); plt.close(fig); return p


def train_clinical_mlp(scope, case, seed, search_iter=0, variant="current", peak="denoised",
                       tag="", log_every=1):
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = tag or f"{variant}_{peak}"
    tr = TabularClinicalDataset(case, "train", scope, variant, peak)
    va = TabularClinicalDataset(case, "val",  scope, variant, peak, prep=tr.prep)
    te = TabularClinicalDataset(case, "test", scope, variant, peak, prep=tr.prep)

    hps = mlp_hp(search_iter, seed) if search_iter and search_iter > 0 else \
        [{"lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": 0.3, "batch_size": 32, "hidden": (256, 128)}]
    best = (-1.0, None, None, 1); history = []
    base = f"clinical_only/{scope}/mlp/{case}/s{seed}"
    for j, hp in enumerate(hps):
        a, st = _fit(tr, va, hp, device, label=f"{base} hp{j+1}/{len(hps)}", hp_idx=j + 1,
                     history=history, log_every=log_every)
        if a > best[0]:
            best = (a, st, hp, j + 1)
        print(f"   [hp {j+1}/{len(hps)}] val_auroc={a:.4f}{'  <- best' if best[1] is st else ''}", flush=True)
    best_auroc, best_state, best_hp, best_j = best

    mid = f"{tag}_clinical_only_{scope}_mlp_{case}_s{seed}"
    hist_dir = os.path.join(RESULTS_DIR, "dl_history"); os.makedirs(hist_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(hist_dir, mid + ".csv"), index=False)
    curve = _save_curve(history, best_j, mid, f"{base}  (best hp{best_j}, val AUROC {best_auroc:.3f})")

    model = MLP(tr.in_dim, hidden=best_hp["hidden"], dropout=best_hp["dropout"]).to(device)
    if best_state:
        model.load_state_dict(best_state)
    loss_fn = nn.BCEWithLogitsLoss()
    vp, vy, _ = _eval(model, va, loss_fn, device); thr = _opt_threshold(vy, vp)
    tp, ty, _ = _eval(model, te, loss_fn, device)
    te_keys = pd.DataFrame(np.asarray(te.keys), columns=KEY_COLS)       # test order matches tp/ty

    mdir = os.path.join(RESULTS_DIR, "saved_models", "dl"); os.makedirs(mdir, exist_ok=True)
    mpath = os.path.join(mdir, mid + ".pt")
    torch.save({"state_dict": best_state, "hidden": list(best_hp["hidden"]),
                "dropout": best_hp["dropout"], "in_dim": tr.in_dim, "scope": scope}, mpath)

    row = {"family": "deep", "feature_set": "clinical_only", "case": case, "scope": scope,
           "model": "mlp", "seed": seed, "variant": tag, "best_val_auroc": round(float(best_auroc), 4),
           "n_selected": tr.in_dim, "n_hp_tried": len(hps), "best_hp": json.dumps(
               {**best_hp, "hidden": list(best_hp["hidden"])}),
           "history_csv": os.path.join(hist_dir, mid + ".csv"), "curve_png": curve, "model_path": mpath}
    row.update(eval_dispatch(te_keys, ty, tp, thr, case, variant, peak))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="agg", choices=["per_day", "agg"])
    ap.add_argument("--case", default="1day")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="current")
    ap.add_argument("--peak", default="denoised")
    ap.add_argument("--tag", default="")
    ap.add_argument("--search-iter", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.epochs is not None:
        import config as _c; _c.MAX_EPOCHS = a.epochs; globals()["MAX_EPOCHS"] = a.epochs
    row = train_clinical_mlp(a.scope, a.case, a.seed, search_iter=a.search_iter,
                             variant=a.variant, peak=a.peak, tag=a.tag, log_every=a.log_every)
    print(f"[mlp] clinical_only/{a.scope}/{a.case}/s{a.seed}: tried {row['n_hp_tried']} hp | "
          f"val_auroc={row['best_val_auroc']} | full AUROC {row['full_test_AUROC']:.3f} | "
          f"subpop {row['subpop_AUROC_mean']}±{row['subpop_AUROC_std']} (n={row['n_eval_subpops']})")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        pd.DataFrame([row]).to_csv(a.out, index=False)
        print(f"[mlp] wrote {a.out}")
