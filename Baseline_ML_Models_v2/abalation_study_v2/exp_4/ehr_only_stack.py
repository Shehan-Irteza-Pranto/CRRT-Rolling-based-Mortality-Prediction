"""
ehr_only_stack.py — EHR-only (clinical-only) counterpart of the deployed Windowed-Transformer Fusion Stack.
=========================================================================================================
The deployed model = DL fusion (transformer machine branch + clinical-MLP branch) + top-3 classical
(combined 40 machine + 60 clinical) -> logistic meta-learner fit on validation. This script builds the
same stack with NO machine data at all:

  DL member (two candidates)
    - clin-branch : the fusion model's clinical branch trained ALONE (same mRMR-40 clinical tensors as
                    ABL._fusion_data, same CF.CH width / WIN dropout / AdamW lr-wd / balanced sampler /
                    120 epochs, patience 15 / 5 seeds) = the deployed DL fusion minus the sequence encoder.
    - mlp60       : the Table III "Clinical MLP" (clinical mRMR-60, randomized HP search), re-loaded from
                    the 5 seeds saved by rerun_clinical_mlp_60.py (results/saved_models/dl/*clinical_only_per_day_mlp_1day_s*.pt).
  Classical members
    - the 5 CF base learners (GradBoost / RF / LightGBM / XGBoost / LogReg) on the clinical-only table,
      (a) at clinical mRMR-60 (= Table IV "Clinical-only, 60") and (b) on ALL clinical features (what the
      deployed stack's clinical-classical candidates used).
  Stacks: DL + top-3-by-val classical -> LogisticRegression meta-learner on validation; best-of-5 seeds
  by validation AUROC; threshold = max-F1 on validation; metrics full / 1:1 / 1:2 via MR._rows_for_variant.
  Also: classical-only top-3 stack, and a top-4-by-val stack over all clinical candidates.

Writes results/ehr_only_stack_1day.csv and results/ehr_only_stack_{val,test}_probs_1day.csv (primary stack).
GPU (tiny) + CPU (classical tuning).
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
for p in [_AS, os.path.join(_AS, "models"), os.path.join(_AS, "eval")]:
    sys.path.insert(0, p)

import feature_select as FS                                    # noqa: E402 (parent models/feature_select.py)
FS.SELECTION_DIR = os.path.join(_AS, "results", "feature_selection_clin60")   # Table III clin-60 cache
import stack_with_tuned_transformer as ST                      # noqa: E402
import ablation_ensemble_branch as ABL                         # noqa: E402
import make_resampled_variants as MR                           # noqa: E402
import classical_rich as CR                                    # noqa: E402
_cfg_mcm = sys.modules.pop("config", None)                  # Machine_Clinical_Models/config was cached by CF;
import dl_clinical_mlp as DM                                   # noqa: E402  dl_clinical_mlp needs Deep_Learning_Models/config
if _cfg_mcm is not None:
    sys.modules["config"] = _cfg_mcm
from pipeline_utils import scale_features, transform_with_preprocessor   # noqa: E402
CF = ST.CF; DEV = ST.DEV; WIN = ST.WIN
CASE = "1day"; N_SEEDS = 5; EPOCHS, PATIENCE = 120, 15
N_CLIN_CLASSICAL = 60
SPLITS = ("train", "val", "test")
OUT_DIR = os.path.join(_HERE, "results")
OUT = os.path.join(OUT_DIR, f"ehr_only_stack_{CASE}.csv")


# ------------------------------------------------------------------ row alignment
def _order_map(src_keys, tgt_keys):
    """index array such that src[idx] is in tgt row order (keys = DataFrames with KEY_COLS)."""
    s = pd.DataFrame(np.asarray(src_keys)[:, :3], columns=CF.KEY_COLS).astype(int)
    s["_i"] = np.arange(len(s))
    t = pd.DataFrame(np.asarray(tgt_keys)[:, :3], columns=CF.KEY_COLS).astype(int)
    m = t.merge(s, on=CF.KEY_COLS, how="left")
    assert m["_i"].notna().all() and len(m) == len(t), "row sets differ between clinical and machine tables"
    return m["_i"].to_numpy(int)


# ------------------------------------------------------------------ DL member 1: fusion clinical branch alone
class ClinBranchNet(nn.Module):
    """TFusion with the sequence branch removed: clinical MLP branch -> head."""
    def __init__(self, clin_dim):
        super().__init__()
        self.clin = nn.Sequential(nn.Linear(clin_dim, CF.CH), nn.BatchNorm1d(CF.CH), nn.ReLU(),
                                  nn.Dropout(WIN["dropout"]))
        self.head = nn.Sequential(nn.Linear(CF.CH, CF.CH), nn.ReLU(), nn.Dropout(WIN["dropout"]),
                                  nn.Linear(CF.CH, 1))

    def forward(self, clin):
        return self.head(self.clin(clin)).squeeze(-1)


def _infer_c(model, c, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(c), bs):
            out.append(torch.sigmoid(model(c[i:i+bs].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def train_clin_branch(data, clin):
    ctr, ytr = clin["train"], data["train"][2]
    cva, cte = clin["val"], clin["test"]
    yva = data["val"][4].astype(int); yte = data["test"][4].astype(int)
    vps, tps = [], []
    for sd in range(N_SEEDS):
        torch.manual_seed(sd); np.random.seed(sd)
        model = ClinBranchNet(ctr.shape[-1]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=WIN["lr"], weight_decay=WIN["wd"])
        lossf = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(ctr, ytr), batch_size=CF.BATCH, sampler=ABL._sampler(ytr))
        best = (-1, None, 0)
        for ep in range(EPOCHS):
            model.train()
            for cb, yb in loader:
                opt.zero_grad(); loss = lossf(model(cb.to(DEV)), yb.to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            va = roc_auc_score(yva, _infer_c(model, cva))
            if va > best[0]:
                best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            else:
                best = (best[0], best[1], best[2] + 1)
            if best[2] >= PATIENCE:
                break
        model.load_state_dict(best[1])
        vps.append(_infer_c(model, cva)); tps.append(_infer_c(model, cte))
        print(f"  clin-branch seed {sd}: val {best[0]:.4f}  test {roc_auc_score(yte, tps[-1]):.4f}  "
              f"(stopped ep {ep+1})", flush=True)
    return np.array(vps), np.array(tps)


# ------------------------------------------------------------------ DL member 2: Table III clinical MLP (60) reloaded
def mlp60_probs(tgt_val_keys, tgt_test_keys):
    tr = DM.TabularClinicalDataset(CASE, "train", "per_day", "current", "denoised")
    va = DM.TabularClinicalDataset(CASE, "val", "per_day", "current", "denoised", prep=tr.prep)
    te = DM.TabularClinicalDataset(CASE, "test", "per_day", "current", "denoised", prep=tr.prep)
    assert tr.in_dim == N_CLIN_CLASSICAL, f"expected clin-60 selection cache, got in_dim={tr.in_dim}"
    iv, it = _order_map(va.keys, tgt_val_keys), _order_map(te.keys, tgt_test_keys)
    vps, tps = [], []
    for s in range(N_SEEDS):
        p = os.path.join(DM.RESULTS_DIR, "saved_models", "dl",
                         f"current_denoised_clinical_only_per_day_mlp_{CASE}_s{s}.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["in_dim"] == tr.in_dim
        m = DM.MLP(ck["in_dim"], hidden=tuple(ck["hidden"]), dropout=ck["dropout"]); m.load_state_dict(ck["state_dict"]); m.eval()
        with torch.no_grad():
            vp = torch.sigmoid(m(va.X)).numpy(); tp = torch.sigmoid(m(te.X)).numpy()
        vps.append(vp[iv]); tps.append(tp[it])
        print(f"  mlp60 seed {s}: val {roc_auc_score(va.y.numpy()[iv], vps[-1]):.4f}  "
              f"test {roc_auc_score(te.y.numpy()[it], tps[-1]):.4f}", flush=True)
    return np.array(vps), np.array(tps)


# ------------------------------------------------------------------ classical members (clinical-only)
def clinical_classical(n_sel, tgt_val_keys, tgt_test_keys):
    tabs = [pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv")) for s in SPLITS]
    tr, va, te = tabs
    feats = [x for x in tr.columns if x not in CF.META]
    Xtr, ytr = tr[feats].to_numpy(float), tr["label"].to_numpy(int)
    Xva, Xte = va[feats].to_numpy(float), te[feats].to_numpy(float)
    imp = CF.SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
    Xtr, Xte, prep = scale_features(Xtr, Xte); Xva = transform_with_preprocessor(prep, Xva)
    if n_sel:
        idx, names = FS.select_columns(feats, Xtr, ytr, "clinical_only", n_clinical=n_sel)
        Xtr, Xva, Xte = Xtr[:, idx], Xva[:, idx], Xte[:, idx]
        print(f"  clinical mRMR-{n_sel}: {len(names)} features", flush=True)
    else:
        print(f"  clinical ALL: {len(feats)} features", flush=True)
    iv = _order_map(va[CF.KEY_COLS].to_numpy(), tgt_val_keys)
    it = _order_map(te[CF.KEY_COLS].to_numpy(), tgt_test_keys)
    out = {}
    for m in CF.MODELS:
        best = CR._tune(m, Xtr, ytr)
        if best is None:
            continue
        vp, tp = CR._multiseed_probs(best, Xtr, ytr, Xva, Xte)       # [N_SEEDS, n]
        out[m] = (vp[:, iv], tp[:, it])
        print(f"    {m:18s} val AUROC (seed-mean) {np.mean([roc_auc_score(YVA, v) for v in out[m][0]]):.4f}", flush=True)
    return out


# ------------------------------------------------------------------ stacking
def _fit_stack(c, names, yva):
    if len(names) == 1:
        return c[names[0]]
    Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
    mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
    return mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]


def best_seed_stack(dl, dl_key, cls, yva, k_cls=3, top_k_any=None):
    """dl: {key: (vp[S,n], tp[S,n])} or None; cls: {model: (vp[S,n], tp[S,n])}.
    Per seed: candidates -> members -> stack; best-of-seeds by val AUROC. Returns (vp, tp, seed, members)."""
    best = (-1.0, None, None, None, None)
    for s in range(N_SEEDS):
        c = {}
        if dl_key:
            c[dl_key] = (dl[dl_key][0][s], dl[dl_key][1][s])
        for m, d in cls.items():
            c[f"cls:{m}"] = (d[0][s], d[1][s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        if top_k_any:
            members = sorted(vauc, key=vauc.get, reverse=True)[:top_k_any]
        else:
            top = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:k_cls]
            members = ([dl_key] if dl_key else []) + top
        vp, tp = _fit_stack(c, members, yva)
        v = roc_auc_score(yva, vp)
        if v > best[0]:
            best = (v, vp, tp, s, members)
    return best[1], best[2], best[3], best[4]


def main():
    global YVA
    data, clin = ABL._fusion_data(CASE)                 # machine-key row order, mRMR-40 clinical tensors
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int); YVA = yva
    kva, kte = data["val"][3][CF.KEY_COLS].to_numpy(), data["test"][3][CF.KEY_COLS].to_numpy()
    print(f"rows: val {len(yva)} (pos {yva.sum()})  test {len(yte)} (pos {yte.sum()})", flush=True)

    print("\n[1] clinical-branch DL alone (5 seeds) ...", flush=True)
    cb_vp, cb_tp = train_clin_branch(data, clin)
    print("\n[2] Table III clinical MLP-60 (reload 5 seeds) ...", flush=True)
    m60_vp, m60_tp = mlp60_probs(kva, kte)
    dl = {"dl:clin_branch": (cb_vp, cb_tp), "dl:mlp60": (m60_vp, m60_tp)}

    print(f"\n[3] clinical-only classical @ mRMR-{N_CLIN_CLASSICAL} ...", flush=True)
    cls60 = clinical_classical(N_CLIN_CLASSICAL, kva, kte)
    print("\n[4] clinical-only classical @ ALL features ...", flush=True)
    clsall = clinical_classical(0, kva, kte)

    rows = []
    def add(name, vp, tp, seed=None, members=None, note=""):
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
        for split, keys, y, p in (("val", kva, yva, vp), ("test", kte, yte, tp)):
            d = pd.DataFrame(keys, columns=CF.KEY_COLS); d["label"] = y; d["prob"] = p
            d.to_csv(os.path.join(OUT_DIR, f"ehr_only_stack_{slug}_{split}_probs_{CASE}.csv"), index=False)
        rows.append(ST._row(name, CASE, vp, tp, yva, yte,
                            {"sel_seed": seed, "members": "+".join(m.split(":")[-1] for m in members) if members else "",
                             "note": note}))
        r = rows[-1]
        print(f"  {name:62s} AUROC {r['full_AUROC']}  AUPRC {r['full_AUPRC']}  F1 {r['full_F1']}  "
              f"1:1 {r['r1to1_AUROC']}  1:2 {r['r1to2_AUROC']}  [{r['members']}]", flush=True)

    print("\n=== EHR-only results (held-out test, best-val seed, thr = max-F1 on val) ===", flush=True)
    # single models
    s = int(np.argmax([roc_auc_score(yva, v) for v in cb_vp]))
    add("DL: clinical branch alone (mRMR-40)", cb_vp[s], cb_tp[s], s, ["dl:clin_branch"])
    s = int(np.argmax([roc_auc_score(yva, v) for v in m60_vp]))
    add("DL: Table III clinical MLP (mRMR-60, reloaded)", m60_vp[s], m60_tp[s], s, ["dl:mlp60"],
        note="should match Table III Clinical MLP 0.710")
    # primary stack
    vp, tp, s, mem = best_seed_stack(dl, "dl:clin_branch", cls60, yva)
    add("EHR-only Stack: clin-branch DL + top-3 clinical classical (mRMR-60)", vp, tp, s, mem, note="PRIMARY")
    pv, pt = vp, tp
    vp, tp, s, mem = best_seed_stack(dl, "dl:mlp60", cls60, yva)
    add("EHR-only Stack: MLP-60 + top-3 clinical classical (mRMR-60)", vp, tp, s, mem)
    vp, tp, s, mem = best_seed_stack(dl, None, cls60, yva)
    add("EHR-only Stack: top-3 clinical classical only (mRMR-60)", vp, tp, s, mem)
    # top-4 by validation over all clinical candidates (2 DL + 5 classical@60)
    best = (-1.0, None, None, None, None)
    for sd in range(N_SEEDS):
        c = {k: (v[0][sd], v[1][sd]) for k, v in dl.items()}
        c.update({f"cls:{m}": (d[0][sd], d[1][sd]) for m, d in cls60.items()})
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        mem = sorted(vauc, key=vauc.get, reverse=True)[:4]
        vp, tp = _fit_stack(c, mem, yva); v = roc_auc_score(yva, vp)
        if v > best[0]:
            best = (v, vp, tp, sd, mem)
    add("EHR-only Stack: top-4 by val over all clinical candidates (mRMR-60)", best[1], best[2], best[3], best[4])
    # all-feature classical variants (deployed stack's clinical-classical convention)
    vp, tp, s, mem = best_seed_stack(dl, "dl:clin_branch", clsall, yva)
    add("EHR-only Stack: clin-branch DL + top-3 clinical classical (ALL feats)", vp, tp, s, mem)
    vp, tp, s, mem = best_seed_stack(dl, None, clsall, yva)
    add("EHR-only Stack: top-3 clinical classical only (ALL feats)", vp, tp, s, mem)

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    for split, keys, y, p in (("val", kva, yva, pv), ("test", kte, yte, pt)):
        d = pd.DataFrame(keys, columns=CF.KEY_COLS); d["label"] = y; d["prob"] = p
        d.to_csv(os.path.join(OUT_DIR, f"ehr_only_stack_{split}_probs_{CASE}.csv"), index=False)
    print(f"\nreference: deployed combined Stack (40 machine + 60 clinical) = 0.766; component-ablation "
          f"'machine branch zeroed' row = 0.755 (classical still combined)")
    print(f"saved -> {OUT}", flush=True)


YVA = None
if __name__ == "__main__":
    main()
