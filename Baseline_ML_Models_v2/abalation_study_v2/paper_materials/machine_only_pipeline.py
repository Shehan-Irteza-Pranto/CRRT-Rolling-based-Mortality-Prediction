"""
machine_only_pipeline.py
========================================================================================
End-to-end, single-file reproduction of the BEST machine-only CRRT-mortality model:
raw pressure signal  →  gap-aware day signal  →  two representation branches
(A: richer engineered features + classical models ; B: window-reduced attention CNN1D)
→  probability ensemble  →  day-level evaluation with patient-bootstrap CI.

This is the consolidated paper artifact for the exp_3 machine-only ensemble. It assembles the
already-validated exp_3 modules into one readable script. Three small, generic helpers are imported
from the project (mRMR selection, window aggregation, the CNN conv encoder); everything specific to
this pipeline is inline. Classical models use fixed, documented hyperparameters (the tuned values)
so the script is deterministic — full RandomizedSearchCV tuning lives in exp_3/features/classical_rich.py.

Leak-safe: impute / scale / mRMR / CNN weights fit on TRAIN; ensemble meta-learner and the decision
threshold fit on VAL; all reported metrics on the held-out TEST set; CIs bootstrap over PATIENTS.

Usage:
    conda activate crrt
    python machine_only_pipeline.py --case 1day      # or --case 2day
========================================================================================
"""
import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------
# 0. PATHS & CONFIG
# ---------------------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))                 # .../abalation_study_v2/paper_materials
_AS_DIR = os.path.dirname(_HERE)                                     # .../abalation_study_v2
_V2_DIR = os.path.dirname(_AS_DIR)                                   # .../Baseline_ML_Models_v2
sys.path.insert(0, os.path.join(_AS_DIR, "models"))                 # mrmr_select, window_aggregate
sys.path.insert(0, _AS_DIR)                                         # abl_config
sys.path.insert(0, os.path.join(_V2_DIR, "Deep_Learning_Models", "models"))   # CNN1DEncoder

from abl_config import DENOISE_VARIANTS, feature_dir, KEY_COLS       # noqa: E402
from feature_select import mrmr_select                              # noqa: E402  (mRMR, project)
from dl_reduce import window_aggregate                              # noqa: E402  (5-stat window reduce)

RAW_PRESSURE = DENOISE_VARIANTS["current"]["denoised"]              # per-minute denoised pressures
RAW4 = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE"]

# signal
DAY_LEN, N_CHANNELS, GAP_MAX_MIN = 1440, 6, 10
PREFIX = {0: "ACCESS_PRESSURE", 1: "FILTER_PRESSURE", 2: "EFFLUENT_PRESSURE",
          3: "RETURN_PRESSURE", 4: "TMP", 5: "PFD"}
MIN_VALID, EOD_WIN = 30, 120

# modelling
N_MRMR   = 30           # machine features kept by mRMR
N_SEEDS  = 5            # multi-seed averaging (classical refit & CNN training)
N_BOOTSTRAP = 1000      # patient bootstrap for the CI
META_COLS = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}

# CNN / training
REDUCE_K = 10           # window length (min) -> 1440/10 = 144 windows
CNN_CHANNELS, CNN_BLOCKS, CNN_KERNEL, CNN_DROPOUT = 64, 3, 5, 0.3
LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE = 3e-4, 1e-4, 64, 120, 15


def machine_table(split, case):
    """Parent machine_only per-day table = circuit-day membership + label (+ parent 133 features)."""
    return os.path.join(feature_dir("current", "denoised", "machine_only", False, case), f"{split}.csv")


# =======================================================================================
# 1. SIGNAL PROCESSING — gap-aware 6-channel day signal in physical units
#    (rows come out in machine_table order, so features concat with no join)
# =======================================================================================
def _gap_fill_day(grid4, present, gap_max=GAP_MAX_MIN):
    mask = present.copy(); obs = np.where(present)[0]
    if len(obs) == 0:
        return grid4, mask
    first, last = obs[0], obs[-1]
    for a, b in zip(obs[:-1], obs[1:]):
        run = b - a - 1
        if run <= 0:
            continue
        if run <= gap_max:                                     # small gap -> linear interpolate
            for c in range(grid4.shape[1]):
                grid4[a:b + 1, c] = np.linspace(grid4[a, c], grid4[b, c], run + 2)
            mask[a + 1:b] = True
    mask[:first] = False; mask[last + 1:] = False              # large/edge gaps -> masked
    return grid4, mask


def _raw_index(pids):
    raw = pd.read_csv(RAW_PRESSURE, usecols=["PID", "Visit_no", "Day", "Time"] + RAW4)
    raw = raw[raw["PID"].isin(pids)]
    t = pd.to_datetime(raw["Time"])
    raw = raw.assign(_min=(t.dt.hour * 60 + t.dt.minute).to_numpy())
    idx = {}
    for key, gdf in raw.groupby(["PID", "Visit_no", "Day"], sort=False):
        gdf = gdf.drop_duplicates("_min")
        idx[tuple(int(k) for k in key)] = (gdf["_min"].to_numpy(), gdf[RAW4].to_numpy(np.float32))
    return idx


def load_raw_days(split, case):
    """[N,6,1440] physical (NaN at masked) + mask[N,1440] + keys(PID,Visit_no,Day,label)."""
    keys = pd.read_csv(machine_table(split, case), usecols=["PID", "Visit_no", "Day", "label"])
    idx = _raw_index(set(keys["PID"]))
    X = np.full((len(keys), N_CHANNELS, DAY_LEN), np.nan, np.float32)
    M = np.zeros((len(keys), DAY_LEN), bool)
    for i, r in enumerate(keys.itertuples()):
        g = idx.get((int(r.PID), int(r.Visit_no), int(r.Day)))
        grid4 = np.full((DAY_LEN, 4), np.nan, np.float32)
        if g is not None:
            mins, vals = g; grid4[mins] = vals
        grid4, mask = _gap_fill_day(grid4, ~np.isnan(grid4[:, 0]))
        a, f, e, rr = grid4[:, 0], grid4[:, 1], grid4[:, 2], grid4[:, 3]
        six = np.stack([a, f, e, rr, (f + rr) / 2 - e, f - rr], axis=0)   # access,filter,effl,ret,TMP,PFD
        six[:, ~mask] = np.nan
        X[i] = six; M[i] = mask
    return X, M, keys


# =======================================================================================
# 2. RICH PHYSIOLOGY FEATURES (Branch A input) — spectral / complexity / excursion /
#    end-of-day deterioration / cross-channel coupling. Concatenated onto the parent 133.
# =======================================================================================
def _contig(series):
    obs = np.where(~np.isnan(series))[0]
    if len(obs) < MIN_VALID:
        return None, None
    seg = series[obs[0]:obs[-1] + 1].astype(float)
    nan = np.isnan(seg)
    if nan.any():
        xp = np.where(~nan)[0]; seg[nan] = np.interp(np.where(nan)[0], xp, seg[xp])
    return seg, series[obs]


def _perm_entropy(x, order=3):
    n = len(x) - order + 1
    if n < order + 1:
        return np.nan
    patt = np.argsort(np.argsort(np.lib.stride_tricks.sliding_window_view(x, order), axis=1), axis=1)
    _, counts = np.unique(patt, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))


def _spectral(seg):
    f, pxx = welch(seg - seg.mean(), fs=1.0, nperseg=min(256, len(seg)))
    f, pxx = f[1:], pxx[1:]
    tot = pxx.sum()
    if tot <= 0 or len(f) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pn = pxx / tot
    return (float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn))),
            float(pn[f < 1 / 30].sum()), float(pn[f > 1 / 10].sum()),
            min(float(1.0 / f[int(np.argmax(pxx))]), 1440.0))


def _excursion(vals):
    med = np.median(vals); above = vals > med
    cross = int(np.sum(above[1:] != above[:-1]))
    best = run = 0
    for a in above:
        run = run + 1 if a else 0; best = max(best, run)
    return cross / len(vals), best / len(vals)


def _eod(seg):
    if len(seg) < 2 * EOD_WIN:
        return np.nan, np.nan
    tail = seg[-EOD_WIN:]
    return float(tail.mean() - seg[:EOD_WIN].mean()), float(np.polyfit(np.arange(len(tail)), tail, 1)[0])


def _channel_feats(series):
    seg, vals = _contig(series)
    keys = ["spec_entropy", "bandpow_lo", "bandpow_hi", "dom_period", "perm_entropy",
            "cross_rate", "max_run_above", "eod_delta", "eod_slope"]
    if seg is None:
        return dict.fromkeys(keys, np.nan)
    se, lo, hi, dom = _spectral(seg); cr, mra = _excursion(vals); ed, es = _eod(seg)
    return dict(zip(keys, [se, lo, hi, dom, _perm_entropy(seg), cr, mra, ed, es]))


def _corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < MIN_VALID or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _slope(series):
    obs = np.where(~np.isnan(series))[0]
    return float(np.polyfit(obs, series[obs], 1)[0]) if len(obs) >= MIN_VALID else np.nan


def _day_rich_row(x6):
    row = {}
    for c in range(6):
        for k, v in _channel_feats(x6[c]).items():
            row[f"{PREFIX[c]}_{k}"] = v
    for k, v in _channel_feats(x6[3] - x6[0]).items():          # ARD = RETURN - ACCESS
        row[f"ARD_{k}"] = v
    row["TMP_PFD_corr"] = _corr(x6[4], x6[5])
    row["ACCESS_RETURN_corr"] = _corr(x6[0], x6[3])
    row["FILTER_EFFLUENT_corr"] = _corr(x6[1], x6[2])
    st, sp = _slope(x6[4]), _slope(x6[5])
    row["TMP_PFD_cotrend"] = (np.sign(st) * np.sign(sp)) if not (np.isnan(st) or np.isnan(sp)) else np.nan
    return row


def build_rich_table(split, case):
    """Parent 133-feature table + 67 new physiology features -> 200-feature DataFrame (+ keys/label)."""
    X, M, keys = load_raw_days(split, case)
    base = pd.read_csv(machine_table(split, case))
    assert (base[KEY_COLS].to_numpy() == keys[KEY_COLS].to_numpy()).all(), "row order mismatch"
    new = pd.DataFrame([_day_rich_row(X[i]) for i in range(len(keys))])
    return pd.concat([base.reset_index(drop=True), new.reset_index(drop=True)], axis=1)


# =======================================================================================
# 3. BRANCH A — classical models on the 200 features (impute -> scale -> mRMR -> 5 models x 5 seeds)
# =======================================================================================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
try:
    import lightgbm as lgb
    import xgboost as xgb
    _HAS_BOOST = True
except ImportError:
    _HAS_BOOST = False


def _classical_registry():
    """Fixed, documented hyperparameters (the tuned operating point; see classical_rich.py)."""
    reg = {
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                               class_weight="balanced", n_jobs=-1, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                       learning_rate=0.05, random_state=0),
        "LogisticRegression": LogisticRegression(C=1.0, class_weight="balanced",
                                                 max_iter=5000, random_state=0),
    }
    if _HAS_BOOST:
        reg["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                             class_weight="balanced", n_jobs=-1, verbosity=-1, random_state=0)
        reg["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=0)
    return reg


def branch_A(case):
    """Return {model_name: (val_prob, test_prob)} + (yva, yte, pid_va, pid_te)."""
    tr, va, te = (build_rich_table(s, case) for s in ("train", "val", "test"))
    feats = [c for c in tr.columns if c not in META_COLS]
    Xtr, ytr = tr[feats].to_numpy(float), tr["label"].to_numpy(int)
    Xva, yva = va[feats].to_numpy(float), va["label"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["label"].to_numpy(int)

    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Xtr, Xva, Xte = (sc.transform(imp.transform(X)) for X in (Xtr, Xva, Xte))

    chosen = mrmr_select(Xtr, ytr, list(feats), min(N_MRMR, Xtr.shape[1]))
    ci = [feats.index(c) for c in chosen]
    Xtr, Xva, Xte = Xtr[:, ci], Xva[:, ci], Xte[:, ci]
    n_new = sum(1 for c in chosen if c not in set(pd.read_csv(machine_table("train", case), nrows=0).columns))
    print(f"  [A] mRMR kept {len(chosen)} features ({n_new} from the new physiology set)")

    out = {}
    for name, base_est in _classical_registry().items():
        vps, tps = [], []
        for sd in range(N_SEEDS):
            est = clone(base_est)
            try:
                est.set_params(random_state=sd)
            except ValueError:
                pass
            est.fit(Xtr, ytr)
            vps.append(est.predict_proba(Xva)[:, 1]); tps.append(est.predict_proba(Xte)[:, 1])
        out[name] = (np.mean(vps, 0), np.mean(tps, 0))
        print(f"  [A] {name:18s} test AUROC {roc_auc(yte, out[name][1]):.4f}")
    return out, yva, yte, va["PID"].to_numpy(), te["PID"].to_numpy()


# =======================================================================================
# 4. BRANCH B — attention CNN1D on the window-reduced signal (z-scored), 5 seeds
# =======================================================================================
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from cnn1d import CNN1DEncoder                                       # parent conv stack (reused)


class AttnCNN1D(nn.Module):
    """Parent conv encoder + masked ATTENTION pooling (weights end-of-day windows) + linear head."""
    def __init__(self, in_ch):
        super().__init__()
        self.enc = CNN1DEncoder(in_ch, CNN_CHANNELS, CNN_BLOCKS, CNN_KERNEL, CNN_DROPOUT)
        self.attn = nn.Sequential(nn.Linear(CNN_CHANNELS, CNN_CHANNELS // 2), nn.Tanh(),
                                  nn.Linear(CNN_CHANNELS // 2, 1))
        self.drop = nn.Dropout(CNN_DROPOUT); self.head = nn.Linear(CNN_CHANNELS, 1)

    def forward(self, x, mask):
        h = self.enc.conv(x.transpose(1, 2))                       # [B, C, T]
        score = self.attn(h.transpose(1, 2)).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(score, dim=1).unsqueeze(1)               # [B,1,T]
        return self.head(self.drop((h * a).sum(-1))).squeeze(-1)


def _zscore_fit(X, M):
    mean = np.zeros(N_CHANNELS, np.float32); std = np.ones(N_CHANNELS, np.float32)
    for c in range(N_CHANNELS):
        v = X[:, c, :][M & ~np.isnan(X[:, c, :])]
        if v.size:
            mean[c] = v.mean(); s = v.std(); std[c] = s if s > 1e-6 else 1.0
    return mean, std


def _reduced(split, case, norm):
    X, M, keys = load_raw_days(split, case)
    mean, std = norm
    Xn = (X - mean[None, :, None]) / std[None, :, None]
    Xn[np.isnan(Xn)] = 0.0
    Xr, mr = window_aggregate(np.transpose(Xn, (0, 2, 1)), M, k=REDUCE_K)   # [N,144,30],[N,144]
    return (Xr.astype(np.float32), mr, keys["label"].to_numpy(np.int64),
            keys["PID"].to_numpy(), keys["Day"].to_numpy())


def _train_cnn(Xtr, mtr, ytr, Xva, mva, yva, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AttnCNN1D(Xtr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    w = np.where(ytr == 1, 1.0 / max(ytr.sum(), 1), 1.0 / max((ytr == 0).sum(), 1))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ytr), True)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(mtr), torch.from_numpy(ytr).float())
    loader = DataLoader(ds, batch_size=BATCH, sampler=sampler)

    def infer(X, m):
        model.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i+256]).to(device); mb = torch.from_numpy(m[i:i+256]).to(device)
                ps.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
        return np.concatenate(ps)

    best = (-1.0, None, 0)
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(device), mb.to(device)), yb.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        va = roc_auc(yva, infer(Xva, mva))
        if va > best[0]:
            best = (va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
        else:
            best = (best[0], best[1], best[2] + 1)
        if best[2] >= PATIENCE:
            break
    model.load_state_dict(best[1])
    return infer


def branch_B(case):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X0, M0, _ = load_raw_days("train", case)             # fit z-score norm on TRAIN valid minutes
    norm = _zscore_fit(X0, M0)
    Xtr, mtr, ytr, _, _ = _reduced("train", case, norm)
    Xva, mva, yva, pid_va, _ = _reduced("val", case, norm)
    Xte, mte, yte, pid_te, _ = _reduced("test", case, norm)

    vps, tps, aucs = [], [], []
    for sd in range(N_SEEDS):
        infer = _train_cnn(Xtr, mtr, ytr, Xva, mva, yva, device, sd)
        vps.append(infer(Xva, mva)); tps.append(infer(Xte, mte)); aucs.append(roc_auc(yte, tps[-1]))
    print(f"  [B] attention CNN1D test AUROC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
          f"(ensemble {roc_auc(yte, np.mean(tps, 0)):.4f})")
    return (np.mean(vps, 0), np.mean(tps, 0)), yva, yte, pid_va, pid_te


# =======================================================================================
# 5. ENSEMBLE + 6. EVALUATION
# =======================================================================================
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score, precision_recall_curve)


def roc_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.0


def _opt_threshold(y, s):
    if np.sum(y) < 2:
        return 0.5
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))])


def _panel(y, s, thr):
    pred = (s >= thr).astype(int)
    return {"AUROC": round(roc_auc(y, s), 4), "AUPRC": round(average_precision_score(y, s), 4),
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
            aucs.append(roc_auc(y[idx], s[idx]))
    return (round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4))


def main(case):
    print(f"\n================ machine-only pipeline | case = {case} ================")
    print("[1-3] Branch A: signal -> rich features -> classical models")
    A, yva, yte, pid_va, pid_te = branch_A(case)
    print("[4]   Branch B: signal -> window-reduce -> attention CNN1D")
    (dl_val, dl_test), yva_b, yte_b, _, _ = branch_B(case)
    assert np.array_equal(yte, yte_b) and np.array_equal(yva, yva_b), "branch row mismatch"

    # ---- 5. ensemble (mean + logistic stack fit on VAL) -------------------------------
    from sklearn.linear_model import LogisticRegression as LR
    names = list(A.keys()) + ["attn_cnn1d"]
    Vv = np.column_stack([A[n][0] for n in A] + [dl_val])
    Vt = np.column_stack([A[n][1] for n in A] + [dl_test])

    rows = []
    for j, nm in enumerate(names):
        thr = _opt_threshold(yva, Vv[:, j]); lo, hi = _boot_ci(yte, Vt[:, j], pid_te)
        rows.append({"model": nm, "AUROC_CI95": f"[{lo}, {hi}]", **_panel(yte, Vt[:, j], thr)})
    mean_t, mean_v = Vt.mean(1), Vv.mean(1)
    lo, hi = _boot_ci(yte, mean_t, pid_te)
    rows.append({"model": "ENSEMBLE:mean", "AUROC_CI95": f"[{lo}, {hi}]",
                 **_panel(yte, mean_t, _opt_threshold(yva, mean_v))})
    meta = LR(max_iter=5000).fit(Vv, yva)
    stk_t, stk_v = meta.predict_proba(Vt)[:, 1], meta.predict_proba(Vv)[:, 1]
    lo, hi = _boot_ci(yte, stk_t, pid_te)
    rows.append({"model": "ENSEMBLE:logit_stack", "AUROC_CI95": f"[{lo}, {hi}]",
                 **_panel(yte, stk_t, _opt_threshold(yva, stk_v))})

    df = pd.DataFrame(rows)
    out = os.path.join(_HERE, f"machine_only_pipeline_{case}.csv"); df.to_csv(out, index=False)
    print("\n================ RESULTS (held-out test, day-level) ================")
    print(df.to_string(index=False))
    best = df.sort_values("AUROC", ascending=False).iloc[0]
    print(f"\nBest machine-only model: {best['model']}  AUROC {best['AUROC']} {best['AUROC_CI95']}")
    print(f"Saved -> {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1day", choices=["1day", "2day"])
    main(ap.parse_args().case)
