"""
models/dl_data.py
=================
Datasets for the aggregated-DL models.

Per circuit-day (PID, Visit_no, Day=D) a sample is:
  * current-day raw stream  x [T,4], mask [T]   (reuse Deep_Learning_Models SequenceDataset)
  * prior-days machine summary  [208]           expanding {mean,min,max,median} of the
                                                52 day-stats over days 1..D-1 (zeros if none)
  * clinical summary  [n_clin]                  clinical_agg_day vector at day D (combined only)
  * label y

Scalers for the two summary branches are fit on TRAIN and reused for val/test.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
sys.path.insert(0, _HERE)
from abl_config import seq_cache_dir, feature_dir, MRMR_ENABLED, N_MRMR_CLINICAL  # noqa: E402
from dl_reduce import reduce_stream                                 # noqa: E402
from feature_select import get_or_compute_selection                # noqa: E402

_META = {"PID", "Visit_no", "Day", "Date", "Death_Date", "Mortality", "days_until_death", "label"}
_SUMMARY_STATS = ["mean", "min", "max", "median"]


class _Seq:
    """Self-contained loader for the abalation sequence cache (.npz)."""
    def __init__(self, case, split, variant, peak):
        d = np.load(os.path.join(seq_cache_dir(variant, peak), f"{case}_{split}.npz"))
        self.X = torch.from_numpy(d["X"]).float()          # [N, T, 4]
        self.mask = torch.from_numpy(d["mask"]).bool()     # [N, T]
        self.y = torch.from_numpy(d["y"]).float()          # [N]
        self.keys = d["keys"]                              # [N, 3]


def _day_stats_table(case, split, variant, peak):
    """Per-day 52-stat machine table (abalation root), keyed PID,Visit_no,Day."""
    path = os.path.join(feature_dir(variant, peak, "machine_only", False, case), f"{split}.csv")
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in _META]
    return df, feats


def _prior_days_summary(df, feats):
    """Map (PID,Visit_no,Day) -> expanding {mean,min,max,median} of `feats` over days < Day.

    Returns (lookup dict, out_dim). Days with no prior history map to a zero vector.
    """
    out = {}
    out_dim = len(feats) * len(_SUMMARY_STATS)
    for (pid, vis), g in df.sort_values(["PID", "Visit_no", "Day"]).groupby(["PID", "Visit_no"]):
        vals = g[feats].to_numpy(dtype=float)
        days = g["Day"].to_numpy()
        for i, d in enumerate(days):
            prior = vals[:i]                       # strictly earlier days
            if len(prior) == 0:
                vec = np.zeros(out_dim, dtype=np.float32)
            else:
                with np.errstate(all="ignore"):
                    vec = np.concatenate([
                        np.nanmean(prior, axis=0), np.nanmin(prior, axis=0),
                        np.nanmax(prior, axis=0), np.nanmedian(prior, axis=0),
                    ]).astype(np.float32)
                vec = np.nan_to_num(vec, nan=0.0)
            out[(int(pid), int(vis), int(d))] = vec
    return out, out_dim


def _clinical_agg_table(case, split, variant, peak):
    """clinical aggregated vector per (PID,Visit_no,Day); None if the table is absent."""
    path = os.path.join(feature_dir(variant, peak, "clinical_only", True, case), f"{split}.csv")
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in _META]
    lut = {(int(r.PID), int(r.Visit_no), int(r.Day)): df.loc[i, feats].to_numpy(dtype=np.float32)
           for i, r in df.iterrows()}
    return lut, feats, len(feats)


class AggDataset(Dataset):
    """Current-day sequence + optional prior-days machine summary + clinical summary."""

    def __init__(self, case, split, pipeline, scope, variant="current", peak="denoised",
                 mach_scaler=None, clin_scaler=None,
                 want_machine=None, want_clinical=None, reduce=None):
        self.seq = _Seq(case, split, variant, peak)
        keys = [tuple(int(v) for v in k) for k in self.seq.keys]
        n = len(keys)

        # reduce the long current-day stream (window-stat aggregation or patching)
        Xr, maskr, self.seq_feat_dim = reduce_stream(
            self.seq.X.numpy(), self.seq.mask.numpy(), reduce)
        self.Xr = torch.from_numpy(Xr).float()
        self.maskr = torch.from_numpy(maskr).bool()

        # decide which branches are active
        self.use_machine = (scope == "agg") if want_machine is None else want_machine
        self.use_clinical = (pipeline == "combined") if want_clinical is None else want_clinical

        # prior-days machine summary
        self.machine_dim = 0
        if self.use_machine:
            df, feats = _day_stats_table(case, split, variant, peak)
            lut, dim = _prior_days_summary(df, feats)
            M = np.stack([lut.get(k, np.zeros(dim, np.float32)) for k in keys])
            if mach_scaler is None:
                mach_scaler = StandardScaler().fit(np.nan_to_num(M))
            self.M = torch.from_numpy(mach_scaler.transform(np.nan_to_num(M)).astype(np.float32))
            self.machine_dim = dim
        self.mach_scaler = mach_scaler

        # clinical summary (combined only) — mRMR-selected on TRAIN, reused for val/test
        self.clinical_dim = 0
        self.C = None
        if self.use_clinical:
            lut, feats, dim = _clinical_agg_table(case, split, variant, peak)
            if lut is not None:
                Cn = np.stack([lut.get(k, np.zeros(dim, np.float32)) for k in keys])
                if clin_scaler is None:                       # TRAIN: fit impute+scale, then mRMR
                    imp = SimpleImputer(strategy="median"); Cn = imp.fit_transform(Cn)
                    sc = StandardScaler().fit(Cn); Cn = sc.transform(Cn)
                    if MRMR_ENABLED:
                        ckey = f"{variant}_{peak}_clinical_branch_{case}"
                        sel, _ = get_or_compute_selection(
                            ckey, feats, Cn, self.seq.y.numpy().astype(int), "clinical_only")
                    else:
                        sel = list(range(dim))
                    clin_scaler = (sc, imp, sel)
                else:                                          # VAL/TEST: reuse fitted objects
                    sc, imp, sel = clin_scaler
                    Cn = sc.transform(imp.transform(Cn))
                Cn = Cn[:, sel]
                self.C = torch.from_numpy(Cn.astype(np.float32))
                self.clinical_dim = len(sel)
            else:
                self.use_clinical = False
        self.clin_scaler = clin_scaler
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, mask, y = self.Xr[i], self.maskr[i], self.seq.y[i]
        m = self.M[i] if self.machine_dim else torch.zeros(0)
        c = self.C[i] if self.clinical_dim else torch.zeros(0)
        return x, mask, m, c, y


def pos_weight(dataset):
    y = dataset.seq.y
    n_pos = float(y.sum()); n_neg = float(len(y) - n_pos)
    return torch.tensor(n_neg / max(n_pos, 1.0))
