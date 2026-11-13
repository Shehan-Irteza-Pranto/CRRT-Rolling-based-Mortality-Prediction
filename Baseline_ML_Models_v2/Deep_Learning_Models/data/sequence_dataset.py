"""
sequence_dataset.py
===================
torch Datasets over the .npz tensors built by prepare_sequences.py, plus a paired
sequence+clinical loader for the fusion rung. Tensors are already padded to MAX_LEN,
so collation is a plain stack; the boolean mask drives pooling / packing / attention
padding downstream.

Requires torch (see ../requirements.txt). Framework-free data prep lives in
prepare_sequences.py.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQ_CACHE_DIR, DEFAULT_STREAM, TABULAR_SOURCES, META_COLS, TARGET_COL

import torch
from torch.utils.data import Dataset


def _npz_path(case, split, stream=DEFAULT_STREAM):
    return os.path.join(SEQ_CACHE_DIR, stream, f"{case}_{split}.npz")


class SequenceDataset(Dataset):
    """Yields (x [T, 4] float32, mask [T] bool, y float32) for a (case, split)."""

    def __init__(self, case, split, stream=DEFAULT_STREAM):
        d = np.load(_npz_path(case, split, stream))
        self.X = torch.from_numpy(d["X"]).float()          # [N, T, 4]
        self.mask = torch.from_numpy(d["mask"]).bool()     # [N, T]
        self.y = torch.from_numpy(d["y"]).float()          # [N]
        self.keys = d["keys"]                              # [N, 3] (PID, Visit_no, Day)
        self.channels = list(d["channels"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.mask[i], self.y[i]

    def pos_weight(self):
        """pos_weight = n_neg / n_pos for weighted BCE."""
        n_pos = float(self.y.sum()); n_neg = float(len(self.y) - n_pos)
        return torch.tensor(n_neg / max(n_pos, 1.0))


def _load_tabular(branch, case, split):
    """Load engineered features for a tabular branch (clinical / machine / combined)."""
    import pandas as pd
    src = TABULAR_SOURCES[branch]
    if branch == "machine":
        path = os.path.join(src, case, "denoised", f"{split}.csv")
    else:
        path = os.path.join(src, case, f"{split}.csv")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


class TabularDataset(Dataset):
    """Engineered-feature tensors for the MLP rung. Fit scaler on train, pass to val/test."""

    def __init__(self, branch, case, split, scaler=None, imputer=None):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        df, feat_cols = _load_tabular(branch, case, split)
        X = df[feat_cols].to_numpy(dtype=np.float32)
        if imputer is None:
            imputer = SimpleImputer(strategy="median").fit(X)
        if scaler is None:
            scaler = StandardScaler().fit(imputer.transform(X))
        X = scaler.transform(imputer.transform(X)).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(df[TARGET_COL].to_numpy(dtype=np.float32))
        self.keys = df[["PID", "Visit_no", "Day"]].to_numpy()
        self.feat_cols = feat_cols
        self.scaler, self.imputer = scaler, imputer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class FusionDataset(Dataset):
    """Pairs each sequence with its clinical feature vector by (PID, Visit_no, Day) key.

    Sequence membership is authoritative (machine branch); clinical features are joined on.
    Missing clinical rows are zero-filled (post-scaling) and flagged so the model can learn
    a missing-clinical regime if needed.
    """

    def __init__(self, case, split, clinical_scaler=None, clinical_imputer=None,
                 stream=DEFAULT_STREAM):
        self.seq = SequenceDataset(case, split, stream)
        tab = TabularDataset("clinical", case, split,
                             scaler=clinical_scaler, imputer=clinical_imputer)
        self.clinical_scaler, self.clinical_imputer = tab.scaler, tab.imputer
        self.clinical_dim = tab.X.shape[1]

        key2row = {tuple(int(v) for v in k): i for i, k in enumerate(tab.keys)}
        C = torch.zeros(len(self.seq), self.clinical_dim)
        present = torch.zeros(len(self.seq), 1)
        for j, k in enumerate(self.seq.keys):
            row = key2row.get(tuple(int(v) for v in k))
            if row is not None:
                C[j] = tab.X[row]; present[j, 0] = 1.0
        self.C = torch.cat([C, present], dim=1)   # last col = clinical-present flag

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        x, mask, y = self.seq[i]
        return x, mask, self.C[i], y
