"""
models/fusion.py
================
Rung 5 — the model that ties the study together.

Sequence encoder (machine, raw streams) + MLP branch (clinical, engineered features),
late-fused before the classifier. This is where we test whether DL recovers — or beats —
the classical finding that machine pressure data contributes disproportionate, complementary
mortality signal. The number to beat/characterize: GB AUROC 0.801 on aggregated combined.

Late fusion (concat of the two branch embeddings) is the default; it keeps each branch's
inductive bias intact and makes per-branch ablation trivial (zero out one embedding).
"""

import torch
import torch.nn as nn

from cnn1d import CNN1DEncoder
from rnn import RNNEncoder
from transformer import TransformerEncoderModel


def build_seq_encoder(name, **kw):
    if name == "cnn1d":
        enc = CNN1DEncoder(**kw); return enc, enc.out_dim
    if name == "rnn":
        enc = RNNEncoder(**kw); return enc, enc.out_dim
    if name == "transformer":
        enc = TransformerEncoderModel(**kw)
        # use .encode() for the embedding; wrap so the call signature matches the others
        wrapper = _TransformerEncodeWrapper(enc); return wrapper, enc.out_dim
    raise ValueError(f"unknown seq encoder: {name}")


class _TransformerEncodeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask):
        return self.model.encode(x, mask)


class FusionClassifier(nn.Module):
    def __init__(self, seq_encoder="cnn1d", clinical_dim=64, fusion_dim=128,
                 dropout=0.3, seq_kwargs=None):
        super().__init__()
        seq_kwargs = seq_kwargs or {}
        self.seq_enc, seq_dim = build_seq_encoder(seq_encoder, **seq_kwargs)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, fusion_dim), nn.BatchNorm1d(fusion_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(seq_dim + fusion_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x, mask, clinical, drop_branch=None):
        """drop_branch in {None, 'seq', 'clinical'} for ablation (zero a branch's embedding)."""
        s = self.seq_enc(x, mask)
        c = self.clinical_branch(clinical)
        if drop_branch == "seq":
            s = torch.zeros_like(s)
        elif drop_branch == "clinical":
            c = torch.zeros_like(c)
        return self.head(torch.cat([s, c], dim=1)).squeeze(-1)
