"""
models/cnn1d.py
===============
Rung 2 — 1D-CNN over the raw 4-channel pressure streams.

Captures local waveform morphology — the within-window dynamics that engineered stats
like skewness/range/iqr only crudely approximate. Input is [B, T, 4]; conv operates over
time with channels-first [B, 4, T]. Masked global pooling ignores padded timesteps.
"""

import torch
import torch.nn as nn


class CNN1DEncoder(nn.Module):
    """Stack of temporal conv blocks -> masked global pooling -> embedding [B, channels]."""

    def __init__(self, in_ch=4, channels=64, n_blocks=3, kernel=5, dropout=0.3):
        super().__init__()
        blocks, c = [], in_ch
        for _ in range(n_blocks):
            blocks += [
                nn.Conv1d(c, channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c = channels
        self.conv = nn.Sequential(*blocks)
        self.out_dim = channels

    def forward(self, x, mask):
        # x [B, T, 4] -> [B, 4, T]
        h = self.conv(x.transpose(1, 2))          # [B, channels, T]
        m = mask.unsqueeze(1).float()             # [B, 1, T]
        h = h * m
        pooled = h.sum(-1) / m.sum(-1).clamp(min=1.0)   # masked mean over time
        return pooled                             # [B, channels]


class CNN1DClassifier(nn.Module):
    def __init__(self, in_ch=4, channels=64, n_blocks=3, kernel=5, dropout=0.3):
        super().__init__()
        self.encoder = CNN1DEncoder(in_ch, channels, n_blocks, kernel, dropout)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x, mask):
        return self.head(self.encoder(x, mask)).squeeze(-1)
