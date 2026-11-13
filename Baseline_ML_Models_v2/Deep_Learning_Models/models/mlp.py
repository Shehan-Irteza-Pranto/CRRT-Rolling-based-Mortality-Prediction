"""
models/mlp.py
=============
Rung 1 — Perceptron / MLP on the engineered feature vectors (clinical / machine / combined).

This is the FAIR head-to-head against the GBMs and satisfies the professor's "perceptron".
Expect it to roughly match or trail tuned gradient boosting — report that honestly. It is a
baseline rung, not the headline. The headline is the raw-stream sequence models.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))   # single logit (BCEWithLogits)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B]

    def embed(self, x):
        """Penultimate representation (used as the clinical branch inside fusion)."""
        for m in self.net[:-1]:
            x = m(x)
        return x
