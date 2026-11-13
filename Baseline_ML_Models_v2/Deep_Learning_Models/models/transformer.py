"""
models/transformer.py
=====================
Rung 4 — Transformer encoder over the raw streams. The strongest sequence rung and the
source of native attention-based interpretability (see interpretability/attention_rollout.py).

Design:
  - linear input projection 4 -> d_model
  - sinusoidal positional encoding
  - a learnable [CLS] token; its final state is the pooled window representation
  - key_padding_mask excludes padded timesteps from attention

For long windows (T up to ~1440) full self-attention is O(T^2); if memory-bound, downsample
the stream (e.g. mean-pool every k minutes) in prepare_sequences.py, or swap in a
linear-attention layer. Documented as an open decision in PAPER_PLAN.md.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderModel(nn.Module):
    def __init__(self, in_ch=4, d_model=128, n_heads=8, n_layers=4, ff=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = d_model
        self.head = nn.Linear(d_model, 1)

    def _encode(self, x, mask):
        B = x.size(0)
        h = self.proj(x)                                   # [B, T, d]
        cls = self.cls.expand(B, -1, -1)                   # [B, 1, d]
        h = torch.cat([cls, h], dim=1)                     # prepend CLS
        h = self.pe(h)
        # key_padding_mask: True = ignore. CLS is always attended (False).
        pad = ~mask                                        # [B, T] True where padded
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        h = self.encoder(h, src_key_padding_mask=kpm)
        return h                                           # [B, T+1, d]

    def encode(self, x, mask):
        """Pooled [CLS] embedding [B, d] — used by the fusion rung."""
        return self._encode(x, mask)[:, 0]

    def forward(self, x, mask):
        return self.head(self.encode(x, mask)).squeeze(-1)
