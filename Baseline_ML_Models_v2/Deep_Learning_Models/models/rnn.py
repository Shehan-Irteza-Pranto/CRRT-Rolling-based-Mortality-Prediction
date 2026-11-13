"""
models/rnn.py
=============
Rung 3 — BiLSTM / GRU over the raw streams.

Captures longer temporal dependencies across the window. Bidirectional because we classify
a *completed* circuit-day window (not online forecasting), so the model may use the whole
sequence. Padded timesteps are excluded via pack_padded_sequence on the mask lengths.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, in_ch=4, hidden=128, layers=1, cell="lstm", dropout=0.3):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_ch, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hidden * 2

    def forward(self, x, mask):
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        # masked mean over valid timesteps
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return pooled                              # [B, 2*hidden]


class RNNClassifier(nn.Module):
    def __init__(self, in_ch=4, hidden=128, layers=1, cell="lstm", dropout=0.3):
        super().__init__()
        self.encoder = RNNEncoder(in_ch, hidden, layers, cell, dropout)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x, mask):
        return self.head(self.encoder(x, mask)).squeeze(-1)
