"""
model.py — attention-pooling CNN1D (exp_3 Phase 2)
==================================================
Reuse the parent CNN1DEncoder conv stack, but replace masked global-mean pooling with masked
ATTENTION pooling so the classifier can up-weight the windows that matter (e.g. end-of-day
deterioration). Falls back to masked-mean when ATTENTION=False.
"""
import os
import sys
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dlcfg import (reduced_in_ch, CNN_CHANNELS, CNN_BLOCKS, CNN_KERNEL,        # noqa: E402
                   CNN_DROPOUT, ATTENTION, DL_DIR)
sys.path.insert(0, os.path.join(DL_DIR, "models"))
from cnn1d import CNN1DEncoder                                                 # noqa: E402


class AttnCNN1D(nn.Module):
    def __init__(self, in_ch, channels=CNN_CHANNELS, n_blocks=CNN_BLOCKS,
                 kernel=CNN_KERNEL, dropout=CNN_DROPOUT, attention=ATTENTION):
        super().__init__()
        self.enc = CNN1DEncoder(in_ch, channels, n_blocks, kernel, dropout)   # use .conv only
        self.attention = attention
        if attention:
            self.attn = nn.Sequential(nn.Linear(channels, channels // 2), nn.Tanh(),
                                      nn.Linear(channels // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def forward(self, x, mask):
        h = self.enc.conv(x.transpose(1, 2))                  # [B,C,T]
        if self.attention:
            ht = h.transpose(1, 2)                            # [B,T,C]
            score = self.attn(ht).squeeze(-1)                 # [B,T]
            score = score.masked_fill(~mask, -1e9)            # finite floor (avoids all-(-inf)->NaN)
            a = torch.softmax(score, dim=1).unsqueeze(1)      # [B,1,T]
            pooled = (h * a).sum(-1)                          # [B,C]
        else:
            m = mask.unsqueeze(1).float()
            pooled = (h * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        return self.head(self.drop(pooled)).squeeze(-1)


def build_model():
    return AttnCNN1D(in_ch=reduced_in_ch())


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(4, 144, reduced_in_ch()); mask = torch.ones(4, 144, dtype=torch.bool)
    mask[:, 100:] = False
    out = m(x, mask)
    print("logit", out.shape, "params", sum(p.numel() for p in m.parameters()))
