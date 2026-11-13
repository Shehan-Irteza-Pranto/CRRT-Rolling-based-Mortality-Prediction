"""
models/moment_head.py
=====================
Rung 4b (OPTIONAL) — frozen MOMENT embedding + linear/MLP head.

A clean drop-in that nods at the waveform/MOMENT dissertation line without overloading the
paper. Keep it optional given the timeline (see PAPER_PLAN.md open decisions). MOMENT expects
univariate series; the 4 channels are embedded independently and concatenated, then a small
trainable head classifies. The backbone stays frozen — only the head trains — so the fair-
tuning-budget argument is trivially satisfied.

STATUS: stub. Requires `pip install momentfm` and wiring the actual backbone. Left as a TODO
so the rest of the pipeline does not depend on the heavyweight download.
"""

import torch
import torch.nn as nn


class MomentHead(nn.Module):
    def __init__(self, in_ch=4, embed_dim=1024, hidden=256, dropout=0.3, backbone=None):
        super().__init__()
        # backbone: a frozen MOMENT model exposing .embed(series)->[B, embed_dim].
        # TODO: from momentfm import MOMENTPipeline; load + freeze; set self.backbone.
        self.backbone = backbone
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim * in_ch, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @torch.no_grad()
    def _embed_channels(self, x, mask):
        # x [B, T, C] -> per-channel MOMENT embedding -> concat [B, C*embed_dim]
        if self.backbone is None:
            raise NotImplementedError(
                "MOMENT backbone not wired. pip install momentfm, load + freeze, "
                "then set self.backbone (see PAPER_PLAN.md rung 4b).")
        embs = [self.backbone.embed(x[:, :, c]) for c in range(self.in_ch)]
        return torch.cat(embs, dim=1)

    def forward(self, x, mask):
        return self.head(self._embed_channels(x, mask)).squeeze(-1)
