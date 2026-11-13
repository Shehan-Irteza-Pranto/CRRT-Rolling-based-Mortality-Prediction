"""
models/dl_aggregated.py
=======================
Aggregated (multi-day) deep-learning model for the Abalation_study.

Three-branch late fusion (extends Deep_Learning_Models/models/fusion.py):

    current-day raw stream [T,4]  --seq encoder (cnn1d|rnn(bilstm)|transformer)--> e_seq
    prior-days machine summary    --MLP-->                                         e_mach
    clinical summary (optional)   --MLP-->                                         e_clin
                              concat([e_seq, e_mach, e_clin]) -> MLP head -> logit

Pipeline -> branches used:
  * machine_only  (aggregated) : seq + prior-days machine summary
  * combined      (aggregated) : seq + prior-days machine summary + clinical summary
  * machine_only  (per-day)    : seq only           (set both summary dims to 0)
  * combined      (per-day)    : seq + clinical summary
  * clinical_only              : use a plain MLP instead (no sequence) — see note.
  * "mlp" method               : plain MLP on the current-day 52 stats (no sequence).

The sequence encoders themselves are reused verbatim from the DL package via
`build_seq_encoder` (handles the transformer CLS-embedding wrapper).
"""

import os
import sys
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
from abl_config import DL_DIR                                       # noqa: E402

# DL package puts its models on sys.path and imports them by bare name.
sys.path.insert(0, os.path.join(DL_DIR, "models"))
from fusion import build_seq_encoder                               # noqa: E402  (cnn1d/rnn/transformer)


def _mlp_branch(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
        nn.ReLU(), nn.Dropout(dropout),
    )


def _mlp_encoder(in_dim, hidden, dropout):
    """Multi-layer MLP encoder (same recipe as the standalone clinical MLP) returning a
    representation. -> (module, out_dim). Used as the combined model's clinical branch."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


class AggFusionClassifier(nn.Module):
    """Current-day sequence encoder fused with optional machine-summary and clinical branches.

    For the v2 COMBINED model the clinical branch is a multi-layer MLP (`clinical_hidden`,
    the best clinical-only setup) run in PARALLEL to the sequence encoder, and the two
    representations are concatenated and fused — i.e. clinical-MLP + sequence late fusion.
    """

    def __init__(self, seq_encoder="cnn1d", machine_summary_dim=0, clinical_summary_dim=0,
                 fusion_dim=128, dropout=0.3, seq_kwargs=None, clinical_hidden=(256, 128)):
        super().__init__()
        self.seq_enc, seq_dim = build_seq_encoder(seq_encoder, **(seq_kwargs or {}))
        fused = seq_dim

        self.machine_branch = None
        if machine_summary_dim and machine_summary_dim > 0:
            self.machine_branch = _mlp_branch(machine_summary_dim, fusion_dim, dropout)
            fused += fusion_dim

        self.clinical_branch = None
        if clinical_summary_dim and clinical_summary_dim > 0:
            self.clinical_branch, clin_out = _mlp_encoder(clinical_summary_dim, clinical_hidden, dropout)
            fused += clin_out

        self.head = nn.Sequential(
            nn.Linear(fused, fusion_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x, mask, machine_summary=None, clinical_summary=None, drop_branch=None):
        parts = [self.seq_enc(x, mask)]                      # [B, seq_dim]
        if self.machine_branch is not None:
            m = self.machine_branch(machine_summary)
            if drop_branch == "machine":
                m = torch.zeros_like(m)
            parts.append(m)
        if self.clinical_branch is not None:
            c = self.clinical_branch(clinical_summary)
            if drop_branch == "clinical":
                c = torch.zeros_like(c)
            parts.append(c)
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def build_agg_model(pipeline, scope, seq_encoder="cnn1d",
                    machine_summary_dim=0, clinical_summary_dim=0,
                    fusion_dim=128, dropout=0.3, seq_kwargs=None, clinical_hidden=(256, 128)):
    """Construct the right branch configuration for a pipeline x scope.

    pipeline in {machine_only, combined}; scope in {per_day, agg}.
      * machine_only : sequence (+ prior-day machine summary when scope == agg)
      * combined     : clinical-MLP branch + sequence encoder, fused (v2 design). The
                       machine signal enters through the [T,7] sequence, so there is NO
                       separate machine-summary branch for combined.
    (clinical_only and the 'mlp' method use Deep_Learning_Models/models/mlp.py.)
    """
    if pipeline == "combined":
        mach_dim, clin_dim = 0, clinical_summary_dim          # machine via sequence; clinical via MLP
    else:
        mach_dim, clin_dim = (machine_summary_dim if scope == "agg" else 0), 0
    return AggFusionClassifier(seq_encoder=seq_encoder,
                               machine_summary_dim=mach_dim,
                               clinical_summary_dim=clin_dim,
                               fusion_dim=fusion_dim, dropout=dropout, seq_kwargs=seq_kwargs,
                               clinical_hidden=clinical_hidden)


# --- smoke test: forward pass for every encoder x branch combination ---
if __name__ == "__main__":
    B, T, C = 8, 120, 4
    MACH, CLIN = 208, 116
    x = torch.randn(B, T, C)
    mask = torch.ones(B, T, dtype=torch.bool); mask[:, 90:] = False   # variable lengths
    ms = torch.randn(B, MACH)
    cs = torch.randn(B, CLIN)

    for enc in ["cnn1d", "rnn", "transformer"]:
        # combined-agg: all three branches
        model = build_agg_model("combined", "agg", seq_encoder=enc,
                                machine_summary_dim=MACH, clinical_summary_dim=CLIN)
        out = model(x, mask, machine_summary=ms, clinical_summary=cs)
        assert out.shape == (B,), (enc, out.shape)

        # machine-only agg: seq + machine summary
        m2 = build_agg_model("machine_only", "agg", seq_encoder=enc, machine_summary_dim=MACH)
        out2 = m2(x, mask, machine_summary=ms)
        assert out2.shape == (B,)

        # machine-only per-day: seq only
        m3 = build_agg_model("machine_only", "per_day", seq_encoder=enc)
        out3 = m3(x, mask)
        assert out3.shape == (B,)

        # one backward step to confirm the graph is trainable
        loss = nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(B))
        loss.backward()
        print(f"[ok] {enc}: combined-agg {tuple(out.shape)}, mach-agg {tuple(out2.shape)}, "
              f"per-day {tuple(out3.shape)}, backward OK")
    print("All encoder x branch combinations forward+backward cleanly.")
