"""
training/losses.py
==================
Class-imbalance losses. Train on the FULL imbalanced set (as the classical pipeline does)
and let the loss handle skew — stated explicitly in the paper. Early-stop on validation AUROC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    """BCEWithLogits with pos_weight = n_neg / n_pos (computed from the train split)."""

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float))

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017). gamma focuses on hard examples; alpha balances."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


def build_loss(name, pos_weight=None, gamma=2.0):
    if name == "weighted_bce":
        return WeightedBCE(pos_weight if pos_weight is not None else 1.0)
    if name == "focal":
        return FocalLoss(gamma=gamma)
    raise ValueError(f"unknown loss: {name}")
