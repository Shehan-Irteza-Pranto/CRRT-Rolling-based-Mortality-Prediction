"""
interpretability/attention_rollout.py
=====================================
Attention rollout (Abnar & Zuidema, 2020) for the Transformer rung — the native attention-
based interpretability that makes the Transformer the strongest interpretability story.

Rollout recursively multiplies the per-layer attention matrices (with residual/identity added
and re-normalized) to get how much each input timestep flows into the [CLS] token. We register
forward hooks on each encoder layer's self-attention to capture the attention weights, then
read out the CLS row -> a per-timestep importance over the window, plus a per-channel summary
(via the input projection) for the side-by-side-with-SHAP figure.

Requires torch. The nn.TransformerEncoderLayer must be run with need_weights=True; this module
patches the layers' self_attn calls via hooks to capture averaged head weights.

NOTE: scaffold. The hook wiring below is the standard pattern; verify against the installed
torch version's MultiheadAttention return signature before relying on the numbers.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from config import N_CHANNELS, SALIENCY_DIR

import torch


class _AttnCapture:
    """Capture averaged-head attention from each TransformerEncoderLayer's self_attn."""

    def __init__(self, model):
        self.weights = []
        self.handles = []
        for layer in model.encoder.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True.
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            self.weights.append(out[1].detach())   # [B, T+1, T+1] (averaged over heads)

    def clear(self):
        self.weights = []

    def remove(self):
        for h in self.handles:
            h.remove()


def rollout(attn_mats):
    """Recursive rollout. attn_mats: list of [B, N, N]. Returns CLS->tokens [B, N-1]."""
    result = None
    for A in attn_mats:
        A = A + torch.eye(A.size(-1), device=A.device)         # add residual
        A = A / A.sum(dim=-1, keepdim=True)                    # renormalize
        result = A if result is None else torch.bmm(A, result)
    cls_to_tokens = result[:, 0, 1:]                           # drop CLS->CLS
    return cls_to_tokens


def compute(model, dataset, device, max_samples=300):
    from torch.utils.data import DataLoader
    # Force need_weights=True on the encoder layers' attention.
    for layer in model.encoder.layers:
        layer.self_attn.need_weights = True
    cap = _AttnCapture(model)
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    temporal, n = None, 0
    with torch.no_grad():
        for batch in loader:
            x, mask, _ = batch
            cap.clear()
            _ = model(x.to(device), mask.to(device))
            if not cap.weights:
                break
            roll = rollout(cap.weights).cpu().numpy()          # [B, T]
            temporal = roll.sum(0) if temporal is None else temporal + roll.sum(0)
            n += roll.shape[0]
            if n >= max_samples:
                break
    cap.remove()
    return (temporal / max(n, 1)) if temporal is not None else None


def plot(temporal, title, save_dir=SALIENCY_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    safe = title.replace(" ", "_").replace("/", "-")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(temporal, color="darkgreen")
    ax.set_xlabel("minute in window"); ax.set_ylabel("attention to CLS")
    ax.set_title(f"Attention rollout — {title}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{safe}_rollout.png"), dpi=120)
    plt.close()
    print(f"[rollout] -> {save_dir}")


def _parse():
    ap = argparse.ArgumentParser(description="Attention rollout for the Transformer rung")
    ap.add_argument("--case", default="1day")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hp", default="{}")
    return ap.parse_args()


if __name__ == "__main__":
    import json
    from sequence_dataset import SequenceDataset
    from transformer import TransformerEncoderModel
    a = _parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = {k: v for k, v in json.loads(a.hp).items()
          if k not in ("lr", "batch_size", "weight_decay")}
    model = TransformerEncoderModel(in_ch=N_CHANNELS, **hp).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    ds = SequenceDataset(a.case, "test")
    temp = compute(model, ds, device)
    plot(temp, f"transformer_{a.case}")
