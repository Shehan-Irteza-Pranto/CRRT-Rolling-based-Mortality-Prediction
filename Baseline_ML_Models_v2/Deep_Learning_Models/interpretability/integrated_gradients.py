"""
interpretability/integrated_gradients.py
========================================
Integrated Gradients / DeepLift attributions for the CNN and RNN rungs over the raw
pressure streams. The payoff is CONVERGING EVIDENCE: if DL temporal saliency independently
re-surfaces filter and return pressure as dominant (as classical SHAP did), the clinical
claim is much harder to dismiss than SHAP alone.

Uses captum (see ../requirements.txt). Attributions are [T, 4]; we summarize to a per-channel
importance (mean |attr| over time and samples) so it lines up directly with the SHAP per-
feature/channel story, and also keep the temporal profile for waveform-saliency figures.

Usage
-----
    python integrated_gradients.py --model cnn1d --case 1day --ckpt <path-to-state_dict>
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from config import CHANNELS, N_CHANNELS, SALIENCY_DIR, DEFAULT_STREAM

import torch


def _wrap_forward(model, mask):
    """captum needs forward(inputs)->scalar-per-sample; close over the fixed mask batch."""
    def f(x):
        return model(x, mask)
    return f


def attribute(model, dataset, device, max_samples=300, stream=DEFAULT_STREAM):
    from captum.attr import IntegratedGradients
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    per_channel, temporal, n = np.zeros(N_CHANNELS), None, 0
    for batch in loader:
        x, mask, _ = batch
        x = x.to(device).requires_grad_(True); mask = mask.to(device)
        ig = IntegratedGradients(_wrap_forward(model, mask))
        attr = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=32)  # [B, T, 4]
        a = attr.abs().detach().cpu().numpy()
        per_channel += a.mean(axis=(0, 1)) * a.shape[0]
        prof = a.mean(axis=2)            # [B, T]
        temporal = prof.sum(0) if temporal is None else temporal + prof.sum(0)
        n += a.shape[0]
        if n >= max_samples:
            break
    per_channel /= max(n, 1); temporal = temporal / max(n, 1)
    return dict(zip(CHANNELS, per_channel.tolist())), temporal


def plot_and_save(channel_imp, temporal, title, save_dir=SALIENCY_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    safe = title.replace(" ", "_").replace("/", "-")

    fig, ax = plt.subplots(figsize=(6, 4))
    chans = list(channel_imp); vals = [channel_imp[c] for c in chans]
    ax.barh(chans, vals, color="indianred")
    ax.set_xlabel("Mean |Integrated Gradient|")
    ax.set_title(f"Per-channel saliency — {title}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{safe}_channel_ig.png"), dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(temporal, color="steelblue")
    ax.set_xlabel("minute in window"); ax.set_ylabel("mean |IG|")
    ax.set_title(f"Temporal saliency — {title}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{safe}_temporal_ig.png"), dpi=120)
    plt.close()
    print(f"[ig] saliency plots -> {save_dir}")


def _parse():
    ap = argparse.ArgumentParser(description="Integrated Gradients for CNN/RNN rungs")
    ap.add_argument("--model", required=True, choices=["cnn1d", "rnn"])
    ap.add_argument("--case", default="1day")
    ap.add_argument("--ckpt", required=True, help="state_dict path saved by train.py")
    ap.add_argument("--hp", default="{}")
    return ap.parse_args()


if __name__ == "__main__":
    import json
    from sequence_dataset import SequenceDataset
    a = _parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = {k: v for k, v in json.loads(a.hp).items()
          if k not in ("lr", "batch_size", "weight_decay")}
    if a.model == "cnn1d":
        from cnn1d import CNN1DClassifier as M
    else:
        from rnn import RNNClassifier as M
    model = M(in_ch=N_CHANNELS, **hp).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    ds = SequenceDataset(a.case, "test")
    ch, temp = attribute(model, ds, device)
    print("per-channel importance:", ch)
    plot_and_save(ch, temp, f"{a.model}_{a.case}")
