"""
training/search.py
==================
Fair, documented hyperparameter search — comparable to the GBMs' 30-iter RandomizedSearch
x 5-fold budget, so a reviewer cannot call the comparison unfair.

Strategy: random search over config.SEARCH_SPACE, N_SEARCH_ITER samples per model. Each
sampled config is trained on ONE seed and scored by validation AUROC (model selection signal,
identical to the early-stop metric). The winning config is then re-run by train.py across all
>= 5 seeds for the reported mean +/- std. This mirrors RandomizedSearchCV: search on a cheap
signal, report on the full protocol.

Usage
-----
    python search.py --model cnn1d --case 1day            # find best HP on val AUROC
    python search.py --model transformer --case 2day --n-iter 30

Writes results/search_{model}_{case}.json with the ranked configs and the winner.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEARCH_SPACE, N_SEARCH_ITER, RANDOM_STATE, RESULTS_DIR, CASES

import train as T   # reuse the harness (train_one returns best_val_auroc)
import torch


def sample_config(model, rng):
    """Draw one HP config: shared knobs + model-specific width/depth."""
    cfg = {
        "lr":           rng.choice(SEARCH_SPACE["lr"]),
        "batch_size":   int(rng.choice(SEARCH_SPACE["batch_size"])),
        "dropout":      float(rng.choice(SEARCH_SPACE["dropout"])),
        "weight_decay": float(rng.choice(SEARCH_SPACE["weight_decay"])),
    }
    space = SEARCH_SPACE.get(model, {})
    for k, choices in space.items():
        v = choices[rng.randint(len(choices))]
        cfg[k] = list(v) if isinstance(v, (list, tuple)) else (
            int(v) if isinstance(v, (int, np.integer)) else v)
    cfg["lr"] = float(cfg["lr"])
    return cfg


def search(model, case, branch, n_iter, stream, seq_encoder=None):
    rng = np.random.RandomState(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = []
    for i in range(n_iter):
        hp = sample_config(model, rng)
        if seq_encoder is not None:               # search one fusion encoder at a time
            hp["seq_encoder"] = seq_encoder
        try:
            res, _, _, _ = T.train_one(model, case, branch, hp, seed=RANDOM_STATE,
                                       device=device, stream=stream)
            score = res["best_val_auroc"]
        except Exception as e:
            score = float("nan"); res = {"error": str(e)}
        trials.append({"hp": hp, "val_auroc": score})
        print(f"  [{i+1}/{n_iter}] val_auroc={score:.4f}  hp={hp}")

    trials = [t for t in trials if t["val_auroc"] == t["val_auroc"]]
    trials.sort(key=lambda t: t["val_auroc"], reverse=True)
    out = {"model": model, "case": case, "branch": branch, "seq_encoder": seq_encoder,
           "n_iter": n_iter, "trials": trials,
           "best": trials[0] if trials else None}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Name the file with the SAME tag training uses, so run_all --use-search finds it:
    # mlp_<branch>, fusion_<encoder>, else the model name.
    label = seq_encoder if model == "fusion" else branch
    tag = T.make_tag(model, label)
    path = os.path.join(RESULTS_DIR, f"search_{tag}_{case}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[search] best val_auroc={out['best']['val_auroc']:.4f} -> {path}")
    return out


def _parse():
    ap = argparse.ArgumentParser(description="Fair HP search (matched to GBM budget)")
    ap.add_argument("--model", required=True,
                    choices=["mlp", "cnn1d", "rnn", "transformer", "fusion"])
    ap.add_argument("--case", default="1day", choices=CASES)
    ap.add_argument("--branch", default="machine")
    ap.add_argument("--seq-encoder", default=None,
                    choices=["cnn1d", "rnn", "transformer"],
                    help="for --model fusion: fix the sequence encoder (search it alone)")
    ap.add_argument("--n-iter", type=int, default=N_SEARCH_ITER)
    ap.add_argument("--stream", default="denoised")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    search(a.model, a.case, a.branch, a.n_iter, a.stream, seq_encoder=a.seq_encoder)
