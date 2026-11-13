"""
training/train.py
=================
Multi-seed training harness. Enforces the rigor protocol centrally so no rung can drift:

  * trains on the FULL imbalanced set with weighted-BCE / focal loss
  * early-stops / selects on VALIDATION AUROC
  * derives the F1 decision threshold on validation (identical to classical)
  * runs >= 5 seeds and writes one result row per (model, case, seed)
  * evaluates on FULL and BALANCED 50/50 test sets with the shared metric protocol

Final aggregation (mean +/- std, bootstrap CIs, tables) is done by evaluate.py.

Usage
-----
    python train.py --model cnn1d  --cases 1day 2day --seeds 5
    python train.py --model mlp    --branch combined --cases 1day 2day
    python train.py --model fusion --cases 1day 2day --seed-list 0 1 2 3 4

NOTE: requires torch (not yet installed here). prepare_sequences.py must have been run first
for the sequence/fusion models.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))

from config import (
    CASES, SEEDS, LOSS, FOCAL_GAMMA, EARLY_STOP_METRIC, EARLY_STOP_PATIENCE,
    MAX_EPOCHS, RESULTS_DIR, DEFAULT_STREAM, N_CHANNELS,
    MODELS_DIR, HISTORY_DIR, PRED_DIR,
    LR, WEIGHT_DECAY, GRAD_CLIP, USE_LR_SCHEDULER,
    LR_SCHED_FACTOR, LR_SCHED_PATIENCE, MIN_LR,
)
from metrics import (
    find_optimal_threshold, compute_metrics, create_balanced_test,
)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from losses import build_loss

SEQ_MODELS = {"cnn1d", "rnn", "transformer", "moment"}
TABULAR_MODELS = {"mlp"}


def make_tag(model_name, label_branch):
    """Stable file tag per rung: mlp_<branch>, fusion_<encoder>, else the model name.
    `label_branch` is the report branch (= seq_encoder for fusion, branch for the rest)."""
    if model_name == "mlp":
        return f"mlp_{label_branch}"
    if model_name == "fusion":
        return f"fusion_{label_branch}"
    return model_name


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Model construction ---

def build_model(name, in_dim=None, **hp):
    if name == "cnn1d":
        from cnn1d import CNN1DClassifier
        return CNN1DClassifier(in_ch=N_CHANNELS, **hp)
    if name == "rnn":
        from rnn import RNNClassifier
        return RNNClassifier(in_ch=N_CHANNELS, **hp)
    if name == "transformer":
        from transformer import TransformerEncoderModel
        return TransformerEncoderModel(in_ch=N_CHANNELS, **hp)
    if name == "mlp":
        from mlp import MLP
        return MLP(in_dim=in_dim, **hp)
    if name == "fusion":
        from fusion import FusionClassifier
        return FusionClassifier(**hp)
    raise ValueError(f"unknown model: {name}")


# --- Datasets ---

def make_loaders(model, case, branch, batch_size, stream=DEFAULT_STREAM):
    """Return (train_loader, val_set, test_set, extras) for the given model family."""
    if model in SEQ_MODELS:
        from sequence_dataset import SequenceDataset
        tr = SequenceDataset(case, "train", stream)
        va = SequenceDataset(case, "val", stream)
        te = SequenceDataset(case, "test", stream)
        extras = {"pos_weight": tr.pos_weight(), "in_dim": None, "kind": "seq"}
    elif model == "fusion":
        from sequence_dataset import FusionDataset
        tr = FusionDataset(case, "train", stream=stream)
        va = FusionDataset(case, "val", clinical_scaler=tr.clinical_scaler,
                           clinical_imputer=tr.clinical_imputer, stream=stream)
        te = FusionDataset(case, "test", clinical_scaler=tr.clinical_scaler,
                           clinical_imputer=tr.clinical_imputer, stream=stream)
        pw = tr.seq.pos_weight()
        extras = {"pos_weight": pw, "clinical_dim": tr.C.shape[1], "kind": "fusion"}
    else:  # tabular MLP
        from sequence_dataset import TabularDataset
        tr = TabularDataset(branch, case, "train")
        va = TabularDataset(branch, case, "val", scaler=tr.scaler, imputer=tr.imputer)
        te = TabularDataset(branch, case, "test", scaler=tr.scaler, imputer=tr.imputer)
        n_pos = float(tr.y.sum()); n_neg = float(len(tr.y) - n_pos)
        extras = {"pos_weight": torch.tensor(n_neg / max(n_pos, 1.0)),
                  "in_dim": tr.X.shape[1], "kind": "tabular"}
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, va, te, extras


# --- Forward helpers (unify call signatures across model families) ---

def _forward(model, kind, batch, device):
    if kind == "seq":
        x, mask, y = batch
        return model(x.to(device), mask.to(device)), y.to(device)
    if kind == "fusion":
        x, mask, c, y = batch
        return model(x.to(device), mask.to(device), c.to(device)), y.to(device)
    x, y = batch
    return model(x.to(device)), y.to(device)


@torch.no_grad()
def _predict(model, kind, dataset, device, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, ys = [], []
    for batch in loader:
        logits, y = _forward(model, kind, batch, device)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


@torch.no_grad()
def _evaluate_loss(model, loss_fn, kind, dataset, device, batch_size=128):
    """Like _predict but also returns the mean (sample-weighted) validation loss,
    so the per-epoch training curve can be logged alongside val AUROC."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, ys, tot_loss, n = [], [], 0.0, 0
    for batch in loader:
        logits, y = _forward(model, kind, batch, device)
        bs = len(y)
        tot_loss += float(loss_fn(logits, y)) * bs          # loss_fn is mean-reduced
        n += bs
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.cpu().numpy())
    return tot_loss / max(n, 1), np.concatenate(probs), np.concatenate(ys)


# --- One training run (single seed) ---

def train_one(model_name, case, branch, hp, seed, device, stream=DEFAULT_STREAM,
              save_artifacts=False):
    set_seed(seed)
    loader, va, te, extras = make_loaders(model_name, case, branch,
                                          hp.get("batch_size", 32), stream)
    kind = extras["kind"]

    build_kw = {k: v for k, v in hp.items()
                if k not in ("lr", "batch_size", "weight_decay", "grad_clip")}
    if kind == "tabular":
        build_kw["in_dim"] = extras["in_dim"]
    if kind == "fusion":
        build_kw["clinical_dim"] = extras["clinical_dim"]
    model = build_model(model_name, **build_kw).to(device)

    loss_fn = build_loss(LOSS, pos_weight=extras["pos_weight"], gamma=FOCAL_GAMMA).to(device)
    lr = hp.get("lr", LR)
    weight_decay = hp.get("weight_decay", WEIGHT_DECAY)
    grad_clip = hp.get("grad_clip", GRAD_CLIP)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Adaptive LR: drop LR when val AUROC plateaus, so the late-epoch overfitting
    # regime is trained at a much smaller step. Selection is still best-val + early stop.
    sched = None
    if USE_LR_SCHEDULER:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=LR_SCHED_FACTOR,
            patience=LR_SCHED_PATIENCE, min_lr=MIN_LR)

    best_val, best_state, best_epoch, patience = -1.0, None, 0, 0
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        run_loss, n_batches = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            logits, y = _forward(model, kind, batch, device)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run_loss += float(loss.detach()); n_batches += 1
        train_loss = run_loss / max(n_batches, 1)

        val_loss, val_prob, val_y = _evaluate_loss(model, loss_fn, kind, va, device)
        val_auroc = roc_auc_score(val_y, val_prob) if len(np.unique(val_y)) > 1 else 0.0
        cur_lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step(val_auroc)

        improved = val_auroc > best_val
        if improved:
            best_val = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1

        history.append({
            "model": model_name, "case": case, "branch": branch, "seed": seed,
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auroc": round(float(val_auroc), 6),
            "lr": cur_lr,
            "best_so_far": int(improved),
        })
        print(f"   [{model_name} {case} seed{seed}] epoch {epoch + 1:3d}/{MAX_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auroc={val_auroc:.4f}  lr={cur_lr:.1e}  "
              f"patience={patience}/{EARLY_STOP_PATIENCE}"
              f"{'  *' if improved else ''}")

        if patience >= EARLY_STOP_PATIENCE:
            print(f"   early stop at epoch {epoch + 1} (best val AUROC {best_val:.4f} "
                  f"@ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Validation-derived F1 threshold (identical to classical protocol)
    val_prob, val_y = _predict(model, kind, va, device)
    thr = find_optimal_threshold(val_y, val_prob)

    # Evaluate FULL test and BALANCED 50/50 test
    te_prob, te_y = _predict(model, kind, te, device)
    # For fusion, record which sequence encoder was used in the `branch` field so
    # the variants (fusion/cnn1d, fusion/rnn, fusion/transformer) stay distinct
    # rows in evaluate.py and the comparison report.
    label_branch = hp.get("seq_encoder", "cnn1d") if model_name == "fusion" else branch
    for h in history:                       # keep history branch label consistent with the report
        h["branch"] = label_branch
    out = {"model": model_name, "case": case, "branch": label_branch, "seed": seed,
           "best_val_auroc": float(best_val), "best_epoch": best_epoch,
           "epochs_trained": len(history), "threshold": float(thr)}
    for thresh, prefix in [(0.5, "full_test"), (thr, "full_test_opt")]:
        for k, v in compute_metrics(te_y, (te_prob >= thresh).astype(int), te_prob).items():
            out[f"{prefix}_{k}"] = v
    bal_idx = create_balanced_test(None, te_y)
    by, bp = te_y[bal_idx], te_prob[bal_idx]
    for thresh, prefix in [(0.5, "balanced_test"), (thr, "balanced_test_opt")]:
        for k, v in compute_metrics(by, (bp >= thresh).astype(int), bp).items():
            out[f"{prefix}_{k}"] = v

    if save_artifacts:
        tag = make_tag(model_name, label_branch)
        # 1) Best checkpoint (val-AUROC-selected) + everything needed to rebuild it
        #    later for validation: build_kw feeds build_model(model_name, **build_kw).
        os.makedirs(MODELS_DIR, exist_ok=True)
        ckpt_path = os.path.join(MODELS_DIR, f"{tag}_{case}_seed{seed}.pt")
        torch.save({
            "model_name": model_name, "build_kw": build_kw, "kind": kind,
            "tag": tag, "case": case, "branch": label_branch, "seed": seed,
            "threshold": float(thr), "best_val_auroc": float(best_val),
            "best_epoch": best_epoch, "stream": stream,
            "state_dict": model.state_dict(),
        }, ckpt_path)
        # 2) Test predictions for the ROC / precision-recall curves (full + balanced)
        os.makedirs(PRED_DIR, exist_ok=True)
        np.savez(os.path.join(PRED_DIR, f"{tag}_{case}_seed{seed}.npz"),
                 y_true=te_y, y_score=te_prob, bal_idx=bal_idx,
                 threshold=np.float64(thr), tag=tag, case=case, seed=seed)

    return out, te_prob, te_y, history


# --- Multi-seed driver ---

def run(model_name, cases, branch, seeds, hp, stream=DEFAULT_STREAM):
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    rows, hist_rows = [], []
    for case in cases:
        for seed in seeds:
            print(f"[train] {model_name} case={case} branch={branch} seed={seed} dev={device}")
            res, _, _, hist = train_one(model_name, case, branch, hp, seed, device, stream,
                                        save_artifacts=True)
            print(f"   full AUROC={res['full_test_AUROC']:.4f} "
                  f"bal AUROC={res['balanced_test_AUROC']:.4f} val={res['best_val_auroc']:.4f} "
                  f"(best epoch {res['best_epoch']}/{res['epochs_trained']})")
            rows.append(res)
            hist_rows.extend(hist)
    df = pd.DataFrame(rows)
    # fusion's label branch is the seq encoder; mlp's is the feature branch.
    label_branch = hp.get("seq_encoder", "cnn1d") if model_name == "fusion" else branch
    tag = make_tag(model_name, label_branch)

    # Per-epoch training curve (train/val loss + val AUROC) for every (case, seed).
    hist_path = os.path.join(HISTORY_DIR, f"training_history_{tag}.csv")
    pd.DataFrame(hist_rows).to_csv(hist_path, index=False)
    print(f"[train] wrote {hist_path}  ({len(hist_rows)} epoch rows)")

    path = os.path.join(RESULTS_DIR, f"raw_results_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"[train] wrote {path}")
    return df


def _parse():
    ap = argparse.ArgumentParser(description="Train a DL rung over >=5 seeds")
    ap.add_argument("--model", required=True,
                    choices=["mlp", "cnn1d", "rnn", "transformer", "fusion"])
    ap.add_argument("--branch", default="machine",
                    choices=["clinical", "machine", "combined"],
                    help="tabular feature source for the MLP rung")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--seeds", type=int, default=None, help="use the first N config SEEDS")
    ap.add_argument("--seed-list", nargs="+", type=int, default=None)
    ap.add_argument("--stream", default=DEFAULT_STREAM)
    ap.add_argument("--hp", default="{}", help="JSON of hyperparameters (else config defaults)")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    seeds = a.seed_list if a.seed_list else (SEEDS[: a.seeds] if a.seeds else SEEDS)
    run(a.model, a.cases, a.branch, seeds, json.loads(a.hp), a.stream)
