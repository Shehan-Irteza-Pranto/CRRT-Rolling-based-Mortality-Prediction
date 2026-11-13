"""
models/dl_train.py
==================
Train ONE aggregated-DL config and evaluate it with the resampled protocol.

Reuses the DL rigor protocol (weighted-BCE/focal loss, early-stop on val AUROC,
val-derived F1 threshold) but for the 3-input batch of AggDataset. Runs on the
single GPU exposed by CUDA_VISIBLE_DEVICES (the launcher pins one GPU per process).

CLI (used by run_dl_benchmark.py per job):
    CUDA_VISIBLE_DEVICES=0 python models/dl_train.py \
        --pipeline machine_only --scope per_day --encoder cnn1d --case 1day --seed 0 \
        --out results/dl_rows/machine_only_perday_cnn1d_1day_s0.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABL  = os.path.dirname(_HERE)
sys.path.insert(0, _ABL)
sys.path.insert(0, os.path.join(_ABL, "eval"))
sys.path.insert(0, _HERE)
from abl_config import (DL_DIR, DL_REDUCE, DL_REDUCE_ENABLED, RESULTS_DIR,   # noqa: E402
                        DL_BALANCED_SAMPLER, KEY_COLS)
from subpop_eval import eval_dispatch                              # noqa: E402
from dl_aggregated import build_agg_model                          # noqa: E402
from dl_data import AggDataset, pos_weight                         # noqa: E402

# DL package config + losses (this process imports only the DL `config`).
sys.path.insert(0, DL_DIR)
sys.path.insert(0, os.path.join(DL_DIR, "common"))
sys.path.insert(0, os.path.join(DL_DIR, "training"))
from config import (MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, WEIGHT_DECAY,   # noqa: E402
                    GRAD_CLIP, LOSS, FOCAL_GAMMA,
                    USE_LR_SCHEDULER, LR_SCHED_FACTOR, LR_SCHED_PATIENCE, MIN_LR)
from metrics import find_optimal_threshold                              # noqa: E402
from losses import build_loss                                          # noqa: E402
from dl_search import sample_hp                                        # noqa: E402

import torch                                                          # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler      # noqa: E402
from sklearn.metrics import roc_auc_score                           # noqa: E402

# fixed-HP fallback (search_iter == 0) — the prior defaults
# clinical_hidden = the best clinical-only MLP setup, reused as the combined model's
# clinical branch (fused in parallel with the sequence encoder).
DEFAULT_HP = {"lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": 0.3,
              "batch_size": 32, "fusion_dim": 128, "seq_kwargs": None,
              "clinical_hidden": (256, 128)}


def _set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _batch_forward(model, batch, device, mach_dim, clin_dim, drop_branch=None):
    x, mask, m, c, y = batch
    kw = {}
    if mach_dim:
        kw["machine_summary"] = m.to(device)
    if clin_dim:
        kw["clinical_summary"] = c.to(device)
    logits = model(x.to(device), mask.to(device), drop_branch=drop_branch, **kw)
    return logits, y.to(device)


@torch.no_grad()
def _predict(model, ds, device, mach_dim, clin_dim, bs=128):
    model.eval()
    probs, ys = [], []
    for batch in DataLoader(ds, batch_size=bs):
        logits, y = _batch_forward(model, batch, device, mach_dim, clin_dim)
        probs.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


@torch.no_grad()
def _eval_epoch(model, ds, loss_fn, device, mach_dim, clin_dim, bs=128):
    """One val pass -> (probs, ys, mean_loss) so AUROC + loss come from one forward sweep."""
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    for batch in DataLoader(ds, batch_size=bs):
        logits, y = _batch_forward(model, batch, device, mach_dim, clin_dim)
        loss_sum += float(loss_fn(logits, y).item()) * y.size(0); n += y.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), loss_sum / max(n, 1)


def _save_curve(history, best_j, mid, title):
    """Two stacked subplots for the best HP set -> results/curves/<mid>.png:
    top = train_loss + val_loss (the over/under-fit gap), bottom = val_auroc."""
    rows = [h for h in history if h["hp"] == best_j]
    if not rows:
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in rows]
    fig, (axL, axA) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axL.plot(ep, [r["train_loss"] for r in rows], "C0-o", ms=3, label="train loss")
    axL.plot(ep, [r.get("val_loss", float("nan")) for r in rows], "C3-^", ms=3, label="val loss")
    axL.set_ylabel("loss"); axL.legend(loc="best", fontsize=8); axL.set_title(title, fontsize=9)
    axA.plot(ep, [r["val_auroc"] for r in rows], "C1-s", ms=3, label="val AUROC")
    axA.set_ylabel("val AUROC"); axA.set_xlabel("epoch"); axA.legend(loc="best", fontsize=8)
    fig.tight_layout()
    cdir = os.path.join(RESULTS_DIR, "curves"); os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, mid + ".png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def _build(pipeline, scope, encoder, mach_dim, clin_dim, hp, in_ch, device):
    """build_agg_model with the encoder's input dim pinned to the reduced feature dim."""
    sk = dict(hp.get("seq_kwargs") or {})
    sk["in_ch"] = in_ch
    return build_agg_model(pipeline, scope, seq_encoder=encoder,
                           machine_summary_dim=mach_dim, clinical_summary_dim=clin_dim,
                           fusion_dim=hp.get("fusion_dim", 128), dropout=hp.get("dropout", 0.3),
                           seq_kwargs=sk,
                           clinical_hidden=tuple(hp.get("clinical_hidden", (256, 128)))).to(device)


def _fit(pipeline, scope, encoder, case, tr, va, mach_dim, clin_dim, hp, in_ch, device,
         label="", hp_idx=0, history=None, log_every=1):
    """Train one model under HP `hp` with LR scheduling + early stop on val AUROC.

    Anti-overfit stack: weighted-BCE/focal loss, dropout (hp), AdamW weight-decay (hp),
    gradient clipping, ReduceLROnPlateau on val AUROC, and best-checkpoint restore.
    Logs per-epoch train_loss / val_auroc / lr (stdout + optional `history` list).
    Returns (best_val_auroc, best_state_dict).
    """
    model = _build(pipeline, scope, encoder, mach_dim, clin_dim, hp, in_ch, device)
    bs = hp.get("batch_size", 32)
    if DL_BALANCED_SAMPLER:
        # ~50/50 batches via minority oversampling; loss is NOT reweighted (avoid double-count)
        y = tr.seq.y.numpy()
        n_pos = float(y.sum()); n_neg = float(len(y) - n_pos)
        w = np.where(y == 1, 1.0 / max(n_pos, 1.0), 1.0 / max(n_neg, 1.0))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr), replacement=True)
        loader = DataLoader(tr, batch_size=bs, sampler=sampler)
        loss_fn = build_loss(LOSS, pos_weight=torch.tensor(1.0), gamma=FOCAL_GAMMA).to(device)
    else:
        loader = DataLoader(tr, batch_size=bs, shuffle=True)
        loss_fn = build_loss(LOSS, pos_weight=pos_weight(tr), gamma=FOCAL_GAMMA).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.get("lr", LR),
                            weight_decay=hp.get("weight_decay", WEIGHT_DECAY))
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=LR_SCHED_FACTOR,
                patience=LR_SCHED_PATIENCE, min_lr=MIN_LR)
             if USE_LR_SCHEDULER else None)

    best_auroc, best_state, patience = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            logits, y = _batch_forward(model, batch, device, mach_dim, clin_dim)
            loss = loss_fn(logits, y)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            bs = y.size(0); loss_sum += float(loss.item()) * bs; n_seen += bs
        train_loss = loss_sum / max(n_seen, 1)
        vp, vy, val_loss = _eval_epoch(model, va, loss_fn, device, mach_dim, clin_dim)
        auroc = roc_auc_score(vy, vp) if len(np.unique(vy)) > 1 else 0.0
        lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step(auroc)
        improved = auroc > best_auroc
        if improved:
            best_auroc, patience = auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if history is not None:
            history.append({"hp": hp_idx, "epoch": epoch, "train_loss": round(train_loss, 5),
                            "val_loss": round(float(val_loss), 5),
                            "val_auroc": round(float(auroc), 5), "lr": lr})
        if epoch % log_every == 0 or improved or patience >= EARLY_STOP_PATIENCE:
            print(f"[{label}] epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_auroc={auroc:.4f}  lr={lr:.1e}{'  *best' if improved else ''}", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"[{label}] early stop @ epoch {epoch} (best val_auroc={best_auroc:.4f})", flush=True)
            break
    return best_auroc, best_state


def train_one_agg(pipeline, scope, encoder, case, seed, search_iter=0,
                  variant="current", peak="denoised",
                  drop_branch=None, tag="", device=None, no_reduce=False, log_every=1):
    """Randomized HP search (search_iter sets) -> keep best val AUROC -> resampled eval.

    search_iter == 0 uses the documented fixed defaults (no search).
    Reads the abalation data root for (variant, peak); reduces the current-day
    stream per-encoder (DL_REDUCE) unless no_reduce.
    """
    _set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = tag or f"{variant}_{peak}"

    reduce_spec = None if (no_reduce or not DL_REDUCE_ENABLED) else DL_REDUCE.get(encoder)
    # combined = clinical-MLP + sequence; machine enters via the [T,7] stream, so the
    # separate prior-day machine-summary branch is OFF for combined.
    want_machine = False if pipeline == "combined" else None
    tr = AggDataset(case, "train", pipeline, scope, variant, peak, reduce=reduce_spec,
                    want_machine=want_machine)
    va = AggDataset(case, "val",   pipeline, scope, variant, peak, reduce=reduce_spec,
                    want_machine=want_machine, mach_scaler=tr.mach_scaler, clin_scaler=tr.clin_scaler)
    te = AggDataset(case, "test",  pipeline, scope, variant, peak, reduce=reduce_spec,
                    want_machine=want_machine, mach_scaler=tr.mach_scaler, clin_scaler=tr.clin_scaler)
    mach_dim, clin_dim, in_ch = tr.machine_dim, tr.clinical_dim, tr.seq_feat_dim

    hp_list = sample_hp(encoder, search_iter, seed) if search_iter and search_iter > 0 else [DEFAULT_HP]
    best = (-1.0, None, None, 1)     # (val_auroc, state, hp, hp_idx)
    history = []                     # per-epoch curves across all HP sets
    base_label = f"{pipeline}/{scope}/{encoder}/{case}/s{seed}"
    for j, hp in enumerate(hp_list):
        auroc, state = _fit(pipeline, scope, encoder, case, tr, va, mach_dim, clin_dim, hp, in_ch, device,
                            label=f"{base_label} hp{j+1}/{len(hp_list)}", hp_idx=j + 1,
                            history=history, log_every=log_every)
        if auroc > best[0]:
            best = (auroc, state, hp, j + 1)
        print(f"   [hp {j+1}/{len(hp_list)}] val_auroc={auroc:.4f}"
              f"{'  <- best' if best[1] is state else ''}", flush=True)

    mid = f"{tag}_{pipeline}_{scope}_{encoder}_{case}_s{seed}"
    # persist per-epoch curves (CSV) + a loss/AUROC plot of the best HP set
    hist_dir = os.path.join(RESULTS_DIR, "dl_history"); os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, mid + ".csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    best_auroc, best_state, best_hp, best_j = best
    curve_path = _save_curve(history, best_j, mid,
                             f"{base_label}  (best hp{best_j}, val AUROC {best_auroc:.3f})")
    model = _build(pipeline, scope, encoder, mach_dim, clin_dim, best_hp, in_ch, device)
    if best_state:
        model.load_state_dict(best_state)

    # save the best model weights + everything needed to rebuild it
    mdir = os.path.join(RESULTS_DIR, "saved_models", "dl"); os.makedirs(mdir, exist_ok=True)
    model_path = os.path.join(mdir, mid + ".pt")
    torch.save({"state_dict": best_state, "best_hp": best_hp, "in_ch": int(in_ch),
                "machine_dim": mach_dim, "clinical_dim": clin_dim,
                "clinical_hidden": list(best_hp.get("clinical_hidden", (256, 128))),
                "pipeline": pipeline, "scope": scope, "encoder": encoder,
                "best_val_auroc": float(best_auroc)}, model_path)

    vp, vy = _predict(model, va, device, mach_dim, clin_dim)
    thr = find_optimal_threshold(vy, vp)
    tp, ty = _predict(model, te, device, mach_dim, clin_dim)
    te_keys = pd.DataFrame(np.asarray(te.seq.keys), columns=KEY_COLS)   # test order matches tp/ty

    reduce_str = "raw" if reduce_spec is None else f"{reduce_spec[0]}{reduce_spec[1]}"
    row = {"family": "deep", "feature_set": pipeline, "case": case,
           "scope": scope, "model": encoder, "seed": seed, "variant": tag,
           "best_val_auroc": round(float(best_auroc), 4),
           "reduce": reduce_str, "seq_feat_dim": int(in_ch),
           "sampler": "balanced" if DL_BALANCED_SAMPLER else "weighted_loss",
           "machine_dim": mach_dim, "clinical_dim": clin_dim,
           "fusion": ("clinical_mlp+seq" if pipeline == "combined" else "seq+machine_summary"),
           "n_hp_tried": len(hp_list), "best_hp": json.dumps(best_hp),
           "drop_branch": drop_branch or "none",
           "history_csv": hist_path, "curve_png": curve_path, "model_path": model_path}
    row.update(eval_dispatch(te_keys, ty, tp, thr, case, variant, peak))
    return row


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=["machine_only", "combined"])
    ap.add_argument("--scope", default="per_day", choices=["per_day", "agg"])
    ap.add_argument("--encoder", default="cnn1d", choices=["cnn1d", "rnn", "transformer"])
    ap.add_argument("--case", default="1day")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="current")
    ap.add_argument("--peak", default="denoised",
                    help="data-root label (denoised/noised, or 'x' for LOO variants)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--search-iter", type=int, default=0,
                    help="randomized HP sets to try (0 = fixed defaults; >=10 recommended for final runs). "
                         "DL search is costlier than classical 30x5-fold, so this is the tunable budget.")
    ap.add_argument("--epochs", type=int, default=None, help="override MAX_EPOCHS (smoke test)")
    ap.add_argument("--no-reduce", action="store_true", help="use the raw [1440,4] stream (no downsampling)")
    ap.add_argument("--log-every", type=int, default=1, help="print every Nth epoch (best epochs always print)")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    if a.epochs is not None:
        import config as _c
        _c.MAX_EPOCHS = a.epochs
        globals()["MAX_EPOCHS"] = a.epochs
    row = train_one_agg(a.pipeline, a.scope, a.encoder, a.case, a.seed,
                        search_iter=a.search_iter, variant=a.variant, peak=a.peak, tag=a.tag,
                        no_reduce=a.no_reduce, log_every=a.log_every)
    print(f"[dl] {a.pipeline}/{a.scope}/{a.encoder}/{a.case}/s{a.seed}: "
          f"reduce={row['reduce']} (F={row['seq_feat_dim']}) | tried {row['n_hp_tried']} hp | "
          f"val_auroc={row['best_val_auroc']} | "
          f"full AUROC {row['full_test_AUROC']:.3f} | "
          f"subpop {row['subpop_AUROC_mean']}±{row['subpop_AUROC_std']} (n={row['n_eval_subpops']})")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        pd.DataFrame([row]).to_csv(a.out, index=False)
        print(f"[dl] wrote {a.out}")
