# Deep Learning Models — CRRT-on-AKI Mortality

The **deep half** of a systematic benchmark. The classical half already lives in
`../Clinical_Only_Models`, `../Machine_Only_Models`, `../Machine_Clinical_Models`.
This folder adds sequence and neural models so the paper becomes one complete study:

> **classical vs. deep × clinical vs. machine vs. combined × per-day vs. aggregated × 1- vs. 2-day**,
> for CRRT-on-AKI mortality — testing whether **CRRT pressure-waveform temporal structure carries
> disproportionate, complementary mortality signal that sequence models can exploit beyond engineered
> summary features**, corroborated by convergent classical/DL interpretability.

This is **not** "we added deep learning." The MLP is a baseline rung, not the headline. The contribution
is sequence modeling of the raw 4-channel pressure streams recovering within-window dynamics that the 52
engineered summary statistics (mean, q25, skewness, range, n_valid, …) structurally discard.

---

## The central data decision

| Branch | Input | Shape | Source |
|---|---|---|---|
| **Sequence (machine)** | raw per-minute pressure streams | `[T, 4]` channels = access / filter / effluent / return | `Data/Processed/Machine/pressure/Denoised/CRRT_pressure_wo_peak.csv` |
| **Tabular (clinical)** | engineered clinical/lab/PTI features | `[n_clinical]` | `Data/Processed/Model_Ready/clinical_only/{case}/` |
| **Tabular (machine)** *(MLP baseline only)* | the 52 engineered pressure stats | `[52]` | `Data/Processed/Model_Ready/{case}/denoised/` |

Sequence models are fed **raw windowed streams, not the 52 summary features**. Clinical stays tabular and is
**late-fused** into the sequence encoder. One sequence = one circuit-day `(PID, Visit_no, Day)`, which joins
1:1 to the classical model-ready rows — so **splits and labels are identical by construction** (verified:
zero PID overlap train/test; patient-level splits).

---

## The baseline ladder (legible comparison)

| Rung | Model | Input | Role |
|---|---|---|---|
| 0 | GB / XGBoost (existing) | engineered features | the numbers to beat — **AUROC 0.801** GB-on-aggregated-combined |
| 1 | **Perceptron / MLP** | engineered features (clinical / machine / combined) | fair head-to-head vs GBM, satisfies "perceptron". Expect ≈ or < GB; report honestly |
| 2 | **1D-CNN** | raw streams `[T,4]` | local waveform morphology (what skewness/range crudely approximate) |
| 3 | **RNN (BiLSTM / GRU)** | raw streams `[T,4]` | longer temporal dependencies; bidirectional (classifying a *completed* window) |
| 4 | **Transformer encoder** | raw streams `[T,4]` | strongest sequence model; `[CLS]` pooling + native attention interpretability |
| 4b | **MOMENT frozen embedding + head** *(optional)* | raw streams | nods at the waveform/MOMENT dissertation line; keep optional given timeline |
| 5 | **Fusion head** | sequence encoder (machine) ⊕ MLP branch (clinical) | the headline test: does DL recover/beat the complementary-machine-signal finding? |

---

## Rigor protocol (where DL clinical-ML papers get rejected)

Enforced centrally in `common/` and `config.py` so the DL pipeline **cannot drift** from the classical one:

1. **Identical protocol** — same patient-level splits, same balanced 50/50 test eval, same metrics
   (AUROC / AUPRC / F1 / precision / recall), validation-derived F1 threshold. Reuses the exact classical
   metric functions (`common/metrics.py`).
2. **Mean ± std over ≥5 seeds** — DL is high-variance and positive bags are sparse (≈106/188 balanced).
   A single run is not credible. Paired with bootstrap CIs. **The single most common rejection reason.**
3. **Fair tuning budget** — GBMs got 30-iter RandomizedSearch × 5-fold. DL gets a comparable, *documented*
   search (LR, depth/width, dropout, batch size) — see `training/search.py`.
4. **Class imbalance** — train on the full imbalanced set with **weighted BCE or focal loss**; early-stop on
   **validation AUROC**. Stated explicitly, not hidden.
5. **Per-channel normalization + masking** — normalization stats fit on *train only*; masks handle
   variable-length / missing pressure segments (the `n_valid` gaps).

## Interpretability thread (an asset — extend, don't drop)

SHAP is one of the stronger parts of the current draft. Extend it to DL for **converging evidence**:

- **CNN / RNN** → Integrated Gradients / DeepSHAP over the pressure streams.
- **Transformer** → attention rollout.

If DL temporal saliency *independently* re-surfaces **filter** and **return** pressure as dominant, that is a
much harder clinical claim to dismiss than SHAP alone — exactly the "interpretable DL on physiological
signals" framing.

---

## Layout

```
Deep_Learning_Models/
├── README.md                  # this file
├── PAPER_PLAN.md              # task breakdown / checklist for the paper
├── requirements.txt           # torch + deps (none installed yet)
├── config.py                  # central config — mirrors classical conventions
├── common/
│   └── metrics.py             # EXACT classical metric protocol (reused, not reimplemented)
├── data/
│   ├── prepare_sequences.py   # raw streams -> [T,4] tensors, identical splits, train-fit norm
│   └── sequence_dataset.py    # torch Dataset + padded/masked collate
├── models/
│   ├── mlp.py                 # rung 1
│   ├── cnn1d.py               # rung 2
│   ├── rnn.py                 # rung 3
│   ├── transformer.py         # rung 4
│   ├── moment_head.py         # rung 4b (optional)
│   └── fusion.py              # rung 5
├── training/
│   ├── losses.py              # weighted BCE / focal
│   ├── train.py               # multi-seed harness, early-stop on val AUROC
│   ├── search.py              # documented HP search (fair budget)
│   └── evaluate.py            # identical eval protocol + bootstrap CIs
├── interpretability/
│   ├── integrated_gradients.py
│   └── attention_rollout.py
└── results/                   # raw_results.csv, per-seed logs, saliency plots, CMs
```

## Run order

```bash
pip install -r requirements.txt
python data/prepare_sequences.py --cases 1day 2day --stream denoised   # build sequence tensors

# Option A — everything from scratch (all 8 rungs, all seeds), then aggregate + plot:
python training/run_all.py                                             # writes results + curves

# Option B — one rung at a time:
python training/train.py --model cnn1d   --cases 1day 2day --seeds 5    # train one rung, 5 seeds
python training/train.py --model fusion  --cases 1day 2day --seeds 5
python training/evaluate.py --aggregate                                 # mean±std, bootstrap CIs, tables
python training/plots.py                                               # loss / AUROC / PR curves
```

### Per-rung artifacts written by `train.py`

| Output | Path | Contents |
|---|---|---|
| Best checkpoint | `results/best_models/<tag>_<case>_seed<seed>.pt` | val-AUROC-selected weights + `build_kw`/threshold to rebuild for validation |
| Per-epoch log | `results/training_history/training_history_<tag>.csv` | `epoch, train_loss, val_loss, val_auroc, best_so_far` per (case, seed) |
| Test predictions | `results/predictions/<tag>_<case>_seed<seed>.npz` | `y_true, y_score, bal_idx, threshold` for ROC / PR curves |
| Curves | `results/curves/<tag>_<case>_{training,test}_curves.png` | loss + val-AUROC vs epoch; ROC + precision-recall on test |

Reload a saved best model for validation:

```python
import torch
from training.train import build_model
ck = torch.load("results/best_models/cnn1d_1day_seed0.pt")
model = build_model(ck["model_name"], **ck["build_kw"]); model.load_state_dict(ck["state_dict"]); model.eval()
# ck["threshold"] is the validation-derived F1 decision threshold
```

> **Status:** scaffold. PyTorch is not yet installed in this environment — `requirements.txt` pins it.
> `data/prepare_sequences.py` and `common/metrics.py` are framework-free and runnable now; the model/
> training modules are PyTorch and untested until torch is installed. See `PAPER_PLAN.md` for what remains.
