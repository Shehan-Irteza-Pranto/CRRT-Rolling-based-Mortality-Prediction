# Deep-Learning Methods — full architecture & training specification

Every neural model used in the experiments, with layer-level detail, hyperparameters, loss
functions, and training protocol — written for direct transcription into the paper's Methods
section. Source files cited per model. Defaults are the documented `Deep_Learning_Models/config.py`
values; the per-model width/depth search space is given where a randomized search was run.

> Data/signal preprocessing (cleaning, gap-aware signal, derived channels, feature equations,
> sequence reduction) is specified with equations in **`DATA_PROCESSING.md`**.

---

## 0. Shared data representation & training protocol (applies to all sequence models)

**Input.** Per-circuit-day pressure stream on a 0–1439 minute grid. Base benchmark uses the 4 raw
channels (ACCESS, FILTER, EFFLUENT, RETURN); the enriched study (abalation_study_v2) uses 7 channels
(4 raw + derived TMP, PFD, ARD). Padded/absent minutes carry a boolean **mask**; `MAX_LEN = 1440`,
right-crop if longer, pad value 0 (masked out).

**Sequence reduction before the encoder** (`models/dl_reduce.py`):
- CNN1D, RNN → **window aggregation**, window `k = 10` min → 144 windows, each window = `[mean, min,
  max, std, slope]` per channel (feature dim = channels × 5).
- Transformer → **patchify**, patch `P = 60` min (feature dim = channels × P), linear-embedded.

**Loss** (`training/losses.py`), train on the full imbalanced set:
- **Weighted BCE** (default): `BCEWithLogits` with `pos_weight = n_neg / n_pos` (from train split).
- **Focal loss** (option): `FL = α·(1−p_t)^γ·CE`, with `γ = 2.0`, `α = 0.25` (Lin et al. 2017).
- When the **balanced sampler** is on (default), batches are ~50/50 via `WeightedRandomSampler`
  (minority oversampled) and the loss `pos_weight` is set to 1.0 to avoid double-counting.

**Optimizer / regularization** (defaults; per-run overridable):
| knob | value |
|---|---|
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay (L2) | 1e-3 |
| gradient clipping | max-norm 1.0 |
| LR scheduler | ReduceLROnPlateau on val AUROC, factor 0.5, patience 5, min-LR 1e-6 |
| batch size | 32 (search: 32 / 64) |
| dropout | 0.3 (search: 0.2 / 0.3 / 0.5) |
| max epochs | 150 |
| early stopping | on **val AUROC**, patience 30, restore best checkpoint |
| seeds | 5 (0–4); report mean ± std |
| HP search | 30-iter randomized search per model (matches the GBM 30-iter × 5-fold budget) |

**Pooling over time** uses the mask so padded steps never contribute (masked mean, or attention).
**Evaluation:** test AUROC/AUPRC/F1–F3 + 1000× bootstrap CI; threshold = max-F1 on validation.

---

## 1. MLP (perceptron) — clinical-only rung & fusion clinical branch
File: `Deep_Learning_Models/models/mlp.py`. Input = engineered feature vector (not a sequence).

```
Linear(in_dim → 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
Linear(256   → 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
Linear(128   → 1)                                  # single logit
```
- Hidden sizes `(256, 128)` (search: `[128]`, `[256,128]`, `[256,128,64]`).
- `in_dim` = number of clinical (or machine) summary features for that pipeline.
- The penultimate 128-d activation (`embed()`) is reused as the **clinical branch** inside fusion.
- Output: 1 logit → sigmoid → P(death). Used for the **clinical-only** pipeline (best DL clinical:
  AUROC 0.699 1-day / 0.700 2-day).

---

## 2. CNN1D — machine sequence encoder
File: `Deep_Learning_Models/models/cnn1d.py`. Input `[B, T, C]` → transpose → `[B, C, T]`.

```
for each of n_blocks (default 3):
    Conv1d(in_ch → 64, kernel=5, padding=2) → BatchNorm1d(64) → ReLU → Dropout(0.3)
masked global-mean pool over valid time steps   → embedding [B, 64]
Linear(64 → 1)                                   → logit
```
- Channels 64 (search: 32 / 64 / 128); n_blocks 3 (search: 2 / 3 / 4); kernel 5 (search: 3 / 5 / 7).
- `in_ch` = reduced feature dim (channels × 5 after window aggregation).
- Length-invariant (padding=2 keeps T; masked pooling ignores padded steps).
- Best DL machine-only encoder (per-day): AUROC 0.577 (1-day) / 0.617 (2-day).

---

## 3. RNN — BiLSTM / GRU sequence encoder
File: `Deep_Learning_Models/models/rnn.py`. Bidirectional (classifying a *completed* day window).

```
pack_padded_sequence(x, lengths=mask.sum(1))           # exclude padded steps
BiLSTM or BiGRU: input_size=in_ch, hidden=128, num_layers=1, bidirectional=True
pad back → masked mean pool over valid steps           → embedding [B, 2·128 = 256]
Linear(256 → 1)                                        → logit
```
- Cell: LSTM (default) or GRU (search). hidden 128 (search: 64 / 128 / 256); layers 1 (search: 1 / 2;
  inter-layer dropout applies only when layers > 1).
- `out_dim = 2 × hidden` (bidirectional concat).
- Best DL combined-2day encoder (fused): RNN, AUROC 0.665.

---

## 4. Transformer — self-attention sequence encoder
File: `Deep_Learning_Models/models/transformer.py`.

```
Linear(in_ch → d_model=128)                            # input projection
prepend learnable [CLS] token; add sinusoidal positional encoding
TransformerEncoder × n_layers (default 4):
    MultiheadAttention(d_model=128, n_heads=8, batch_first)   # key_padding_mask hides padded steps
    + FeedForward(128 → ff=256 → 128), activation = GELU, dropout 0.3, residual + LayerNorm
take final [CLS] state                                 → embedding [B, 128]
Linear(128 → 1)                                        → logit
```
- d_model 128 (search: 32 / 64 / 128); n_heads 8 (search: 4 / 8); n_layers 4 (search: 1 / 2 / 3);
  feed-forward 256 (search: 128 / 256).
- Patchified input (P = 60). CLS is always attended; padded patches masked from attention.
- Native attention weights are used for interpretability (attention rollout).

---

## 5. Fusion (combined pipeline) — sequence + clinical late fusion
Files: `Deep_Learning_Models/models/fusion.py`, `abalation_study_v2/models/dl_aggregated.py`.

Two (per-day) or three (aggregated) parallel branches, **late-fused by concatenation**:

```
sequence branch : CNN1D | RNN | Transformer encoder on the [T, C] stream      → e_seq
clinical branch : MLP encoder (256 → 128, BN-ReLU-Dropout each)               → e_clin (128-d)
[aggregated only] prior-day machine-summary branch:
                  Linear(mach_dim → 128) → BN → ReLU → Dropout                → e_mach (128-d)

fused = concat(e_seq, e_clin[, e_mach])
head:  Linear(fused → 128) → ReLU → Dropout(0.3) → Linear(128 → 1) → logit
```
- For the **combined** pipeline the machine signal enters *only* through the sequence branch (the
  derived channels are in the [T,7] stream), so there is **no** separate machine-summary branch in
  the per-day combined model; the clinical branch is the full clinical MLP `(256,128)`.
- Branch-ablation supported by zeroing one branch's embedding (`drop_branch ∈ {seq, clinical,
  machine}`).
- Fusion-dim 128 (search: 64 / 128); sequence encoder selectable (search: cnn1d / rnn / transformer).
- Best DL combined: CNN1D-fusion 0.668 (1-day), RNN-fusion 0.665 (2-day).

---

## 6. Attention-CNN1D — exp_3 machine-only enhancement (this work)
Files: `abalation_study_v2/exp_3/dl/model.py` (+ `ARCHITECTURE.md`). Replaces the CNN1D's masked-mean
pool with **masked attention pooling** so end-of-day deterioration windows can be up-weighted.

```
CNN1D conv stack (3 × [Conv1d k5 → BN → ReLU → Dropout 0.3], 64 ch)  → h [B, 64, T]
attention score per window:  Linear(64 → 32) → Tanh → Linear(32 → 1)
  invalid windows set to −1e9 → softmax over windows → weights α
day embedding = Σ_t α_t · h_t                                          → [B, 64]
Dropout(0.3) → Linear(64 → 1)                                          → logit
```
- ~53.3k parameters. Training differs slightly from the base protocol: weight-decay **1e-4**,
  gradient-clip **5.0**, early-stop patience **15**, 5-seed probability averaging.
- Used as one of the two base models in the machine-only ensemble (5-seed test AUROC 0.617 1-day /
  0.616 2-day).

---

## 7. Exploratory DL methods (used, then rejected — reported as negative results)
For completeness; these were trained but did **not** beat the plain CNN1D and are not in the final model.

- **Gated-attention MIL** (`exp_2/mil/`, removed): frozen **MOMENT** time-series foundation-model
  embeddings (6144-d per fragment) → gated-attention pooling over a day's fragments (sub-bags) →
  logit. Collapsed to ~0.51 (noisy-OR saturation + unscaled embeddings); day-mean embedding + LogReg
  sanity check gave 0.606.
- **Fragment-recombination CNN1D** (`exp_2/frag_cnn/`): the same CNN1D trained on augmented samples
  (random temporally-ordered fragment subsets per day, ≥30 % positives) with test-time augmentation.
  Overfit (train 0.96 / test 0.567) — below the plain CNN1D. Confirms recombination cannot raise the
  effective sample size.

---

## 8. Method → experiment map
| model | pipeline(s) | where | best test AUROC (1day / 2day) |
|---|---|---|---|
| MLP | clinical-only | E1 benchmark | 0.699 / 0.700 |
| CNN1D | machine-only, combined (seq) | E1 | 0.577 / 0.617 (machine) |
| RNN (BiLSTM/GRU) | machine-only, combined (seq) | E1 | combined 2day 0.665 |
| Transformer | machine-only, combined (seq) | E1 | (≤ CNN1D/RNN) |
| Fusion (seq + clinical MLP) | combined | E1 | 0.668 / 0.665 |
| Attention-CNN1D | machine-only | E3 (this work) | 0.617 / 0.616 |
| MIL / fragment-CNN | machine-only (exploratory) | negative results | 0.51 / 0.567 |

All DL training: AdamW, weighted-BCE (or focal), balanced sampler, ReduceLROnPlateau, early-stop on
val AUROC, 5 seeds. See `RESULTS_COMPENDIUM.md` for full metric tables and `../exp_3/ARCHITECTURE.md`
for the machine-only ensemble diagram.
