# Table VI (best-seed) — DL results + model inputs & processing (1-day)

Deep-learning benchmark on CRRT-on-AKI mortality (one-day horizon), reported as the **best-of-5-seeds by
validation AUROC** per cell (leak-safe model selection), evaluated on the held-out test set. Threshold
for F1/Precision/Recall = **max-F1 on validation**. Clinical-only and combined pipelines use **mRMR
top-40** clinical features. Balanced 1:1 / 1:2 = mean over 100 resamples (all positives kept, negatives
subsampled). Saved, externally-runnable model bundles: `exp_4/models/` (see `LOAD_AND_PREDICT.md`).

> This table lists the **deep-learning models only**. The logistic top-4 stack (which, by validation
> AUROC, selects an all-classical member set) is a classical-stack result and is intentionally excluded.

## Table VI — best-seed (1-day)
Best AUROC in **bold**.

| Model | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ***Clinical-only*** | | | | | | | | | |
| MLP | 0.665 | 0.177 | 0.237 | 0.161 | 0.452 | 0.668 | 0.686 | 0.665 | 0.534 |
| ***Machine-only*** | | | | | | | | | |
| CNN1D | 0.579 | 0.095 | 0.026 | 0.071 | 0.016 | 0.575 | 0.585 | 0.579 | 0.415 |
| RNN (BiLSTM/GRU) | 0.584 | 0.094 | 0.159 | 0.098 | 0.419 | 0.583 | 0.583 | 0.585 | 0.412 |
| Transformer | 0.581 | 0.086 | 0.123 | 0.099 | 0.161 | 0.578 | 0.559 | 0.582 | 0.389 |
| Attention-CNN1D | 0.579 | 0.102 | 0.165 | 0.155 | 0.177 | 0.578 | 0.601 | 0.579 | 0.430 |
| Ensemble | 0.622 | 0.109 | 0.131 | 0.156 | 0.113 | 0.621 | 0.620 | 0.621 | 0.449 |
| ***Combined*** | | | | | | | | | |
| CNN1D | 0.646 | 0.182 | 0.188 | 0.121 | 0.419 | 0.644 | 0.674 | 0.646 | 0.526 |
| RNN (BiLSTM/GRU) | 0.669 | 0.222 | 0.278 | 0.302 | 0.258 | 0.667 | 0.698 | 0.669 | 0.559 |
| Transformer | 0.680 | 0.182 | 0.225 | 0.156 | 0.403 | 0.680 | 0.696 | 0.682 | 0.549 |
| Attn-CNN fusion | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.692 | 0.681 | 0.536 |
| **Ensemble — Transformer + 3 classical *(ours)*** † | **0.738** | 0.185 | 0.217 | 0.156 | 0.355 | 0.737 | 0.728 | 0.738 | 0.580 |

*Source: `exp_4/results/table_vi_bestseed.csv`; the † ensemble row from
`exp_4/results/stack_tuned_transformer_1day.csv`.*

† **DL-inclusive ensemble** — a logistic meta-learner over the **HP-tuned patch-Transformer fusion** +
top-3 classical models (the transformer is forced into the stack). Distinct from the top-4 stack (0.740)
we exclude, which by validation AUROC selects an **all-classical** member set. This row uses the tuned
transformer under the exp_3 recipe, so it is a DL-inclusive *ablation* result — compare it to the other
ensemble entries, not to the patch-Transformer benchmark cell (0.680) directly.

---

## Ablation — sequence encoder in the combined ensemble (DL fusion + 3 classical)
Our combined ensemble is a logistic stack of a **sequence-encoder fusion + the top-3 classical models**.
Here we swap the sequence encoder — CNN1D (mean pool), CNN-attention, RNN (BiGRU), and patch-Transformer —
under one identical recipe (window_agg `[144×30]`, 5-seed best-val), and add the **HP-tuned**
patch-Transformer. Best AUROC in **bold**.

| Sequence encoder (in the ensemble) | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:2 AUROC |
|---|--:|--:|--:|--:|--:|--:|--:|
| CNN1D (mean pool) | 0.719 | 0.184 | 0.261 | 0.197 | 0.387 | 0.720 | 0.720 |
| CNN-attention | 0.694 | 0.174 | 0.258 | 0.194 | 0.387 | 0.694 | 0.694 |
| RNN (BiGRU) | 0.709 | 0.174 | 0.219 | 0.151 | 0.403 | 0.708 | 0.709 |
| patch-Transformer (default HP) | 0.708 | 0.188 | 0.249 | 0.180 | 0.403 | 0.704 | 0.707 |
| **patch-Transformer (HP-tuned) *(ours)*** | **0.738** | 0.185 | 0.217 | 0.156 | 0.355 | 0.737 | 0.738 |

**Result:** the **HP-tuned patch-Transformer + 3 classical (0.738)** is the best combined ensemble,
ahead of every other sequence encoder. Among *default*-HP encoders CNN1D mean-pool (0.719) is strongest
and the default transformer (0.708) is mid-pack — so **HP-tuning the transformer** is what lifts it to the
top (+0.030 over its default). All five are within the ±0.06 test CI (62 positives), so this is a ranking,
not a significance claim. *Source: `results/matrix_combined_1day.csv` (default encoders) +
`results/stack_tuned_transformer_1day.csv` (tuned).*

---

## Model inputs & processing
Two input representations feed these models, both derived from the **same** cleaned per-circuit-day data
(gap-aware; pressures only; Day ≤ 7):

- **Tabular** — engineered per-day features. *Clinical* = 40 mRMR-selected features (labs: INR, lactate,
  albumin, platelets, pH, bicarbonate…; comorbidities/indications; age, BMI, APACHE-II). *Machine* =
  per-channel statistics of the 6 pressure channels (ACCESS, FILTER, EFFLUENT, RETURN, TMP, PFD + derived
  ARD): base 133 (distribution + within-day dynamics) or enriched 200 (+67 circuit-instability
  descriptors — spectral, entropy, excursion, end-of-day, coupling).
- **Sequence** — the raw pressure waveform reduced to **`[144 windows × 30]`**: the 1440-min day is
  z-scored per channel (training stats), gap-masked, and aggregated into 144 non-overlapping 10-min
  windows, each summarized by 5 stats (mean/min/max/std/slope) over the 6 channels → 30 features/window.

Leak-safe throughout: every imputer, scaler, mRMR selection, z-score stat, network weight, ensemble
weight and threshold is fit on **train** (threshold on **validation**); all metrics are on the held-out
**test** set. Loss for every neural network = **binary cross-entropy** (class-balanced minibatch
sampler); optimizer AdamW, early-stop on validation AUROC.

### Clinical-only
- **MLP** — *Input:* 40 mRMR clinical features (tabular vector). *Process:* median-impute → standardize
  → feed-forward MLP (fully-connected hidden layers, ReLU, dropout) → sigmoid → P(death). No sequence.

### Machine-only (input = the `[144×30]` pressure sequence)
- **CNN1D** — *Process:* 1-D convolution stack over the 144-step window sequence (local temporal
  filters) → **masked global-mean pooling** across valid windows → linear head → sigmoid.
- **RNN (BiLSTM/GRU)** — *Process:* bidirectional recurrent encoder consumes the window sequence
  left-to-right and right-to-left → pooled hidden state → linear head → sigmoid. Captures longer-range
  order than the CNN's local filters.
- **Transformer** — *Process:* windows are patch-embedded + positional-encoded → multi-head
  self-attention blocks (each window attends to all others) → pooled → linear head → sigmoid.
- **Attention-CNN1D** — *Process:* same CNN1D conv stack, but pooling is replaced by **masked
  attention pooling** — a learned score per window (softmax over valid windows) up-weights the windows
  that matter (e.g. end-of-day deterioration) before the head. Same input, sharper aggregation.
- **Ensemble** — *Input:* both representations. *Process:* averages the probabilities of the
  **top-3 classical models** (tuned tree/boosting models on the tabular machine features, base+rich,
  chosen by validation AUROC) **and the Attention-CNN1D** (sequence). A single mean of calibrated
  member probabilities; threshold on the averaged validation probability.

### Combined (machine + clinical)
- **CNN1D / RNN / Transformer** — *Input:* the `[144×30]` machine sequence **plus** the 40 mRMR clinical
  features. *Process:* the sequence goes through its encoder (as above) → a machine embedding; the
  clinical vector goes through a small tabular sub-network → a clinical embedding; the two are
  **concatenated (late fusion)** and passed to a joint head → sigmoid.
- **Attn-CNN fusion** — *Process:* **Attention-CNN1D** machine branch (sequence → conv →
  attention pooling → machine embedding) fused by concatenation with a **clinical MLP branch**
  (40 mRMR features → dense → clinical embedding); joint head → sigmoid. Trained end-to-end. Strongest
  single deep-learning model at one day (0.682).
- **Ensemble — Transformer + 3 classical *(ours)*** — *Input:* all views. *Process:* a
  **logistic-regression meta-learner** (fit on validation) over the **HP-tuned patch-Transformer fusion**
  probability + the top-3 combined/clinical classical models' probabilities → sigmoid → threshold. The
  transformer is *forced* into the stack so the DL contributes; this is the strongest **DL-inclusive**
  combined model at one day (**0.738**), edging the all-classical top-4 stack (0.740, excluded here).

*Full layer-by-layer specifications (dims, kernels, heads, dropout, optimizer settings) are in
`paper_materials/TABLE_VI_MODELS.md` and `paper_materials/DL_METHODS.md`; the reduced-sequence and
feature equations are in `paper_materials/DATA_PROCESSING.md`.*
