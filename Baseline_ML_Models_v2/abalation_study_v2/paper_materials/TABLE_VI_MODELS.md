# Table VI — Model Descriptions (input → processing → layers → output → loss)

Detailed specification of every model in Table VI (deep-learning benchmark + machine-only enhancement
+ combined late fusion), written for the paper's Methods section. Prediction target throughout: the
binary circuit-day mortality label (1-day = death on D+2; 2-day = death on D+2/D+3, with a 1-day
blanking gap). Every model outputs one probability P(death) per circuit-day.

Notation: `B` = batch, `T` = time (minutes, ≤1440), `C` = channels, `[·]` = tensor shape.

---

## 0. Shared inputs and the shared training protocol
Three input representations feed the models:

- **(M-seq) machine sequence** — the gap-aware per-minute pressure signal (raw channels access,
  filter, effluent, return + derived TMP, PFD [+ ARD]) on a 0–1439 grid, per-channel z-scored on
  TRAIN valid minutes, masked minutes → 0. Before the encoders it is **window-reduced**: 144
  non-overlapping 10-min windows, each summarized by 5 statistics per channel (mean, min, max, std,
  slope) → **`[144, C·5]`** + a window validity mask `[144]` (for CNN1D / RNN / Attention-CNN).
  The Transformer instead **patchifies** (non-overlapping 60-min patches, flattened) → `[24, C·60]`.
- **(Clin) clinical vector** — the patient's clinical features (demographics, comorbidities,
  severity, labs), median-imputed and standardized; mRMR-selected where noted (MLP, fusion clinical
  branch, combined models).
- **(Comb) combined tabular table** — machine-rich features (200) ⧺ clinical features, used by the
  combined-classical models and the stack.

**Shared DL training protocol** (all neural models):
- **Loss — weighted binary cross-entropy** on the logit `z` (probability `p = σ(z)`):
  `L = −(1/N) Σ_i [ w · y_i·log p_i + (1−y_i)·log(1−p_i) ]`, with `w = pos_weight = n_neg/n_pos`
  computed on TRAIN (option: focal loss `L = −(1/N) Σ α(1−p_t)^γ log p_t`, γ=2, α=0.25). When the
  **class-balanced sampler** is used (≈50/50 per batch via minority oversampling), `w` is set to 1
  to avoid double-counting.
- **Optimizer** AdamW (lr 3e-4, weight-decay 1e-3), gradient-norm clipping, ReduceLROnPlateau on val
  AUROC; **early-stop on validation AUROC**; best checkpoint restored.
- **Seeds** — 5; the five models' probabilities are **averaged** (the "mean over five seeds" in the
  caption).
- **Output** — a single logit → sigmoid → P(death); decision threshold (for F/precision/recall) =
  max-F1 point on validation.

**Classical base models** (inside the ensembles) minimize their own objectives — gradient boosting
and logistic regression minimize log-loss, random forest minimizes Gini impurity — and are tuned by
randomized search (5-fold CV, AUROC) and 5-seed averaged.

---

## 1. Clinical-only — MLP
**Role:** the clinical reference model (Table V/VI clinical-only row).
**Input:** (Clin) clinical feature vector `[B, d]` (d = number of selected clinical features).
**Processing / layers:**
```
Linear(d → 256) → BatchNorm → ReLU → Dropout(0.3)
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
Linear(128 → 1)                     # logit
```
**Output:** logit → σ → P(death). **Loss:** weighted BCE (above).
**Result:** AUROC 0.699 (1-day) / 0.700 (2-day) — the strongest single model overall.

---

## 2. Machine-only — CNN1D
**Role:** 1-D convolutional benchmark on the machine waveform.
**Input:** (M-seq) reduced sequence `[B, 144, C·5]` + mask `[B, 144]` → transpose to channels-first
`[B, C·5, 144]`.
**Processing / layers:**
```
3 × { Conv1d(in → 64, kernel 5, padding 2) → BatchNorm1d(64) → ReLU → Dropout(0.3) }   # → [B,64,144]
masked global-mean pool over valid windows                                            # → [B,64]
Linear(64 → 1)                                                                          # logit
```
Padding 2 keeps the length; the mask makes the pooling ignore padded/invalid windows (length-invariant).
**Output/Loss:** σ(logit) = P(death); weighted BCE. **Result:** 0.577 / 0.617.

---

## 3. Machine-only — RNN (BiLSTM / GRU)
**Role:** recurrent benchmark capturing longer temporal dependencies (bidirectional because a
completed day is classified, not forecast online).
**Input:** (M-seq) `[B, 144, C·5]` + per-day valid lengths (from the mask).
**Processing / layers:**
```
pack_padded_sequence(x, lengths)                       # exclude padded steps
BiLSTM (or BiGRU): input C·5, hidden 128, 1 layer, bidirectional   # → [B,144,256]
masked mean pool over valid steps                       # → [B, 256]  (= 2 × hidden)
Linear(256 → 1)                                         # logit
```
**Output/Loss:** σ(logit); weighted BCE. **Result:** 0.576 / 0.597.

---

## 4. Machine-only — Transformer
**Role:** self-attention benchmark; the source of native attention interpretability.
**Input:** (M-seq) patchified `[B, 24, C·60]`.
**Processing / layers:**
```
Linear(C·60 → d_model=128)                              # patch embedding
prepend a learnable [CLS] token; add sinusoidal positional encoding
TransformerEncoder × 4 layers, each:
    MultiheadAttention(d_model 128, 8 heads)  (key_padding_mask hides padded patches)
    + FeedForward(128 → 256 → 128), GELU, dropout 0.3, residual + LayerNorm
take the final [CLS] state                              # → [B, 128]
Linear(128 → 1)                                         # logit
```
**Output/Loss:** σ(logit); weighted BCE. **Result:** 0.572 / 0.556 (weakest machine encoder here).

---

## 5. Machine-only — Attention-CNN1D (ours)
**Role:** the proposed machine enhancement — the CNN1D but with **attention pooling** so the model
weights informative windows (e.g. end-of-day deterioration) instead of averaging them.
**Input:** (M-seq) `[B, 144, C·5]` + mask.
**Processing / layers:**
```
3 conv blocks (as CNN1D §2)                             # → [B, 64, 144]
attention scorer over windows:  Linear(64 → 32) → Tanh → Linear(32 → 1)   # → [B,144]
   invalid windows → −∞ ; softmax over windows → weights α                # [B,144]
attention pooling:  Σ_t α_t · h_t                       # → [B, 64]
Dropout(0.3) → Linear(64 → 1)                           # logit
```
**Output/Loss:** σ(logit); weighted BCE; 5-seed averaged. **Result:** 0.617 / 0.616 — best single
machine DL model.

---

## 6. Machine-only — Ensemble (ours)
**Role:** the headline machine-only model. **NB — this is a classical + DL blend, not an ensemble of
the DL rows above.**
**Inputs / members (each outputs a per-day probability):**
- **Rich-feature classical models** — GradientBoosting, RandomForest, LogisticRegression, each trained
  on the 200 machine-rich engineered features (median-impute → standardize → **mRMR top-30** →
  tuned → 5-seed averaged). Objectives: log-loss (GB, LR) / Gini (RF).
- **Attention-CNN1D** (§5) — the deep machine model.
**Processing (combiner):** the members are ranked by **validation AUROC**; the top-3 classical + the
CNN are kept and their **test probabilities are averaged** (`ensemble:mean`).
**Output:** mean probability = P(death). **"Loss":** none re-trained at this level — it is a fixed
average; the alternative `logit_stack` fits a logistic-regression meta-learner (logistic loss) on the
members' *validation* probabilities.
**Result:** **0.649 / 0.643** — above every single machine model because the strong classical members
(rich-RF ≈ 0.638, rich-GB ≈ 0.636), which are not shown as separate rows, lift the average.

---

## 7. Combined (late fusion) — CNN1D / RNN / Transformer fusion
**Role:** two-branch networks that fuse the machine waveform with clinical features (the classical
"combined" idea, in a DL form). *(Parent DL benchmark.)*
**Inputs:** (M-seq) machine sequence — here **7 channels** (4 raw + TMP/PFD/**ARD**) → `[144, 35]`
(`seq_feat_dim = 35`) — **and** (Clin) clinical vector (mRMR ~40), per circuit-day.
**Processing / layers:**
```
sequence branch:  CNN1D | RNN | Transformer encoder (§2–§4, minus its final Linear)   # → e_seq
                  (e_seq dim = 64 CNN1D / 256 RNN / 128 Transformer)
clinical branch:  MLP  Linear(d → 128) → BatchNorm → ReLU → Dropout(0.3)   [→ 128-d]   # → e_clin
late fusion:      concat(e_seq, e_clin)
head:             Linear(dim_seq+128 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1)     # logit
```
The two modalities are processed **independently, then concatenated** ("late fusion"), keeping each
branch's inductive bias. **Output/Loss:** σ(logit); weighted BCE. **Results:** CNN1D 0.668/0.660,
RNN 0.668/0.665, Transformer 0.658/0.653.

---

## 8. Combined — Attention-CNN fusion (ours)
**Role:** the proposed combined DL — our Attention-CNN1D machine branch fused with a clinical branch.
Best 2-day model.
**Inputs:** (M-seq) `[B,144,C·5]` + mask, and (Clin) mRMR-40 clinical vector `[B,40]`.
**Processing / layers:**
```
machine branch:   3 conv blocks + attention pooling (§5)               # → e_mach [B, 64]
clinical branch:  Linear(40 → 64) → BatchNorm → ReLU → Dropout(0.3)     # → e_clin  [B, 64]
late fusion:      concat(e_mach, e_clin)                                # [B, 128]
head:             Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)   # logit
```
**Output/Loss:** σ(logit); weighted BCE; balanced sampler; 5-seed averaged. **Result:** 0.687 /
**0.742** — the single best 2-day model.

**vs the CNN1D fusion (§7) — what differs, and are they comparable?** Both are late-fusion machine +
clinical models scored on the **identical held-out test set** (v2 70:10:20 patient split, 196
patients / 887 circuit-days, 62 positives 1-day / 109 2-day, same rolling-gap labels), so their
AUROCs are directly comparable. But they differ in **more than pooling**:
- machine-branch pooling: **masked global-mean (CNN1D)** → **attention pooling (Attn-CNN, ours)**;
- machine channels: **7** (incl. ARD), `seq_feat_dim = 35` → **6** (no ARD), `= 30`;
- clinical branch: **MLP (→256→128) → 128-d** → **`Linear(40→64)` → 64-d**;
- fusion/head dims: concat 192 → `192→128→1` → concat 128 → `128→64→1`;
- codebase: parent DL benchmark → exp_3 (retrained fresh with the balanced sampler).
Because several factors change together, the CNN1D→Attn-CNN gain (especially 2-day 0.660 → **0.742**)
reflects the **whole combined recipe, not attention pooling in isolation** — it is a valid
same-test-set comparison of the two models as built, but not a clean single-factor ablation of
attention pooling.

---

## 9. Combined — Ensemble / Stack (ours)
**Role:** the headline combined model — **stacks** the combined-classical models with the
Attention-CNN fusion. Best 1-day model.
**Inputs / members (each a per-day probability):**
- **Combined-classical models** — GradientBoosting, RandomForest, LightGBM, XGBoost, each on the
  (Comb) table (machine-rich 200 ⧺ clinical) with **grouped mRMR (top-20 machine + top-40 clinical =
  60 features)** → tuned → 5-seed averaged.
- **Attention-CNN fusion** (§8) — the combined DL.
**Processing (two-level stacking):**
```
Level-0:  each member outputs validation + test probabilities.
Level-1:  a logistic-regression meta-learner takes the members' probabilities as its inputs,
          is FIT ON VALIDATION (members' val probs + true val labels), then applied to test.
          "top-4" = the stack over the 4 highest-val-AUROC members.
```
**Output:** the meta-learner's probability = P(death). **Loss:** the Level-1 **logistic regression
minimizes logistic (cross-entropy) loss on the validation probabilities**; the Level-0 members keep
their own objectives (§0). **Result:** **0.740** (1-day) / 0.739 (2-day) — the best combined 1-day
model (which the plain fusion rows do not reach).

---

## Leakage barrier (applies to every model)
| fit on | what |
|---|---|
| **TRAIN** | impute/scale, mRMR, classical tuning, all neural-network weights |
| **VALIDATION** | ensemble/stack meta-learner, member selection, early-stopping, decision threshold |
| **TEST** | nothing — only the reported AUROC / AUPRC / F / precision / recall (+ patient-bootstrap CI) |

*Cross-refs: layer/hyperparameter tables in `DL_METHODS.md`; input/feature equations in
`DATA_PROCESSING.md`; the numbers in `RESULTS_COMPENDIUM.md` (Table 1b, §3c–§3d) and Table VI.*
