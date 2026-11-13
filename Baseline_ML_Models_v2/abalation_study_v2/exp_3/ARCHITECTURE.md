# exp_3 — Machine-Only CRRT Mortality Model: Full Architecture

End-to-end specification of the machine-only (pressure-data) pipeline, from raw signal to final
score. Every tensor shape, channel count, feature count, and layer config is the real value used in
the built code. Use this to draw the framework diagram.

Outcome label = circuit-day mortality, rolling with a 1-day blanking gap: from a circuit-day D,
**1-day** predicts death on D+2, **2-day** predicts death on D+2 or D+3 (the death day D and the gap
day D+1 are excluded). Same input signal; only the horizon differs.
Prediction unit = one **circuit-day** (one CRRT circuit on one calendar day for one patient).

> Equation-complete preprocessing (raw cleaning, gap rule, derived channels, all feature formulas,
> sequence reduction) is in **`../paper_materials/DATA_PROCESSING.md`**.

---

## 0. Top-level data flow

```
                         RAW PRESSURE CSV (per-minute, 4 channels)
                                      │
                       ┌──────────────▼───────────────┐
                       │  SIGNAL PROCESSING (shared)   │   gap-aware, derived channels
                       │  → day signal  X[6,1440]+mask │
                       └───────┬───────────────┬───────┘
              physical units   │               │   z-scored (train stats)
                       ┌───────▼──────┐   ┌─────▼─────────────┐
                       │  BRANCH A    │   │   BRANCH B        │
                       │ engineered   │   │  attention CNN1D  │
                       │ 200 features │   │  reduce→conv→attn │
                       │ →mRMR30→5 ML │   │  →logit (5 seeds) │
                       └───────┬──────┘   └─────────┬─────────┘
                  per-day prob │                    │ per-day prob
                       └────────────┬───────────────┘
                            ┌────────▼─────────┐
                            │  ENSEMBLE (meta) │  mean / logistic-stack / best-by-val
                            └────────┬─────────┘
                            ┌────────▼─────────────────────┐
                            │ EVAL: day-level AUROC/AUPRC/  │
                            │ F1-F3; 95% CI bootstrap over  │
                            │ PATIENTS (held-out test)      │
                            └───────────────────────────────┘
```

Two branches are **deliberately complementary**: Branch A = hand-engineered global statistics;
Branch B = learned local waveform morphology. Their errors are partly uncorrelated, so combining
them helps.

---

## 1. Signal processing (shared front-end)
Files: `features/day_signal.py` (Branch A, physical units), `dl/dl_data.py` (Branch B, z-scored).

**Input source.** `Data/Processed/Machine/pressure/denoised/CRRT_pressure_wo_peak.csv` — per-minute
pressures with priming + downtime removed and alarm peaks smoothed ("denoised" variant).

**1.1 Calendar-minute grid.** Each circuit-day is placed on a fixed 0–1439 minute grid (1440 min/day).
Raw channels (4): `ACCESS_PRESSURE`, `FILTER_PRESSURE`, `EFFLUENT_PRESSURE`, `RETURN_PRESSURE`.
Missing minutes → `NaN`. Result: a `[4, 1440]` matrix + a boolean presence vector `[1440]`.

**1.2 Gap rule** (`GAP_MAX_MIN = 10`):
- Interior gap **≤ 10 min**, bracketed by observed minutes → **linear interpolation**, `mask = 1`.
- Interior gap **> 10 min**, and all **leading / trailing** absence → **left NaN**, `mask = 0`.

So `mask[1440]` marks which minutes are real-or-safely-interpolated (valid) vs unknown.

**1.3 Derived physiology channels** (computed from the gap-filled raw channels):
| channel | formula | physiology |
|---|---|---|
| TMP | (FILTER + RETURN)/2 − EFFLUENT | transmembrane pressure → membrane fouling |
| PFD | FILTER − RETURN | pressure drop across filter → clotting |
| ARD | RETURN − ACCESS | circuit pressure span → access instability *(Branch A only)* |

Signal tensor becomes **`X[6, 1440]`** = [access, filter, effluent, return, TMP, PFD] + `mask[1440]`.
(ARD is derived inside Branch A's feature code, so the DL tensor stays 6 channels.)

**1.4 Cohort & ordering.** Days restricted to **Day ≤ 7**. Patient-level **stratified 70:10:20** split
(`pid_splits_v2.json`), positive-rate matched. Row order fixed by the machine_only table →
**train 3236 / val 462 / test 887** circuit-days (1-day positives: 201 / 30 / 62).

**1.5 Two normalizations** (the branches diverge here):
- **Branch A** keeps **physical units** — alarm thresholds, band-power, and end-of-day deltas need the
  real pressure scale. Masked minutes stay `NaN` (ignored per-feature).
- **Branch B** applies **per-channel z-score**, fit on **TRAIN valid minutes only** (cached, reused for
  val/test); masked minutes → 0 (= channel mean after z-score).

---

## 2. Branch A — engineered features + classical models
Files: `features/build_rich_features.py`, `features/classical_rich.py`.

### 2.1 Feature set = 200 features per circuit-day

**(a) Parent 133 features** — 7 channels (4 raw + TMP + PFD + ARD):
- **Distribution stats (13 × 7 = 91):** mean, std, min, max, median, q25, q75, iqr, range, skewness,
  kurtosis, cv, n_valid.
- **Temporal-dynamics stats (6 × 7 = 42):** slope, trend_r2, mad (mean abs Δ), max_jump, ac1 (lag-1
  autocorr), exc_frac (fraction |z|>2).

**(b) 67 NEW exp_3 features** — the circuit-instability signatures the parent set lacked:
- Per channel, 7 channels (6 base + ARD) × 9 = **63**:
  | group | features |
  |---|---|
  | spectral (Welch PSD) | spec_entropy, bandpow_lo (power at period > 30 min), bandpow_hi (period < 10 min), dom_period (1/peak-freq) |
  | complexity | perm_entropy (permutation entropy, order 3) |
  | excursion | cross_rate (median-crossings / min), max_run_above (longest run above median, fraction) |
  | end-of-day | eod_delta (mean last 2 h − mean first 2 h), eod_slope (slope over last 2 h) |
- **Cross-channel coupling (4):** TMP_PFD_corr, ACCESS_RETURN_corr, FILTER_EFFLUENT_corr,
  TMP_PFD_cotrend (sign of slope_TMP × slope_PFD — clotting raises both together).

**Total = 133 + 67 = 200 features.** (The pipeline is also run on the parent-133-only set as a
"base" control, to isolate the new features' lift.)

### 2.2 Classical pipeline (every step fit on TRAIN only — leak-safe)
```
200 feats → median impute → StandardScaler → mRMR select top-30
          → RandomizedSearchCV (5-fold CV, scoring = AUROC) per model
          → refit with 5 random seeds → AVERAGE the 5 probability vectors
```
- **mRMR** = minimum-Redundancy Maximum-Relevance; keeps 30 features. On 1-day it kept **12 of 30 from
  the NEW set** (dom_period of several channels, eod_slope/eod_delta, perm_entropy, max_run_above).
- **Models (5):** GradientBoosting, RandomForest, LightGBM, XGBoost, LogisticRegression.
- Output per model: one death-probability per circuit-day (on val and test).

---

## 3. Branch B — attention CNN1D
Files: `dl/dl_data.py`, `dl/model.py`, reuses `Deep_Learning_Models/models/cnn1d.py` (`CNN1DEncoder`)
and `models/dl_reduce.py` (`window_aggregate`).

### 3.1 Window reduction (`window_aggregate`, k = 10)
z-scored signal `[1440, 6]` → **144 non-overlapping 10-min windows**; each window → 5 stats per
channel: **[mean, min, max, std, slope]**.
→ reduced sequence **`[144, 30]`** (30 = 6 channels × 5 stats) + window mask `[144]`
(window valid if it contains ≥1 valid minute). This downsamples 1440→144 while preserving local
shape (slope/std capture within-window dynamics).

### 3.2 CNN1D encoder (3 conv blocks)
Input `[B, 144, 30]` → transpose → `[B, 30, 144]` (channels-first for Conv1d).
Each of **3 blocks**:
```
Conv1d(in_ch → 64, kernel = 5, padding = 2)  →  BatchNorm1d(64)  →  ReLU  →  Dropout(p = 0.3)
```
in_ch = 30 for block 1, then 64. Output **`[B, 64, 144]`** (time length preserved by padding=2).

### 3.3 Attention pooling (exp_3's change vs the parent's masked-mean)
Per window t, a learned score:
```
score_t = Linear(64 → 32) → Tanh → Linear(32 → 1)          # over [B,144,64]
score_t = mask ? score_t : −1e9                            # invalid windows suppressed
α = softmax_t(score)                                       # [B, 144], sums to 1 over valid windows
day_embedding = Σ_t α_t · h_t                              # [B, 64]
```
This lets the model **up-weight the windows that matter** (e.g. end-of-day deterioration) instead of
averaging all windows equally.

### 3.4 Classifier head
```
day_embedding[B,64] → Dropout(0.3) → Linear(64 → 1) → logit → sigmoid → prob
```
Total ≈ **53.3k parameters**.

### 3.5 Training
- **Balanced batches:** WeightedRandomSampler oversamples the minority (deaths) ≈ 50/50 per batch.
- **Loss:** BCEWithLogits. **Optimizer:** AdamW (lr 3e-4, weight-decay 1e-4). **Grad clip:** 5.0.
- **Early stopping** on validation AUROC (patience 15, max 120 epochs).
- **5 seeds → average the 5 probability vectors** (kills single-seed luck; per-seed sd ≈ 0.02).

---

## 4. Ensemble / meta-layer
File: `ensemble/ensemble.py`.

**Inputs:** each base model's **val + test probability** vectors, row-aligned by circuit-day.
**Candidate pool:** classical models in both **base-133** and **rich-200** variants + the
attention-CNN1D. Rank candidates by **val AUROC**; keep the DL model + top-3 classical families.

**Three combiners** (all thresholds/weights fit on VAL only):
| combiner | how | note |
|---|---|---|
| **mean** | average base probabilities | no fitting → most robust → **reported headline** |
| logistic stack | LogisticRegression on base probs, fit on VAL | learns weights; can overfit small VAL |
| best-by-val | keep single highest-val base model | noisy with ~30–50 val positives |

Decision threshold for F-scores = max-F1 point on the VAL precision-recall curve.

---

## 5. Evaluation protocol (shared)
- **One score per circuit-day**, computed on the **held-out TEST set** only.
- **Metrics:** AUROC, AUPRC, F1, F2, F3, Precision, Recall.
- **95% CI = bootstrap over PATIENTS** (1000 resamples; resample patients, not days — correlated days
  from one patient must not inflate confidence).

**Leakage barrier (what is fit on what):**
| data split | what it fits |
|---|---|
| TRAIN | impute medians, scaler, mRMR selection, HP tuning, CNN encoder weights |
| VAL | meta-learner weights, decision threshold, early-stopping, model selection |
| TEST | nothing — reported metrics only |

**Headline results (machine-only):**
| cohort | combiner | AUROC | 95% CI |
|---|---|---|---|
| 1-day | ensemble mean | 0.649 | [0.576, 0.723] |
| 1-day | rich-classical single (GB/RF) | 0.636 | [0.573, 0.706] |
| 1-day | attention-CNN1D (5-seed) | 0.617 | [0.542, 0.692] |
| 2-day | best (RF / CNN) | ~0.66 | [0.59, 0.72] |
Baselines (parent, no enrichment): 1-day classical 0.594, 1-day CNN1D 0.606.

---

## 6. Diagram cheat-sheet (boxes & shapes)

| # | stage | tensor / object | key params |
|---|---|---|---|
| 1 | raw per-minute | `[4, 1440]` + presence | 4 raw channels |
| 2 | gap-fill + derive | `X[6, 1440]` + `mask[1440]` | gap ≤10 interp, derive TMP/PFD |
| 3a | Branch A normalize | physical units | NaN kept per-feature |
| 3b | Branch B normalize | z-scored, masked→0 | train stats |
| 4a | A: features | `200`-vector / day | 133 parent + 67 new |
| 5a | A: select + models | `30`-vector → 5 probs | median-impute→scale→mRMR30→5 ML×5 seeds |
| 4b | B: window reduce | `[144, 30]` + `mask[144]` | k=10, 5 stats×6 ch |
| 5b | B: conv encoder | `[B, 64, 144]` | 3×(Conv k5 →BN→ReLU→Drop0.3) |
| 6b | B: attention pool | `[B, 64]` | Linear64→32→Tanh→1, softmax over windows |
| 7b | B: head | logit → prob | Dropout0.3 → Linear64→1 |
| 8 | ensemble | 1 prob / day | mean / logit-stack / best-by-val (fit on VAL) |
| 9 | evaluate | AUROC + CI | day-level, patient bootstrap ×1000 |

**Compact flow:**
`raw[4,1440] → gapfill+derive [6,1440]+mask → { A: 200 feats → mRMR30 → 5 models }  and
{ B: reduce[144,30] → CNN[64,144] → attn-pool[64] → logit } → ensemble(mean/stack) → AUROC`
