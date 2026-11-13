# Deep-learning report — CRRT-on-AKI mortality (1-day horizon)

Deep-learning models only. All results are the **best-of-5-seeds by validation AUROC** on the held-out
test set (leak-safe model selection); threshold for F1/Precision/Recall = **max-F1 on validation**;
balanced 1:1 / 1:2 = mean over 100 prevalence-resamples. Two experiments: **Table 5** (the DL benchmark
per pipeline) and an **encoder-swap ablation** (best model with the sequence encoder replaced).

---

## 1. Process — from features to framework to training

### 1.1 Features (what each model sees)
All features derive from the same cleaned, gap-aware per-circuit-day record (Day ≤ 7). Three
representations feed the DL models:

- **Clinical (tabular, 40 features).** mRMR-selected top-40 from the patient record: laboratory values
  (INR, lactate, albumin, platelets, pH, bicarbonate, potassium, chloride, calcium, magnesium…),
  comorbidities and CRRT indications, and severity/demographics (age, APACHE-II, Charlson).
- **Machine — tabular (base 133 / enriched 200).** Per-channel statistics of the six pressure channels
  (ACCESS, FILTER, EFFLUENT, RETURN, TMP, PFD, + derived ARD): base = distribution + within-day dynamics
  (133); enriched = base + 67 circuit-instability descriptors (spectral, permutation entropy, sustained
  excursion, end-of-day deterioration, cross-channel coupling). Used by the classical stack members.
- **Machine — sequence `[144 × 30]`.** The raw 1440-min pressure waveform, **z-scored per channel**
  (train stats), **gap-masked**, and **window-reduced** into 144 non-overlapping 10-min windows, each
  summarized by 5 statistics (mean, min, max, std, slope) over the 6 channels → 30 features/window. This
  is the input to every sequence encoder (CNN / RNN / Transformer / Attention-CNN).

### 1.2 Framework (the models)
| pipeline | model | input → processing → output |
|---|---|---|
| **Clinical-only** | **MLP** | 40 mRMR clinical features → feed-forward MLP (dense + ReLU + dropout) → sigmoid → P(death) |
| **Machine-only** | **CNN1D** | sequence → 1-D conv stack → **masked global-mean pool** → linear head → sigmoid |
| | **RNN (BiGRU)** | sequence → bidirectional GRU → **masked-mean** over timesteps → head → sigmoid |
| | **Transformer** | sequence → window tokens + positional encoding → **multi-head self-attention** → pooled → head |
| | **Attention-CNN1D** | conv stack → **learned masked attention pooling** over windows → head |
| | **Ensemble** | probability combination of the top-3 classical machine models + the Attention-CNN1D |
| **Combined** | **CNN1D / RNN / Transformer** | machine sequence → encoder → machine embedding ⊕ clinical-MLP embedding (40 mRMR) → **late fusion** → joint head |
| | **Attn-CNN fusion** | Attention-CNN1D machine branch ⊕ clinical-MLP branch, concatenated → head |
| | **Ensemble/Stack** | **logistic meta-learner** over the top-4 components (fusion DL + combined-classical + clinical-classical), fit on validation |

Layer-level specs (dims, kernels, heads, dropout, optimizer) are in `paper_materials/DL_METHODS.md` and
`paper_materials/TABLE_VI_MODELS.md`; feature/sequence equations in `paper_materials/DATA_PROCESSING.md`.

### 1.3 Training strategy
- **Splits:** patient-disjoint train / validation / test (v2 split); one sample per circuit-day.
- **Leak-safe:** every imputer, scaler, mRMR selection, z-score statistic, and network weight is fit on
  **train**; the decision threshold on **validation**; all metrics on the held-out **test** set.
- **Optimization:** binary cross-entropy loss, AdamW, gradient clipping, **class-balanced minibatch
  sampler**, early-stopping on validation AUROC (best checkpoint restored).
- **Seeds / selection:** 5 seeds per model; report the **best-of-5 by validation AUROC** (leak-safe
  model selection) — chosen because, with ~30 validation positives, single-seed variance is large.
- **Evaluation:** full (imbalanced) test set + prevalence-balanced **1:1 / 1:2** (positives kept,
  negatives subsampled, 100 iterations); AUROC/AUPRC threshold-free; F1/Precision/Recall at the
  validation max-F1 threshold; uncertainty via patient-bootstrap 95% CI.

---

## 2. Table 5 — DL benchmark (1-day, best-seed)
Best AUROC per pipeline **bold**.

| Pipeline / Model | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ***Clinical-only*** | | | | | | | | | |
| MLP | **0.665** | 0.177 | 0.237 | 0.161 | 0.452 | 0.668 | 0.686 | 0.665 | 0.534 |
| ***Machine-only*** | | | | | | | | | |
| CNN1D | 0.579 | 0.095 | 0.026 | 0.071 | 0.016 | 0.575 | 0.585 | 0.579 | 0.415 |
| RNN (BiGRU) | 0.584 | 0.094 | 0.159 | 0.098 | 0.419 | 0.583 | 0.583 | 0.585 | 0.412 |
| Transformer | 0.581 | 0.086 | 0.123 | 0.099 | 0.161 | 0.578 | 0.559 | 0.582 | 0.389 |
| Attention-CNN1D | 0.579 | 0.102 | 0.165 | 0.155 | 0.177 | 0.578 | 0.601 | 0.579 | 0.430 |
| **Ensemble** | **0.622** | 0.109 | 0.131 | 0.156 | 0.113 | 0.621 | 0.620 | 0.621 | 0.449 |
| ***Combined*** | | | | | | | | | |
| CNN1D | 0.646 | 0.182 | 0.188 | 0.121 | 0.419 | 0.644 | 0.674 | 0.646 | 0.526 |
| RNN (BiGRU) | 0.669 | 0.222 | 0.278 | 0.302 | 0.258 | 0.667 | 0.698 | 0.669 | 0.559 |
| Transformer | 0.680 | 0.182 | 0.225 | 0.156 | 0.403 | 0.680 | 0.696 | 0.682 | 0.549 |
| Attn-CNN fusion | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.692 | 0.681 | 0.536 |
| **Ensemble/Stack** | **0.740** | 0.170 | 0.244 | 0.178 | 0.387 | 0.739 | 0.718 | 0.739 | 0.565 |

*Source: `results/table_vi_bestseed.csv`.*

---

## 3. Ablation — swapping the sequence encoder in the best model (1-day)
The best pipelines are built around a sequence encoder (machine Ensemble → Attention-CNN1D; combined
Ensemble/Stack → CNN-fusion). Here that encoder is replaced by **CNN1D (mean pooling)**, **CNN-attention**,
**RNN (BiGRU)**, and a **patch-Transformer**, under one identical recipe (window_agg `[144×30]`, 5-seed
best-val). Each is reported as the **standalone DL** and as the **ensemble** (= logistic stack of the DL +
top-3 classical). Best AUROC per section **bold**.

### Machine-only
| Encoder | Variant | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:2 AUROC |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CNN1D (mean pool) | DL | 0.590 | 0.111 | 0.206 | 0.172 | 0.258 | 0.590 | 0.591 |
| | Ensemble | 0.636 | 0.113 | 0.121 | 0.162 | 0.097 | 0.635 | 0.636 |
| CNN-attention | DL | 0.579 | 0.102 | 0.165 | 0.155 | 0.177 | 0.578 | 0.579 |
| | Ensemble | 0.621 | 0.109 | 0.104 | 0.147 | 0.081 | 0.619 | 0.620 |
| RNN (BiGRU) | DL | 0.566 | 0.114 | 0.161 | 0.101 | 0.387 | 0.564 | 0.566 |
| | **Ensemble** | **0.641** | 0.109 | 0.089 | 0.100 | 0.081 | 0.641 | 0.641 |
| patch-Transformer | DL | 0.618 | 0.098 | 0.168 | 0.106 | 0.403 | 0.614 | 0.618 |
| | Ensemble | 0.636 | 0.105 | 0.083 | 0.114 | 0.065 | 0.632 | 0.636 |

### Combined
| Encoder | Variant | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:2 AUROC |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CNN1D (mean pool) | Fusion-DL | 0.711 | 0.218 | 0.253 | 0.202 | 0.339 | 0.711 | 0.712 |
| | Ensemble | 0.719 | 0.184 | 0.261 | 0.197 | 0.387 | 0.720 | 0.720 |
| CNN-attention | Fusion-DL | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.681 |
| | Ensemble | 0.694 | 0.174 | 0.258 | 0.194 | 0.387 | 0.694 | 0.694 |
| RNN (BiGRU) | Fusion-DL | 0.664 | 0.185 | 0.204 | 0.132 | 0.452 | 0.663 | 0.664 |
| | Ensemble | 0.709 | 0.174 | 0.219 | 0.151 | 0.403 | 0.708 | 0.709 |
| patch-Transformer | **Fusion-DL** | **0.724** | 0.188 | 0.250 | 0.198 | 0.339 | 0.723 | 0.725 |
| | Ensemble | 0.708 | 0.188 | 0.249 | 0.180 | 0.403 | 0.704 | 0.707 |

*Source: `results/matrix_machine_1day.csv`, `results/matrix_combined_1day.csv` (full 1:1/1:2 AUPRC in the
CSVs). Note: these use the unified exp_3 recipe (window_agg 30-ch) and are internally comparable, but
differ from the Table 5 numbers above, which come from the main DL-benchmark pipeline (patch reduction /
HP search) — so compare **within** a table, not across.*

---

## 4. Findings
1. **Clinical ≫ machine, and combined is best.** Clinical MLP (0.665) far exceeds any machine-only DL
   (≤0.62); adding the machine sequence to clinical lifts the combined models to 0.65–0.74, topping out
   at the **Ensemble/Stack = 0.740**.
2. **Encoder ranking (standalone):** the **patch-Transformer** is the strongest sequence encoder in both
   pipelines (machine 0.618, combined 0.724); CNN1D mean-pool is a solid second in combined (0.711).
   Attention pooling and the RNN trail as standalone models.
3. **Ensembling raises AUROC toward the classical ceiling but not past it.** Machine ensembles reach
   0.62–0.64 (from DL 0.57–0.62); combined ensembles cluster at 0.69–0.72 — all pulled toward the
   dominant classical members. Recall compresses in the ensembles (few predicted positives), the
   expected effect of averaging conservative scores at ~7% prevalence.
4. **The top numbers are statistically tied.** With 62 test positives the patient-bootstrap CI is ≈±0.06,
   so combined patch-Transformer (0.724), CNN1D-mean ensemble (0.719), and the Ensemble/Stack (0.740) are
   indistinguishable — the combined ceiling is ~0.74, and the sequence-encoder choice does not move it.

*Reproduce: `matrix_machine_1day.py`, `matrix_combined_1day.py` (ablation); Table 5 from the exp_4
best-seed pipeline (`build_pathA_bestseed.py` + `rerun_*_bestseed.py` → `assemble_table.py`).*
