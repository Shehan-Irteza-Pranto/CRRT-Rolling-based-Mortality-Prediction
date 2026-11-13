# Best combined model — architecture (Transformer + 3-classical stacked ensemble)

The best combined model (1-day test AUROC **0.738**) is a **two-level stacked ensemble**: four Level-0
base learners feed a Level-1 logistic-regression meta-learner. Level-0 = one **Transformer-fusion deep
network** (machine sequence + clinical branches) plus the **top-3 classical models**; Level-1 combines
their probabilities. Figure: `figures/best_combined_architecture.png`.

```
        LEVEL-0 BASE LEARNERS                                   LEVEL-1 META-LEARNER
 ┌───────────────────────────────────────┐
 machine seq [144×30] ─▶ Transformer branch ─┐
                                              ├─▶ fusion head ─▶ p_DL ┐
 clinical [40 mRMR]   ─▶ clinical-MLP branch ─┘                       │
                                                                      ├─▶ [p_DL, p₁, p₂, p₃]
 20 machine + 40 clinical (tabular) ─▶ classical model #1 ─▶ p₁       │        │
                                    ─▶ classical model #2 ─▶ p₂       │   LogisticRegression
                                    ─▶ classical model #3 ─▶ p₃       ┘   (fit on VALIDATION)
                                                                               │
                                                                               ▼  P(death)
                                                                          ≥ max-F1-val threshold → label
```

## 1. Inputs (representations)
| Representation | Shape | Definition |
|---|---|---|
| Machine **sequence** | `[B, 144, 30]` | 1440-min pressure waveform → 144 non-overlapping **10-min windows**; each window = 5 stats (mean/min/max/std/slope) × 6 channels (ACCESS, FILTER, EFFLUENT, RETURN, TMP, PFD). z-scored per channel (train stats); gap windows masked. |
| **Clinical** vector | `[B, 40]` | mRMR top-40 clinical features (labs, comorbidities/indications, age/APACHE-II). |
| **Tabular (classical)** | `[B, 60]` | grouped mRMR: 20 machine (from the 200 enriched circuit features) + 40 clinical; median-imputed + standardized. |

## 2. Level-0 (A) — Transformer-fusion deep network
HP-tuned config: **d_model 64, 1 encoder layer, 4 heads, ff 128, dropout 0.4**.

**Machine branch — Transformer encoder** (`TransformerEncoderModel`)
| # | Layer | Output | Config |
|---|---|---|---|
| 1 | Linear input projection | `[B, 144, 64]` | 30 → 64 (per-window token embedding) |
| 2 | Prepend learnable **[CLS]** token | `[B, 145, 64]` | 1×64 parameter |
| 3 | + Sinusoidal positional encoding | `[B, 145, 64]` | added to all tokens |
| 4 | **TransformerEncoderLayer × 1** | `[B, 145, 64]` | 4 heads; FFN 64→128→64 (**GELU**); residual+LayerNorm; dropout 0.4; `batch_first`; **key-padding mask** drops gap windows (CLS always attended) |
| 5 | Take **[CLS]** state | `[B, 64]` | pooled **machine embedding** |

**Clinical branch — MLP**
| # | Layer | Output | Config |
|---|---|---|---|
| 1 | Linear | `[B, 64]` | 40 → 64 |
| 2 | BatchNorm1d | `[B, 64]` | |
| 3 | ReLU → Dropout(0.4) | `[B, 64]` | **clinical embedding** |

**Fusion head** (late fusion)
| # | Layer | Output | Config |
|---|---|---|---|
| 1 | Concatenate [machine ⊕ clinical] | `[B, 128]` | 64 + 64 |
| 2 | Linear → ReLU → Dropout(0.4) | `[B, 64]` | 128 → 64 |
| 3 | Linear | `[B, 1]` | 64 → 1 (logit) |
| 4 | Sigmoid | `[B]` | **p_DL** |

## 3. Level-0 (B, C, D) — top-3 classical models
- **Features:** the tabular `[B, 60]` (20 machine + 40 clinical mRMR), median-impute → standardize.
- **Candidates:** GradientBoosting, RandomForest, LightGBM, XGBoost, LogisticRegression — each tuned by
  `RandomizedSearchCV` and refit over 5 seeds. The **top-3 by validation AUROC** are kept.
- Each emits a probability `p₁, p₂, p₃`.

## 4. Level-1 — logistic meta-learner (stacking)
- Input = the 4 base probabilities `[p_DL, p₁, p₂, p₃]` (the DL fusion is **forced in**).
- `LogisticRegression(max_iter=5000)` **fit on the VALIDATION-set base probabilities** (leak-safe) →
  weighted combination → `P(death)`.
- Decision threshold = **max-F1 on validation**.

## 5. Training & selection (all DL components)
| Item | Setting |
|---|---|
| Loss | Binary cross-entropy (`BCEWithLogitsLoss`) |
| Optimizer | AdamW — **lr 3e-4, weight-decay 5e-3** |
| Class imbalance | `WeightedRandomSampler` (≈balanced minibatches) |
| Regularization | dropout 0.4, weight-decay, gradient clip 5.0 |
| Epochs / early stop | ≤120, early-stop on **validation AUROC** (patience 15), best checkpoint restored |
| Batch size | 64 |
| Seeds | 5 → report the **best seed by validation AUROC** |
| Leakage barrier | impute / scale / mRMR / network weights on **train**; meta-learner + threshold on **validation**; metrics on **test** |

## 6. Result
- **1-day test AUROC 0.738** (95% CI ≈ ±0.06, 62 test positives) — the best **DL-inclusive** combined
  model; statistically tied with the all-classical top-4 stack (0.740), which excludes the DL.

*Sources: `Deep_Learning_Models/models/transformer.py` (encoder); `exp_4/hp_search_transformer.py`
(`TFusion`, tuned HP); `exp_4/stack_with_tuned_transformer.py` (member selection + logistic stack);
`exp_3/results/combined_fusion_ensemble.py` (classical members, data tables).*
