# Pipelines Explained — the 13 rows in `dl_vs_classical`

The comparison report ranks rows labelled `machine_clinical_agg_day`,
`mlp/combined`, `cnn1d/machine`, etc. This document explains each one: what data
it uses, how classical and deep variants relate, and what the suffixes mean.

Every pipeline is the **same prediction task** (CRRT-on-AKI mortality, same
patient-level splits, same balanced-test + validation-threshold protocol). They
differ only along two axes plus the model family.

---

## The two axes that explain every label

### Axis 1 — feature source: `clinical` / `machine` / `combined`
- **clinical** — lab values + PTI (patient demographics / comorbidities). No
  pressure data.
- **machine** — CRRT circuit pressure. ⚠️ **This exists in two completely
  different forms**, and that distinction is the whole point of the study:
  - *classical* `machine` = **52 engineered statistics** (4 pressure signals ×
    13 stats: mean, std, min, max, median, q25, q75, IQR, range, skewness,
    kurtosis, CV, n_valid).
  - *deep sequence* `machine` = the **raw `[T, 4]` waveform** itself — the
    per-minute pressure streams, *not* the 52 stats.
- **combined** — machine + clinical together.

### Axis 2 — temporal scope: per-day vs `_agg_day`
- **per-day** (no suffix) — features from a **single circuit-day**.
- **`_agg_day`** — **cumulative / expanding-window** features across Days 1..D,
  so the model sees the *trajectory* up to the current day (e.g. lab
  mean/min/max/median; pressure expanding mean/std/min/max), not just today.

So `machine_clinical_agg_day` = combined features, aggregated over the patient's
day-1-to-D history.

---

## Classical pipelines (6)

sklearn / gradient-boosting family. Each pipeline tunes **8 algorithms** (KNN,
LogisticRegression, DecisionTree, RF, GradientBoosting, AdaBoost, LightGBM,
XGBoost); the report shows the **single best model** per pipeline.

| Label | Features | Notes |
|---|---|---|
| **clinical_only** | lab + PTI, single day | clinical baseline, no pressure |
| **machine_only** | 52 pressure stats, single day | pressure-only; `full` (all 52) or mRMR top-20 |
| **machine_clinical** | 52 machine + clinical, single day | combined, single day |
| **clinical_agg_day** | labs aggregated over Days 1..D (92) + PTI (112) → 204 feats, mRMR top-40 | clinical trajectory |
| **machine_agg_day** | pressure stats expanding-window over Days 1..D → 208 feats, `full` or mRMR top-20 | pressure trajectory |
| **machine_clinical_agg_day** | 52 machine + 92 lab + 112 PTI → 256 feats, mRMR grouped (top-20 machine + top-40 clinical) | **the headline number to beat — GB AUROC 0.801 @ 1day** |

---

## Deep pipelines (7)

From `Deep_Learning_Models`, reported as **mean ± std over 5 seeds**. The label
is `model/branch`.

| Label | What it is | Classical analogue |
|---|---|---|
| **mlp/clinical** | MLP on the tabular clinical features | clinical_only |
| **mlp/machine** | MLP on the **52 engineered pressure stats** | machine_only — *direct head-to-head vs the GBMs on identical features* |
| **mlp/combined** | MLP on combined machine+clinical tabular features | machine_clinical |
| **cnn1d/machine** | 1D-CNN on the **raw `[T,4]` streams** | — (no classical equivalent; this *is* the hypothesis test) |
| **rnn/machine** | BiLSTM / GRU on the raw streams | — |
| **transformer/machine** | Transformer encoder on the raw streams | — |
| **fusion/machine** | sequence encoder (raw streams) ⊕ clinical MLP, late-fused | machine_clinical — the headline DL test |

(See `MODEL_ARCHITECTURES.md` for the input→output trace of cnn1d / rnn /
transformer.)

---

## Two things that are easy to misread

1. **`/machine` does NOT mean the same thing across deep rows.**
   - `mlp/machine` → the **52 engineered stats** (tabular).
   - `cnn1d/machine`, `rnn/machine`, `transformer/machine`, `fusion/machine` →
     the **raw waveform**. For these, `--branch machine` is just the `train.py`
     default tag and is a **no-op** — sequence/fusion models always load the raw
     stream cache regardless of the branch label.

2. **The MLP rungs are baselines, not the headline.**
   They exist so the deep family meets the GBMs on *exactly the same features*
   (a fair "does a neural net beat the tree on the engineered table?" check).
   The actual contribution is the sequence rungs (cnn1d / rnn / transformer) and
   `fusion`: testing whether learning directly from the **raw pressure waveform**
   recovers mortality signal that the 52 hand-crafted summaries structurally
   discard.

---

*Sources: docstrings of `Clinical_Only_Models/`, `Machine_Only_Models/`,
`Machine_Clinical_Models/` (and their `*_agg_day/` variants)
`run_experiments.py`; `Deep_Learning_Models/training/train.py`,
`models/`, `config.py`.*
