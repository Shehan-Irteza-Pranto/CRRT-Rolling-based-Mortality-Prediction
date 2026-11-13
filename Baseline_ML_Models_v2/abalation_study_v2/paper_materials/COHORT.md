# Cohort Description & Statistics

Cohort, split, baseline-characteristics, and evaluation-set statistics for the paper. Numbers are
computed by `make_cohort_tables.py` (CSVs in `cohort/`).

---

## 1. Cohort overview

CRRT-on-AKI patients; the prediction unit is a **circuit-day** (one CRRT circuit on one calendar day
for one patient), restricted to the first 7 treatment days. Patients are split at the **patient
level** (stratified 70:10:20, positive-rate matched) so no patient appears in more than one split.

| case | split | patients | circuits | circuit-days | deaths | prevalence % |
|---|---|--:|--:|--:|--:|--:|
| 1day | train | 682 | 806 | 3236 | 201 | 6.21 |
| 1day | val | 98 | 124 | 462 | 30 | 6.49 |
| 1day | test | 196 | 223 | 887 | 62 | 6.99 |
| 2day | train | 682 | 806 | 3236 | 350 | 10.82 |
| 2day | val | 98 | 124 | 462 | 46 | 9.96 |
| 2day | test | 196 | 223 | 887 | 109 | 12.29 |

**Totals:** 976 patients, 1,153 CRRT circuits, 4,585 circuit-days. Median ≈ 4 treatment days per
circuit (test set: days 1–7 = 223 / 196 / 163 / 128 / 91 / 52 / 34 circuit-days).

**Two distinct "mortality" quantities — do not conflate:**
- **Patient-level mortality** (in-hospital death ever): 525 / 976 = **53.8 %** of patients
  (used to stratify Table 1).
- **Circuit-day prediction label** (rolling, with a 1-day blanking gap): from a circuit-day on date
  *D*, the **1-day** label = death on *D+2*, the **2-day** label = death on *D+2 or D+3*; the death
  day *D* and the gap day *D+1* are excluded so the task is genuine early warning. This is the
  modelling target — prevalence **~6 %** (1-day) / **~11 %** (2-day), the much lower, clinically
  realistic per-day event rate that drives the class imbalance the evaluation protocol addresses.

---

## 2. Table 1 — baseline characteristics at CRRT initiation, by mortality
Patient-level (one row per patient = earliest circuit-day), stratified by in-hospital mortality.
Continuous: **median [IQR]**, Mann–Whitney U *p*; binary: **n (%)**, χ² *p*.

| variable | survivors | non-survivors | p |
|---|---|---|---|
| N (patients) | 451 | 525 | |
| Age, years | 57.0 [48.0–67.0] | 63.0 [55.0–70.2] | <0.001 |
| Weight, kg | 90.0 [75.0–110.0] | 89.0 [75.6–107.0] | 0.732 |
| BMI, kg/m² | 30.1 [25.2–36.6] | 29.7 [25.2–36.1] | 0.944 |
| Charlson comorbidity score | 2.0 [1.0–3.0] | 2.0 [1.0–4.0] | <0.001 |
| APACHE-II (partial) | 9.0 [5.0–25.0] | 13.0 [5.0–25.0] | 0.037 |
| Lactate | 14.4 [9.0–24.5] | 15.1 [7.3–32.4] | 0.914 |
| Albumin | 2.8 [2.4–3.3] | 2.6 [2.1–3.1] | 0.002 |
| INR | 1.3 [1.1–1.7] | 1.7 [1.3–2.5] | <0.001 |
| Platelets | 133.5 [70.6–210.5] | 93.5 [50.0–164.8] | <0.001 |
| pH | 7.3 [7.3–7.4] | 7.3 [7.3–7.4] | 0.228 |
| Bicarbonate | 21.0 [17.0–24.0] | 20.0 [16.9–23.0] | 0.021 |
| Creatinine | 3.1 [2.1–4.8] | 2.7 [1.8–3.8] | <0.001 |
| Urea | 52.0 [32.5–76.2] | 49.8 [27.9–75.0] | 0.391 |
| Hemoglobin | 8.5 [7.6–10.2] | 8.6 [7.8–9.8] | 0.733 |
| Male sex | 205 (58.7%) | 238 (63.3%) | 0.237 |
| Diabetes mellitus | 135 (38.7%) | 142 (37.8%) | 0.859 |
| Congestive heart failure | 59 (16.9%) | 83 (22.1%) | 0.097 |
| Mild liver disease | 23 (6.6%) | 47 (12.5%) | 0.010 |
| Moderate/severe liver disease | 37 (10.6%) | 69 (18.4%) | 0.004 |
| Any tumour | 24 (6.9%) | 30 (8.0%) | 0.672 |
| Metastatic tumour | 9 (2.6%) | 14 (3.7%) | 0.505 |

**Reading it.** Non-survivors are older and sicker — higher Charlson and APACHE-II, **lower albumin
and platelets, higher INR**, more liver disease, lower bicarbonate (hepatic/coagulation failure and
metabolic acidosis). These are the same coagulation/perfusion markers that dominate the clinical
SHAP ranking (INR, lactate, albumin, platelets), explaining why clinical features carry most of the
predictive signal.

---

## 3. Evaluation test sets — full vs balanced 1:1 vs 1:2
The held-out test set is scored three ways: the **full (imbalanced)** set, and **prevalence-balanced
1:1 / 1:2** sets formed by keeping all positives and randomly drawing 1× or 2× as many negatives,
averaged over **100 resamples** (negatives sampled without replacement; the same seeds are shared
across models for paired comparison).

### Composition
| protocol | N | positives | negatives | prevalence % | resampling iters |
|---|--:|--:|--:|--:|--:|
| full (imbalanced), 1day | 887 | 62 | 825 | 6.99 | 1 |
| balanced 1:1, 1day | 124 | 62 | 62 | 50.00 | 100 |
| balanced 1:2, 1day | 186 | 62 | 124 | 33.33 | 100 |
| full (imbalanced), 2day | 887 | 109 | 778 | 12.29 | 1 |
| balanced 1:1, 2day | 218 | 109 | 109 | 50.00 | 100 |
| balanced 1:2, 2day | 327 | 109 | 218 | 33.33 | 100 |

### Best machine-only ensemble under each protocol
(exp_3 ensemble:mean; decision threshold = max-F1 on validation.)

**1-day** (val-F1 threshold 0.280)
| protocol | AUROC | AUPRC | F1 | F2 | F3 | Precision | Recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| full | 0.649 | 0.121 | 0.175 | 0.167 | 0.164 | 0.192 | 0.161 |
| 1:1 | 0.646 | 0.643 | 0.267 | 0.192 | 0.175 | 0.779 | 0.161 |
| 1:2 | 0.648 | 0.476 | 0.256 | 0.189 | 0.174 | 0.629 | 0.161 |

**2-day** (val-F1 threshold 0.298)
| protocol | AUROC | AUPRC | F1 | F2 | F3 | Precision | Recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| full | 0.643 | 0.211 | 0.153 | 0.143 | 0.140 | 0.172 | 0.138 |
| 1:1 | 0.646 | 0.638 | 0.225 | 0.163 | 0.149 | 0.615 | 0.138 |
| 1:2 | 0.645 | 0.473 | 0.209 | 0.159 | 0.148 | 0.437 | 0.138 |

**Reading it.** AUROC is essentially **invariant to prevalence** (0.646–0.649 across full/1:1/1:2),
confirming the rank-discrimination is real and not a class-ratio artifact. AUPRC and Precision rise
sharply as the negative pool shrinks (0.12 → 0.48 → 0.64 AUPRC at full → 1:2 → 1:1) — the expected
mechanical effect of raising prevalence, which is exactly why the balanced sets are reported
alongside the honest full-test number.

*Companion: per-pipeline full/1:1/1:2 numbers for clinical / machine / combined are in
`RESULTS_COMPENDIUM.md`; sub-population robustness of this ensemble is in `SUBPOPULATION.md`.*
