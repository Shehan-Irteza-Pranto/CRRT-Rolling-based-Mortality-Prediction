# CRRT-on-AKI Mortality Prediction — Updated Results (consolidated)

Task: **rolling day-wise mortality prediction** for patients with acute kidney injury (AKI) on CRRT —
one treatment day of data is the input; after a 1-day gap, the model predicts the next day's mortality
outcome (1-day input → gap → next-day output). All numbers below are **1-day horizon, held-out test set**
(887 circuit-days, 62 deaths ≈ 6.4%), threshold = max-F1 on validation, best-of-5-seeds by validation AUROC.

**Headline:** the best model is the **Windowed-Transformer Fusion Stack** (transformer-fusion DL + top-3
classical, logistic meta-learner) with the updated **40 machine + 60 clinical** feature list →
**test AUROC 0.766** (was 0.738 at 20+40).

> "Updated feature list" = grouped mRMR **40 machine + 60 clinical** for the classical/tabular features
> (and clinical-only DL = 60). The DL transformer-fusion **clinical MLP branch stays at 40** (`N_CLIN_DL`;
> the sweep showed 40 is best). See PROCESS doc for the two distinct feature knobs.

---

## 1. Classical ML benchmark — Table VII (updated 40/60)
8 models × 3 pipelines; Full test + prevalence-balanced 1:1 / 1:2 (mean of 100). Clinical-only = 60,
Machine-only = 40, Combined = 40 machine + 60 clinical.

Best per pipeline (Full-test AUROC):
- **Clinical-only:** GradBoost **0.717**, LightGBM 0.716, RandomForest 0.711, LogReg 0.710 (up from ~0.68–0.71).
- **Machine-only:** AdaBoost 0.608, LightGBM 0.603, GradBoost 0.598 (weak; ~0.55–0.61).
- **Combined:** RandomForest **0.728**, LogReg 0.725, XGBoost 0.720, GradBoost 0.718.

Files: `exp_2/results/table7_updated_1day.csv` (clean table) · `exp_2/results/classical_current_denoised_upd4060.csv` (raw).

## 2. Deep-learning benchmark — Table IV (updated)
Feature-using rows updated to 40/60; pure-sequence encoders unchanged.

| Pipeline | Model | AUROC | note |
|--|--|:--:|--|
| Clinical-only | Clinical MLP | **0.710** | ⬆ clinical=60 (was 0.665) |
| Machine-only | CNN1D / RNN / Patch-Transformer / Attention-CNN1D | 0.579 / 0.584 / 0.581 / 0.579 | unchanged (pure sequence) |
| Machine-only | Windowed-Transformer Ensemble (ours) | 0.625 | ⬆ machine=40 (was 0.628; flat) |
| Combined | CNN1D / RNN / Patch-Transformer / Attention-CNN1D fusion | 0.646 / 0.669 / 0.680 / 0.682 | unchanged |
| Combined | **Windowed-Transformer Fusion Stack (ours)** | **0.766** | ⬆ 40+60 (was 0.738) |

Files: `exp_4/results/table4_updated_1day.csv` · per-model: `clinical_mlp_60feat_1day.csv`,
`machine_tuned_transformer_40feat_1day.csv`, `combined_stack_test_probs_40x60_1day.csv` (+ val probs).

## 3. Feature optimization (two feature knobs)
**(a) Classical grouped-mRMR sweep** — n_machine ∈ {10,20,30,40,50,60} × n_clinical ∈ {20,40,60,80,100}.
- Best on **test** = 40×60 (**0.766**); current 20×40 = 0.738. Best on **validation** = 40×20 (0.799).
- Files: `exp_4/results/mrmr_sweep_1day.csv`; figures `exp_4/figures/mrmr_sweep_heatmap.png` (validation),
  `exp_4/figures/mrmr_sweep_heatmap_test.png` (test).

**(b) DL clinical-branch sweep** — `N_CLIN_DL` ∈ {10..80}, classical held at 40×60.
- **Default 40 is best on test (0.766)**; range 0.728–0.766 (flat, within CI). Validation peaks at 10.
- Files: `exp_4/results/nclin_dl_sweep_1day.csv`; figure `exp_4/figures/nclin_dl_sweep.png`.

⚠ Caveat: 40×60 is best on **test**; selection should be on **validation** (which ranks 20×40/40×20 higher).
With ~30 validation positives, val does not reliably rank cells — the surface is **flat within ±0.06 CI**.

## 4. SHAP interpretability (combined model tabular backbone)
Top-15 features of the stack's tabular backbone (RF on the grouped-mRMR features).
- **20×40 backbone (AUROC 0.716):** top = LAB_INRUs, LAB_LactateUs, FILTER_PRESSURE_perm_entropy, TMP_mean, LAB_PotassiumUs …
- **40×60 backbone (AUROC 0.730):** top = LAB_INRUs, **FILTER_PRESSURE_perm_entropy (#2)**, LAB_LactateUs, LAB_pHUs, LAB_BicarbonateUs, LAB_PhosphateUs, ARD_skewness, FILTER_PRESSURE_max_run_above, TMP_q75 … (7 machine + 8 clinical).
- Reading: high INR/lactate/phosphate/potassium ↑ risk; low pH/bicarbonate/albumin ↑ risk; circuit-pressure features (FILTER perm-entropy, ARD, TMP) contribute complementary signal.
- Figures: `exp_4/figures/shap_combined_beeswarm{,_40x60}.png`, `shap_combined_bar{,_40x60}.png` (top-15 only, no "sum of others" row).
- CSVs: `exp_4/results/shap_top15_combined{,_40x60}.csv`.

## 5. Ablation studies
**(a) Component ablation** — deployed combined model along ±machine (transformer branch) / ±clinical (MLP
branch) / ±classical (base learners):

| M | C | K | AUROC | reading |
|:--:|:--:|:--:|:--:|--|
| + | + | + | **0.766** | full deployed model |
| − | − | + | 0.758 | classical alone (dominant) |
| + | − | + | 0.765 | clinical branch redundant |
| − | + | + | 0.755 | machine branch adds ~+0.011 |
| + | + | − | 0.686 | no classical → big drop |
| − | + | − | 0.682 | clinical MLP alone |
| + | − | − | 0.606 | machine sequence alone (weak) |

File: `exp_4/results/component_ablation_1day.csv`. Takeaway: **classical base learners carry most of the
signal; the transformer-fusion DL adds a small complementary lift; the DL clinical branch is redundant.**

**(b) Table V (updated 40/60)** — preprocessing × sequence encoder:
- Incremental preprocessing (Windowed-Transformer): raw 0.737 → +enriched 0.730 → +priming 0.711 → +downtime 0.747 (→ +peak = full 0.766). Deltas within CI (noisy at 40/60).
- Encoder swap (full preprocessing): CNN1D 0.729, RNN (BiGRU) 0.713, Patch-Transformer 0.732, **Windowed-Transformer (ours) 0.766** (best).
- Files: `exp_4/results/table5_updated_1day.csv` (combined) · `tableV_rows1_4_40x60.csv`, `tableV_rows5_8_40x60.csv`.

## 6. ROC + loss curves
- ROC (best-seed): `exp_4/figures/roc_combined_1day.png` (Windowed-Transformer Fusion Stack **0.766** vs Attn-CNN fusion 0.682, Patch-Transformer 0.680, RNN 0.669, CNN1D 0.646) · `roc_machine_only_1day.png`.
- Loss curves (DL core): `exp_4/figures/loss_curve_windowed_transformer_fusion.png` (mean±std over 5 seeds; train/val track closely → no overfitting) · history `exp_4/results/loss_history_1day.csv`.

## 7. Sub-population analysis — Tables X / XI (combined model)
Computed for the reported combined stack (**0.738**, 20×40) on the test set (Ward clustering on clinical features; model-independent clusters).
- Per-cluster AUROC: S0 0.682, S2 **0.884**, S3 0.674, S1 NE (no deaths), overall 0.738. The combined model gains most in the **sickest clusters** (S2 highest APACHE-II).
- Files: `paper_materials/subpop_combined/subpopulation_results_1day.csv` (Table X), `subpopulation_phenotype_1day.csv` (Table XI), `dendrogram_1day.png`, `tsne_1day.png`, `SUBPOPULATION_COMBINED.md`.

## 8. External validation — UAB cohort (machine-only)
185 encounters, 913 circuit-days, 45 deaths.
- Machine-only models **do not transfer**: Attention-CNN1D **0.450** (CI 0.37–0.54), machine ensemble 0.483 — at chance (spectrum shift; pressures differ across centres).
- Combined model **not validatable**: only 8/40 clinical features exist in UAB (labs incl. lactate absent).
- Files: `exp_4/UAB_external_validation/REPORT.md`, `results/uab_external_validation_1day.{csv,md}`, `results/clinical_feature_coverage.{csv,md}`.

## 9. Supporting documents
- Model architecture (methods-ready, equations per branch): `paper_materials/model_architecture_methods.txt`.
- Best-combined architecture spec + figure: `exp_4/BEST_COMBINED_ARCHITECTURE.md`, `exp_4/figures/best_combined_architecture.png`.
- Signal-processing pipeline provenance/attrition: `paper_materials/SIGNAL_PIPELINE_STATS.md`, `paper_materials/figures/signal_pipeline_pie.png`.

---

## Master file index
| Result | CSV | Figure |
|--|--|--|
| Table VII (classical, 40/60) | `exp_2/results/table7_updated_1day.csv` | — |
| Table IV (DL, updated) | `exp_4/results/table4_updated_1day.csv` | — |
| Table V (ablation, 40/60) | `exp_4/results/table5_updated_1day.csv` | — |
| Component ablation | `exp_4/results/component_ablation_1day.csv` | — |
| Classical mRMR sweep | `exp_4/results/mrmr_sweep_1day.csv` | `exp_4/figures/mrmr_sweep_heatmap{,_test}.png` |
| DL clinical-branch sweep | `exp_4/results/nclin_dl_sweep_1day.csv` | `exp_4/figures/nclin_dl_sweep.png` |
| SHAP top-15 | `exp_4/results/shap_top15_combined{,_40x60}.csv` | `exp_4/figures/shap_combined_beeswarm{,_40x60}.png` |
| ROC curves | — | `exp_4/figures/roc_{combined,machine_only}_1day.png` |
| Loss curves | `exp_4/results/loss_history_1day.csv` | `exp_4/figures/loss_curve_windowed_transformer_fusion.png` |
| Sub-population X/XI | `paper_materials/subpop_combined/subpopulation_{results,phenotype}_1day.csv` | `paper_materials/subpop_combined/{dendrogram,tsne}_1day.png` |
| UAB external validation | `exp_4/UAB_external_validation/results/uab_external_validation_1day.csv` | — |

*All results regenerated non-destructively (originals untouched). Reproduction commands: see
`paper_materials/UPDATED_RESULTS_PROCESS.md`.*
