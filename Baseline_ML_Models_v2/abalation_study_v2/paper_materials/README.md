# paper_materials — draft assets for the CRRT-mortality paper

Everything needed to write up the study, in one place. Built from the experiments under
`abalation_study_v2/` (E1 benchmark, E2 SHAP, E3 machine-only enhancement, ablations, negative results).

## Contents
| file | what it is |
|---|---|
| **RESULTS_COMPENDIUM.md** | All results as paper-ready data tables: main pipeline benchmark (clinical / machine / combined × classical / DL), combined feature-selection, machine-only enhancement, denoise + leave-one-out ablations, SHAP top-15, negative results, and paste-ready conclusions. Every number transcribed from the source CSVs (paths cited per table). |
| **DL_METHODS.md** | Full architecture & training spec for **every** deep-learning model used (MLP, CNN1D, RNN BiLSTM/GRU, Transformer, fusion, attention-CNN1D, and the rejected MIL/fragment models): layer-by-layer dims, activations, pooling, loss functions, optimizer, and hyperparameter search space — for the paper's Methods section. |
| **MACHINE_ENHANCEMENT.md** | The machine-only enhancement (Table VIII): **what the 67 enriched features are** (spectral, entropy, excursion, end-of-day, coupling) and **how they differ from the 133 base features**, which 12 the selector kept, the results table, and a corrected Section-C paragraph. |
| **INTERPRETABILITY.md** | The manuscript **interpretability** section: mRMR selection across all 8 pipelines (cross-pipeline map + stability), SHAP attribution for the best classical model of each pipeline (base **and** the enriched circuit-instability features), phenotype reliability, and paste-ready paragraphs. Figures/tables in `interpretability/`. Built by `make_shap_enriched.py`, `exp_2/make_shap.py`, `make_interpretability_figs.py`. |
| **TABLE_VI_MODELS.md** | Row-by-row walkthrough of **every model in Table VI** (clinical MLP; machine CNN1D/RNN/Transformer/Attention-CNN1D/Ensemble; combined fusion CNN1D/RNN/Transformer/Attn-CNN-fusion/Stack) — input → processing → layers → output → **loss calculation**, plus the shared training protocol and leakage barrier. Paper-ready per-model descriptions. |
| **DATA_PROCESSING.md** | Equation-complete data & signal-processing spec: raw device export → cleaning (priming / downtime / 40-min z-score peak smoothing, with thresholds) → gap-aware day signal → derived channels → both feature branches (all 200 feature equations) → sequence reduction. The canonical preprocessing reference for the Methods section. |
| **SIGNAL_PROCESSING_PIPELINE.md** | The data-cleaning pipeline as stages (priming removal, downtime removal flat/dose, window-based peak smoothing, day cap, gap/missing handling) **and the preprocessing ablation knobs** (8 denoise-variant streams + 4 leave-one-out configs, with the models used). Cleaning + ablation narrative companion to DATA_PROCESSING.md. |
| **COHORT.md** | Cohort description & statistics: split counts (patients/circuits/circuit-days), **Table 1** baseline characteristics by mortality (median[IQR]/n(%) + p-values), and the **full / balanced 1:1 / 1:2** test-set composition and best-ensemble metrics. Built by `make_cohort_tables.py` (CSVs in `cohort/`). |
| **SUBPOPULATION.md** | Sub-population robustness of the best machine-only ensemble: test circuit-days clustered into clinical phenotypes (Ward, **auto-selected count**), per-subgroup AUROC/AUPRC, phenotype description, **t-SNE + dendrogram** (`subpop/`). Built by `subpopulation_analysis.py`. |
| **machine_only_pipeline.py** | One self-contained script reproducing the **best machine-only model** end to end: raw signal → gap-aware day signal → richer physiology features + classical models (Branch A) and window-reduced attention CNN1D (Branch B) → probability ensemble → day-level evaluation with patient-bootstrap CI. |
| **machine_only_pipeline_{1day,2day}.csv** | Output of the script (per-model + ensemble test metrics). Generated on run. |

### Additional result CSVs (in `../exp_3/`)
| file | what it is |
|---|---|
| `../exp_3/results/exp3_variants_full_1to1_1to2.csv` | Every machine-only exp_3 variant (classical + attention-CNN + ensembles) under **full / 1:1 / 1:2** test protocols. (Compendium §3c.) |
| `../exp_3/results/exp3_combined_variants_full_1to1_1to2.csv` | Same, for the **combined** model (machine-rich + 40 mRMR clinical features). (Compendium §3c.) |
| `../exp_3/results/exp3_combined_fusion_results.csv` | **Best combined-with-CNN**: CNN-fusion DL (attention-CNN machine + clinical branch) + combined-classical, stacked; full/1:1/1:2. Best 2-day = 0.742. (Compendium §3d.) |
| `../exp_3/ablation/README.md` + `anchored_loo.py` | **Anchored** leave-one-out of the best ensemble (all3 = headline 0.649/0.643, interpretable deltas). (Compendium §5b.) |

See also: `../exp_3/ARCHITECTURE.md` — the diagram-ready architecture spec (tensor shapes, layer
configs) for the machine-only pipeline; `../exp_3/RESULTS.md` — the E3 write-up.

## Reproduce the best machine-only model
```bash
conda activate crrt
cd Baseline_ML_Models_v2/abalation_study_v2/paper_materials
python machine_only_pipeline.py --case 1day      # or --case 2day
```
Prints a per-model + ensemble table and writes `machine_only_pipeline_<case>.csv`. Expect the
ensemble test AUROC ≈ 0.64–0.65 (1-day) with a patient-bootstrap 95% CI, matching exp_3.

Notes:
- Classical models use fixed, documented hyperparameters (the tuned operating point) for
  determinism; full RandomizedSearchCV tuning is in `../exp_3/features/classical_rich.py`.
- Leak-safe: impute / scale / mRMR / CNN weights fit on TRAIN, ensemble meta-learner + threshold on
  VAL, all reported metrics on the held-out TEST set; CIs bootstrap over patients.
- Reads parent data read-only; needs `mrmr`, `lightgbm`, `xgboost`, `torch` in the `crrt` env.

## Headline numbers (for the abstract)
- Clinical-only AUROC 0.71–0.73; **combined ≈ clinical** (0.728 / 0.738, fixed 20 machine + 40
  clinical features) — CRRT pressure data adds only a small increment once clinical data is present.
- Best **machine-only** model (rich-feature + attention-CNN ensemble): **0.649** (1-day),
  **~0.66** (2-day) — a +0.05 lift over the prior machine-only baseline (0.594).
- Lift driven by circuit-instability physiology (TMP spectral period, end-of-day deterioration,
  signal complexity); corroborated by SHAP (TMP / filter / effluent).
