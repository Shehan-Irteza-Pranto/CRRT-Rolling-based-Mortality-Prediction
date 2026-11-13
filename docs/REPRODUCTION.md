# Reproducing the paper

Paths below are relative to `Baseline_ML_Models_v2/abalation_study_v2/` unless they start with `Code/`,
`Deep_Learning_Models/` or `Clinical_Only_Models/`. Inputs are described in `DATA_PROCESSING.md`. The repository ships code only; every output path below is created when the script runs.

Seeds are fixed in the scripts (model seeds 0 to 4; the 100 evaluation resamples use `RandomState(SEED0 + i)`), so a
re-run reproduces the published numbers up to GPU non-determinism in the transformer.

## 1. Run order (1-day-gap horizon)

```bash
conda activate crrt
# Stage 0: notebooks, run once (signal cleaning, labels, clinical tables)
#   Code/Denoising/CRRT_stream_denoising.ipynb
#   Code/label_processing/label_1day_2day_7_day.ipynb
#   Baseline_ML_Models_v2/Clinical_Only_Models/prepare_clinical_only_data.ipynb
cd Baseline_ML_Models_v2/abalation_study_v2
export ABL_NJOBS=4                                       # classical tuning workers

# Stage 1: splits, 133-feature tables, DL sequence cache, clinical table
python preprocessing/build_splits.py
python refresh_data.py --variants current --peaks denoised noised --cases 1day 2day --force
python preprocessing/build_clinical.py

# Stage 2: base classical + pure-sequence deep models (Table III, rows a)
python run_benchmark.py --variant current --peak denoised --scopes per_day
python run_dl_benchmark.py --gpus 0 1 --variant current --peak denoised --seeds 5

# Stage 3: exp_2, fixed grouped-mRMR classical benchmark, base SHAP, Table IV
python exp_2/run_benchmark.py && python exp_2/make_shap.py
python exp_2/run_table7_updated.py                      # Table IV

# Stage 4: exp_3, 200 features, attention-CNN, combined CNN-fusion stack
python exp_3/features/build_rich_features.py --case 1day
python exp_3/features/classical_rich.py --case 1day
python exp_3/dl/train_eval.py --case 1day
python exp_3/ensemble/ensemble.py --case 1day
python exp_3/results/combined_fusion_ensemble.py
python exp_3/results/make_resampled_variants.py

# Stage 5: exp_4, Windowed-Transformer Fusion Stack and the paper's tables and figures
python exp_4/build_pathA_bestseed.py
python exp_4/rerun_machine_bestseed.py && python exp_4/rerun_combined_bestseed.py
python exp_4/assemble_table.py                           # Table III skeleton
python exp_4/hp_search_transformer.py                    # 12-configuration search -> tuned transformer
python exp_4/machine_tuned_transformer_stack.py
python exp_4/stack_with_tuned_transformer.py
python exp_4/regenerate_stack_test_probs.py --n_machine 40 --n_clinical 60   # the final model
python exp_4/rerun_clinical_mlp_60.py
python exp_4/rerun_machine_ensemble_40.py
python exp_4/assemble_table4_updated.py                  # Table III
python exp_4/sweep_mrmr_stack.py && python exp_4/draw_mrmr_heatmap.py --metric test   # Fig 5
python exp_4/shap_best_combined.py --n_machine 40 --n_clinical 60                     # Table VI, Fig 7
python exp_4/loss_curve_fusion.py && python exp_4/replot_loss_curve.py               # Fig 6
python exp_4/tableV_updated_40x60.py                     # Table V rows 5-8

# Stage 6: exp_5, Table V rows 1-4 (preprocessing ladder)
python exp_5_ablation/gen_variants.py
python exp_5_ablation/tableV_rows1_4_40x60.py

# Stage 7: cohort and pipeline statistics (Table I, Sec. II-A, Fig. 2 numbers)
python paper_materials/make_cohort_tables.py
python paper_materials/signal_pipeline_stats.py && python paper_materials/draw_pipeline_pie.py

# Supplementary: EHR-only stack and patient-clustered bootstrap confidence intervals
python exp_4/ehr_only_stack.py
python exp_4/bootstrap_ci.py && python exp_2/bootstrap_table4.py
```

Run deep-learning jobs one at a time on a shared GPU node; two GPU jobs plus classical tuning exhaust memory.
`run_all.sh` wraps Stages 1 and 2 (resumable; `--quick` for a smoke test, `--fresh` to start over).

## 2. Paper item, script, output

### Cohort and labels (Sec. II, Table I)

| paper item | script | output |
| --- | --- | --- |
| Rolling 1-day label (input day t, gap t+1, target t+2) | `Code/label_processing/label_1day_2day_7_day.ipynb` | `Data/Processed/Labels/labels_1day.csv` |
| Patient-level stratified split 682 / 98 / 196 | `preprocessing/build_splits.py` | `data/splits/pid_splits_v2.json` |
| Cohort counts (976 patients, 1,153 episodes, 4,585 circuit days) | `paper_materials/make_cohort_tables.py` | `paper_materials/cohort/cohort_counts.csv` |
| Table I baseline characteristics | `paper_materials/make_cohort_tables.py` | `paper_materials/cohort/table1_baseline.csv` |
| Clinical feature tables | `Clinical_Only_Models/prepare_clinical_only_data.ipynb`, `preprocessing/build_clinical.py` | `Data/Processed/Model_Ready/clinical_only/`, `data/clinical_only/` |

### Signal acquisition and cleaning (Sec. III-A, Fig. 2)

| paper item | script | output |
| --- | --- | --- |
| Alignment with 12-h buffer (Eq. 1), priming (Eq. 2) and downtime (Eq. 3) removal, 14-day cap, 40-min peak detection and smoothing (Eq. 4) | `Code/Denoising/CRRT_stream_denoising.ipynb`; downtime-threshold variants in `CRRT_stream_denoising_abalation.ipynb` | `Data/Processed/Machine/pressure/{denoised,noised}/` |
| Same cleaning as importable functions | `preprocessing/build_streams.py`, `preprocessing/smoothing.py` |  |
| Attrition per stage and valid-minute statistics | `paper_materials/signal_pipeline_stats.py` | `paper_materials/cohort/signal_pipeline_*.csv` |
| Fig. 2 sources | `paper_materials/draw_pipeline_pie.py`, `draw_pipeline_funnel.py` | `paper_materials/figures/` |

### Data structuring, features, views (Sec. III-B, III-C1, Table II)

| paper item | script |
|---|---|
| Gap-aware reconstruction (interpolate up to 10 min, mask longer; 1440-min grid), derived TMP / PFD / ARD (Eq. 5) | `exp_3/features/day_signal.py`, `preprocessing/build_sequences.py`, `abl_config.py` |
| 133 base machine features (13 distribution + 6 temporal statistics × 7 channels) | `preprocessing/build_features.py` via `refresh_data.py` |
| 67 enriched features, 200 in total | `exp_3/features/build_rich_features.py`; feature definitions in `paper_materials/MACHINE_ENHANCEMENT.md` |
| Grouped mRMR (40 machine + 60 clinical; 60 clinical-only; 40 machine-only) | `models/feature_select.py`; budget in `exp_2/run_table7_updated.py` and `abl_config.py` |
| Sequence view 144 × 30 (10-min windows × 5 statistics × 6 channels) (Eq. 6) | `models/dl_reduce.py`, `exp_3/dl/dl_data.py` |
| Clinical vector view for the DL branch (Eq. 7) | `exp_3/results/combined_fusion_ensemble.py` (`N_CLIN_DL`) |


### Models (Sec. III-C2 to C4, Fig. 3)

| paper item | script |
|---|---|
| Transformer encoder with [CLS] token, sinusoidal positions, key-padding mask (Eq. 9) | `Deep_Learning_Models/models/transformer.py` |
| CNN1D, BiGRU, fusion wrapper | `Deep_Learning_Models/models/{cnn1d,rnn,fusion}.py` |
| Attention-CNN1D | `exp_3/dl/model.py`, `exp_3/dl/train_eval.py` |
| Clinical MLP branch and clinical-only MLP (Eq. 10) | `Deep_Learning_Models/models/mlp.py`, `models/dl_clinical_mlp.py` |
| Transformer-fusion network, tuned configuration (d = 64, 1 layer, 4 heads, feed-forward 128, dropout 0.4) from 12 candidates | `exp_4/hp_search_transformer.py` → `exp_4/results/hp_transformer_1day.csv`  |
| Classical base learners, RandomizedSearchCV 30 iterations × 5-fold, 5-seed refit | `models/classical.py`, grids in `Machine_Clinical_Models/config.py`, `exp_3/features/classical_rich.py` |
| Logistic-regression meta-learner fit on validation, max-F1 threshold (Eq. 12, 13) | `exp_4/stack_with_tuned_transformer.py`, `exp_4/ablation_ensemble_branch.py`, `exp_3/results/combined_fusion_ensemble.py` |
| Training loop: class-weighted BCE, weighted sampler, AdamW 3e-4 / weight decay 5e-3, gradient clip 5, batch 64, early stopping on validation AUROC (patience 15 within a 120-epoch budget), best of 5 seeds | `exp_4/hp_search_transformer.py`, `exp_4/stack_with_tuned_transformer.py` |
| Full / 1:1 / 1:2 evaluation protocol (100 resamples, shared seeds) | `eval/resampled_eval.py`, `exp_3/results/make_resampled_variants.py` |
| Fig. 3 architecture drawing | `exp_4/draw_architecture.py` |


### Results (Sec. IV)

| paper item | script → output |
| --- | --- |
| Fig. 4 peak-detection benchmark | not included in this release (`figures/Fig4_peak_detection_benchmark.png` only) |
| Fig. 5 grouped-mRMR budget sweep | `exp_4/sweep_mrmr_stack.py` → `exp_4/results/mrmr_sweep_1day.csv`; `exp_4/draw_mrmr_heatmap.py --metric test` |
| Table III deep-learning benchmark | `run_dl_benchmark.py --seeds 5`, `exp_4/build_pathA_bestseed.py`, `exp_4/rerun_machine_bestseed.py`, `exp_4/rerun_combined_bestseed.py`, `exp_4/assemble_table.py`, `exp_4/rerun_clinical_mlp_60.py`, `exp_4/machine_tuned_transformer_stack.py`, `exp_4/rerun_machine_ensemble_40.py`, `exp_4/regenerate_stack_test_probs.py --n_machine 40 --n_clinical 60`, `exp_4/assemble_table4_updated.py` → `exp_4/results/table4_updated_1day.csv` |
| Table IV classical benchmark (8 models × 3 pipelines) | `exp_2/run_table7_updated.py` → `exp_2/results/table7_updated_1day.csv` |
| Table V rows 1 to 4, preprocessing ladder | `exp_5_ablation/gen_variants.py`, `exp_5_ablation/tableV_rows1_4_40x60.py` |
| Table V rows 5 to 8, encoder swap | `exp_4/tableV_updated_40x60.py` |
| Fig. 6 train / validation loss over 5 seeds | `exp_4/loss_curve_fusion.py`, `exp_4/replot_loss_curve.py` |
| Table VI and Fig. 7 SHAP (enriched 200-feature model) | `exp_4/shap_best_combined.py --n_machine 40 --n_clinical 60` |
| Table VI SHAP (base 133-feature model) | `exp_2/make_shap.py` |
| Component ablation and clinical-branch width sweep | `exp_4/component_ablation.py`, `exp_4/sweep_nclin_dl.py` |
| EHR-only stack (supplementary) | `exp_4/ehr_only_stack.py`; report `exp_4/EHR_ONLY_STACK.md` |
| Patient-clustered bootstrap CIs and paired differences (supplementary) | `exp_4/bootstrap_ci.py`, `exp_2/bootstrap_table4.py`; report `exp_4/BOOTSTRAP_CI.md` |

The process notes kept with the code (`paper_materials/UPDATED_RESULTS_PROCESS.md`, `UPDATED_RESULTS.md`) describe
how the final 40 × 60 numbers were produced step by step.

## 3. Generated folders (ignored by git)

`abalation_study_v2/data/`, `abalation_study_v2/results/`, `exp_*/data/`, `exp_*/results/`, `exp_*/figures/`,
`exp_5_ablation/{data,work}/`, `exp_4/models/` (saved model bundles; see `exp_4/LOAD_AND_PREDICT.md`).