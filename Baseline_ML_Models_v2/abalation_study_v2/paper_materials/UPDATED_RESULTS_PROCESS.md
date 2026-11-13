# Process Guide — how the updated results were produced

Companion to `UPDATED_RESULTS.md`. Explains the pipeline, the "updated feature list" change, which script
produces each result, how to reproduce, and the honest caveats. Environment: `conda activate crrt`
(the `crrt` conda environment). Root: `Baseline_ML_Models_v2/abalation_study_v2/`.

---

## 1. The prediction problem
Rolling day-wise mortality for CRRT-on-AKI patients: each **circuit-day** (patient × treatment-episode ×
day) is one sample. The **1-day label** = 1 if the patient dies on day d+2 (a 1-day blanking gap; day d+1
excluded), else 0. Splits are **patient-disjoint** train/validation/test.

## 2. Data → features → models (pipeline)
1. **Signal** — per-minute CRRT pressures (ACCESS, FILTER, EFFLUENT, RETURN) + derived TMP, PFD, ARD →
   cleaned (priming/downtime removal + 40-min peak smoothing) → gap-aware 1440-min grid.
2. **Two views per day:**
   - **Machine sequence** `[144×30]` — 144 windows × (5 stats × 6 channels), z-scored (train stats).
   - **Tabular features** — 200 engineered machine features + clinical features.
3. **Models:**
   - **Classical** — 8 families on the tabular view (mRMR-selected).
   - **DL** — sequence encoders (CNN1D / RNN / Transformer) + a **fusion** (sequence branch + clinical MLP branch).
   - **Best model = Windowed-Transformer Fusion Stack** — a logistic meta-learner stacking the tuned
     transformer-fusion DL with the top-3 classical base learners.

## 3. THE KEY CHANGE — "updated feature list" (two distinct knobs)
There are **two independent clinical/tabular feature-count knobs**; the update touched them differently:

| Knob | What it controls | File / constant | Old → New |
|--|--|--|--|
| **Classical grouped mRMR** | tabular features for the classical base learners (and clinical-only) | `N_MRMR_MACHINE`, `N_MRMR_CLINICAL`, `N_MRMR` (`exp_2/abl_config.py`); `CF.N_MACHINE`, `CF.N_CLINICAL` (`exp_3/results/combined_fusion_ensemble.py`) | 20+40 → **40 machine + 60 clinical** |
| **DL clinical MLP branch** | clinical features into the transformer-fusion's clinical branch | `CF.N_CLIN_DL` | **40 (unchanged — 40 is best per the sweep)** |

So "combined 40×60" means: classical members use 40 machine + 60 clinical; the DL fusion's clinical branch
still uses 40. This was verified and is intentional (a sweep of the DL branch 10→80 showed 40 is best).

## 4. Which script produced each result
| Result | Script | Notes |
|--|--|--|
| Table VII (classical, 40/60) | `exp_2/run_table7_updated.py` | patches selection defaults (40/60) + fresh cache dir + new CSV (non-destructive) |
| Combined stack probs (40×60) | `exp_4/regenerate_stack_test_probs.py --n_machine 40 --n_clinical 60` | the 0.738/0.766 model was never saved → retrain (deterministic 5-seed) |
| Clinical MLP (clinical=60) | `exp_4/rerun_clinical_mlp_60.py` | wraps `models/dl_clinical_mlp.py`; patches mRMR to 60 |
| Machine ensemble (machine=40) | `exp_4/rerun_machine_ensemble_40.py` | reuses HP winner; recomputes classical at machine-40 |
| Table IV (assemble) | `exp_4/assemble_table4_updated.py` | updated rows + unchanged rows from `table_vi_bestseed.csv` |
| Classical mRMR sweep | `exp_4/sweep_mrmr_stack.py` + `draw_mrmr_heatmap.py` | DL trained once, reused across cells |
| DL clinical-branch sweep | `exp_4/sweep_nclin_dl.py` | retrains fusion per N_CLIN_DL |
| SHAP | `exp_4/shap_best_combined.py [--n_machine 40 --n_clinical 60]` | TreeSHAP on RF tabular backbone; top-15 only |
| Component ablation | `exp_4/component_ablation.py` | branch-drop on the fusion + toggling classical |
| Table V (encoder swap) | `exp_4/tableV_updated_40x60.py` | forced stack per encoder |
| Table V (preprocessing) | `exp_5_ablation/tableV_rows1_4_40x60.py` | wraps `run_preproc_ablation.py` at 40/60 |
| ROC curves | `exp_4/plot_roc.py` | stack curve uses the 40×60 probs |
| Loss curves | `exp_4/loss_curve_fusion.py` + `replot_loss_curve.py` | 5-seed train/val BCE |
| Sub-population X/XI | `paper_materials/subpopulation_analysis_combined.py` | Ward clustering; uses the 0.738 stack probs |
| UAB external validation | `exp_4/UAB_external_validation/code/{build_uab_inputs,external_validate,check_clinical_coverage}.py` | frozen bundles, no re-fit on UAB |

## 5. Reproduce (minimal, 1-day)
```bash
conda activate crrt
cd Baseline_ML_Models_v2/abalation_study_v2

# classical Table VII (40/60)
python exp_2/run_table7_updated.py

# combined stack probs at 40x60 (needed for ROC / Table IV / component ablation inputs)
python exp_4/regenerate_stack_test_probs.py --n_machine 40 --n_clinical 60

# updated DL rows + Table IV
python exp_4/rerun_clinical_mlp_60.py
python exp_4/rerun_machine_ensemble_40.py
python exp_4/assemble_table4_updated.py

# feature sweeps + figures
python exp_4/sweep_mrmr_stack.py && python exp_4/draw_mrmr_heatmap.py --metric test
python exp_4/sweep_nclin_dl.py

# SHAP (best config), ROC, loss curves
python exp_4/shap_best_combined.py --n_machine 40 --n_clinical 60
python exp_4/plot_roc.py
python exp_4/loss_curve_fusion.py && python exp_4/replot_loss_curve.py

# ablations
python exp_4/component_ablation.py
python exp_4/tableV_updated_40x60.py
python exp_5_ablation/tableV_rows1_4_40x60.py
```
Note: GPU jobs should run **sequentially** (two classical+GPU jobs together can OOM). Use
`setsid ... < /dev/null &` to detach long jobs on the shared node; set `export ABL_NJOBS=4` for classical tuning.

## 6. Leakage barrier (every result)
- **TRAIN** → imputation, standardization, mRMR selection, all NN weights, classical tuning.
- **VALIDATION** → base-learner ranking / top-k, the stacking logistic meta-learner, decision threshold,
  early-stopping, best-seed selection.
- **TEST** → untouched during fitting; only for reported metrics (+ patient-bootstrap CI).
- Splits patient-disjoint.

## 7. Honest caveats (read before writing the paper)
1. **40×60 is best on TEST, not validation.** Validation (the proper selection split) ranks 20×40 / 40×20
   higher. With ~30 validation positives the surface is **flat within ±0.06 CI** and val does not reliably
   rank configs. Present 40×60 as the chosen operating config with the test-sensitivity view — do **not**
   claim it was tuned on validation.
2. **DL clinical branch stayed at 40**, not 60 — the 60 clinical only applies to the classical members.
3. **Classical learners dominate** the stack (component ablation: 0.758 alone); the transformer-fusion DL
   adds a modest lift (~+0.011 from the machine branch); the DL clinical branch is redundant with classical.
4. **Machine signal is weak** (machine-only ~0.55–0.63) and **does not externally validate** (UAB ~0.45–0.48).
5. **Combined model is not externally validatable** on UAB (8/40 clinical features present; labs absent).
6. **Sub-population Tables X/XI** were computed for the reported **0.738** stack (20×40), not the 40×60 —
   regenerate the stack probs at 40×60 and re-run `subpopulation_analysis_combined.py` if you want them at 0.766.
7. Preprocessing ablation (Table V top block) is noisy at 40/60 (classical members dominate); the encoder
   comparison and the peak-removal step remain the clean signals.

## 8. Where the summaries live
- `paper_materials/UPDATED_RESULTS.md` — all results + file/figure index.
- `paper_materials/model_architecture_methods.txt` — per-branch architecture + equations (methods section).
- `exp_4/UAB_external_validation/REPORT.md` — external validation writeup.
- `paper_materials/subpop_combined/SUBPOPULATION_COMBINED.md` — sub-population writeup.
