# EHR-only Stack (1-day) — the deployed Fusion Stack with no machine data

**Question.** Table III / the component ablation report a "machine branch zeroed" row (0.755), but that row
still (a) trains the clinical branch jointly with the machine encoder and (b) stacks classical learners fit
on the combined 40-machine + 60-clinical table. What does the Stack reach when it is built from EHR only?

**Design** (`ehr_only_stack.py` → `results/ehr_only_stack_1day.csv`, run 2026-11-05). Same protocol as the
deployed model: 5 seeds, best-of-5 by validation AUROC, logistic meta-learner fit on validation, threshold =
max-F1 on validation, test metrics full / 1:1 / 1:2 (100 resamples).
- DL candidates: *clin-branch* = the fusion model's clinical-MLP branch trained alone on the same mRMR-40
  clinical tensors (CH 64, dropout 0.4, AdamW 3e-4 / 5e-3, balanced sampler, 120 ep / patience 15);
  *mlp60* = Table III "Clinical MLP" (mRMR-60, HP-searched), re-loaded from its 5 saved seeds (best-val seed
  reproduces 0.710).
- Classical candidates: GradBoost / RF / LightGBM / XGBoost / LogReg on the clinical-only table at mRMR-60
  (Table IV setting) and, as a sensitivity, on all 158 clinical features.

## Results (held-out test, n = 887 patient-days, 62 positives)
| variant | members (best-val seed) | AUROC | AUPRC | F1 | Prec | Rec | 1:1 AUROC | 1:2 AUROC |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| DL: clinical branch alone (mRMR-40) | clin_branch | 0.641 | 0.199 | 0.265 | 0.197 | 0.403 | 0.639 | 0.640 |
| DL: Table III clinical MLP (mRMR-60) | mlp60 | 0.710 | 0.177 | 0.242 | 0.229 | 0.258 | 0.709 | 0.712 |
| EHR-only Stack: clin-branch + top-3 classical (mRMR-60) | clin_branch+RF+LightGBM+GradBoost | 0.717 | 0.223 | 0.308 | 0.255 | 0.387 | 0.714 | 0.717 |
| **EHR-only Stack: MLP-60 + top-3 classical (mRMR-60)** | mlp60+RF+GradBoost+LightGBM | **0.738** | 0.225 | 0.244 | 0.165 | 0.468 | 0.737 | 0.739 |
| EHR-only Stack: top-3 classical only (mRMR-60) | RF+GradBoost+LightGBM | 0.716 | 0.217 | 0.252 | 0.175 | 0.452 | 0.714 | 0.716 |
| EHR-only Stack: top-4 by val, all clinical candidates | mlp60+RF+GradBoost+LightGBM | 0.738 | 0.225 | 0.244 | 0.165 | 0.468 | 0.737 | 0.739 |
| EHR-only Stack: clin-branch + top-3 classical (ALL feats) | clin_branch+GradBoost+RF+LightGBM | 0.708 | 0.206 | 0.288 | 0.242 | 0.355 | 0.705 | 0.708 |
| EHR-only Stack: top-3 classical only (ALL feats) | GradBoost+RF+LightGBM | 0.707 | 0.196 | 0.259 | 0.204 | 0.355 | 0.705 | 0.707 |

Reference: deployed combined Stack (40 machine + 60 clinical) **0.766**; best single EHR-only model
(Table IV GradBoost, 60 feats) 0.717; component-ablation "machine zeroed, classical still combined" 0.755.

## Reading
1. The best EHR-only stack is **0.738** (Table III MLP-60 + RF + GradBoost + LightGBM, all clinical mRMR-60);
   the top-4-by-validation selection picks exactly this set. Adding the machine pressure stream lifts the
   Stack to 0.766 (**+0.028 AUROC**, +0.028 at 1:1, +0.027 at 1:2).
2. Stacking EHR models alone adds ~+0.02 over the best single EHR model (0.717 → 0.738), so roughly half of
   the combined Stack's margin over single clinical models comes from stacking, half from machine data.
3. The fusion model's clinical branch trained alone is weak (0.641): its 64-unit, dropout-0.4 design was
   tuned as a branch next to the sequence encoder, not as a standalone tabular model. Stacks built on it
   (0.717) do not beat the classical-only stack (0.716).
4. The 0.755 component-ablation row overstates "no machine": its classical members still see 40 machine
   features. The clean EHR-only number for the paper's ablation is 0.738.

Probabilities of the primary (clin-branch) stack: `results/ehr_only_stack_{val,test}_probs_1day.csv`.
