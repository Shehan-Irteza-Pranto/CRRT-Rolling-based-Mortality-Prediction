# exp_4 — Table VI rebuilt with best-seed (by validation AUROC) + best-threshold P/R

Addresses the request to re-report the DL benchmark (Table VI) so precision/recall are meaningful:
instead of **averaging** the 5 seeds' probabilities (which compresses scores and deflates P/R), select
**the single seed with the highest validation AUROC** per cell and report that seed's held-out test
metrics at the **max-F1-on-validation** threshold. Clinical-only and combined pipelines use the
**mRMR top-40** clinical features (already the case in the DL code — verified, unchanged).

## What each script does
| script | role | compute |
|---|---|---|
| `build_pathA_bestseed.py` | CNN1D / RNN / Transformer (machine+combined) + clinical MLP: pick the max-`best_val_auroc` seed **row** from `../results/dl_current_denoised.csv` (per_day scope). | none (row selection) |
| `rerun_machine_bestseed.py` | machine **Attention-CNN1D** + **Ensemble**: re-run per seed (`exp_3/dl/train_eval.train_one` + per-seed classical via `classical_rich._multiseed_probs`), pick best-val seed. | GPU |
| `rerun_combined_bestseed.py` | combined **Attn-CNN fusion** + **Ensemble/Stack**: re-run the CNN-fusion + logistic top-4 stack per seed (`combined_fusion_ensemble` with the added `per_seed=True` hook), pick best-val seed. | GPU |
| `assemble_table.py` | merge the three into `results/table_vi_bestseed.csv` + `TABLE_VI_bestseed.md` (attached layout). | none |

Run order: `build_pathA_bestseed.py`; then the two `rerun_*` (one per GPU, background); then
`assemble_table.py`.

## Method notes (for the paper)
- **Seed selection is on VALIDATION AUROC**, evaluation on TEST — leak-safe model selection, not
  cherry-picking on test.
- **Threshold** = argmax-F1 on the chosen seed's validation probabilities, held fixed across full /
  1:1 / 1:2 (`RESAMPLE_RATIOS=(1,2)`, `N_ITER=100`, negatives subsampled, positives kept).
- **Best-seed vs seed-mean is not always higher on test.** With ~30–60 validation positives, the
  best-validation seed can land slightly below the seed-mean on test AUROC (e.g. machine
  Attention-CNN1D 1-day 0.579 vs mean 0.605); this is the honest cost of leak-safe selection. The
  intended payoff is **sharper precision/recall** at the operating point (single-seed scores are not
  averaged/compressed): e.g. clinical MLP recall 0.452 (was 0.300), combined CNN1D 2-day recall 0.651
  (was 0.398), combined Ensemble/Stack 1-day AUROC **0.740** (seed-mean 0.711).
- **The machine Ensemble row stays low-recall** (0.11–0.15) even at best seed, because it is itself a
  *mean* of members (top-3 classical + attn-CNN) — averaging compresses its scores regardless of seed.

## Parent code
Reused read-only except two **additive, backward-compatible** hooks in
`exp_3/results/combined_fusion_ensemble.py`: `_fusion_probs(..., per_seed=False)` and
`_classical_probs(..., per_seed=False)` return per-seed arrays when `per_seed=True` (default False
preserves the original seed-averaged behavior). No existing outputs were changed or deleted.
