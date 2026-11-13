# Ensemble-branch report table (1-day, best-seed)

Each machine-branch model is shown **as a standalone DL fusion** (machine encoder + mRMR-40 clinical
branch) and as an **ensemble** = a logistic stack of that fusion **+ the top-3 classical models**
(combined GradientBoosting / RF / LightGBM / XGBoost, by validation AUROC). All numbers are the
best-of-5-seeds by **validation** AUROC on the held-out test set; threshold = max-F1 on validation;
balanced 1:1 / 1:2 = mean over 100 resamples. Best AUROC in **bold**.

| Model | Variant | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **CNN-attention** | Fusion-DL | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.692 | 0.681 | 0.536 |
| | + 3 classical (ensemble) | 0.694 | 0.174 | 0.258 | 0.194 | 0.387 | 0.694 | 0.708 | 0.694 | 0.558 |
| **RNN (BiGRU)** | Fusion-DL | 0.664 | 0.185 | 0.204 | 0.132 | 0.452 | 0.663 | 0.690 | 0.664 | 0.541 |
| | + 3 classical (ensemble) | 0.709 | 0.174 | 0.219 | 0.151 | 0.403 | 0.708 | 0.710 | 0.709 | 0.560 |
| **Transformer (tuned)** | Fusion-DL | 0.686 | 0.190 | 0.244 | 0.178 | 0.387 | 0.683 | 0.705 | 0.686 | 0.560 |
| | **+ 3 classical (ensemble)** | **0.738** | 0.185 | 0.217 | 0.156 | 0.355 | 0.737 | 0.728 | 0.738 | 0.580 |

**Best model: Transformer (tuned) + 3 classical = AUROC 0.738.**

## Notes
- **Ensembling helps every branch** (Fusion-DL → +3-classical): CNN 0.682→0.694, RNN 0.664→0.709,
  Transformer 0.686→**0.738**. The tuned transformer benefits most and is the strongest ensemble.
- The Transformer branch is the **HP-tuned** configuration (d_model 64, 1 layer, 4 heads, ff 128,
  dropout 0.4, weight-decay 5e-3, lr 3e-4); CNN-attention and RNN (BiGRU) are the ablation defaults.
- Reference (not shown per request): the classical-only Stack top-4 = **0.740** — statistically tied
  with the tuned-transformer ensemble (0.738); the ±0.06 test CI (62 positives) makes them
  indistinguishable.

*Sources: `results/ablation_branch_1day.csv` (CNN / RNN), `results/stack_tuned_transformer_1day.csv`
(tuned Transformer). Scripts: `ablation_ensemble_branch.py`, `hp_search_transformer.py`,
`stack_with_tuned_transformer.py`.*
