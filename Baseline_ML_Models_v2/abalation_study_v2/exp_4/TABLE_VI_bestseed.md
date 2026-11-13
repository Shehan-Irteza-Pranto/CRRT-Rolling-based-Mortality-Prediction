# Table VI (best-seed) — DL benchmark, best-of-5-seeds by validation AUROC

Values = the seed with the highest **validation** AUROC per cell (leak-safe model selection); reported on the held-out test set. Threshold for F1/Precision/Recall = **max-F1 on validation**. Clinical-only & combined use **mRMR top-40** clinical features. Balanced 1:1 / 1:2 = mean over 100 resamples (positives kept, negatives subsampled). Compare to the seed-averaged Table VI in the paper.


### One-day horizon

| Model | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| *Clinical-only* | | | | | | | | | |
| MLP | 0.665 | 0.177 | 0.237 | 0.161 | 0.452 | 0.668 | 0.686 | 0.665 | 0.534 |
| *Machine-only* | | | | | | | | | |
| CNN1D | 0.579 | 0.095 | 0.026 | 0.071 | 0.016 | 0.575 | 0.585 | 0.579 | 0.415 |
| RNN (BiLSTM/GRU) | 0.584 | 0.094 | 0.159 | 0.098 | 0.419 | 0.583 | 0.583 | 0.585 | 0.412 |
| Transformer | 0.581 | 0.086 | 0.123 | 0.099 | 0.161 | 0.578 | 0.559 | 0.582 | 0.389 |
| Attention-CNN1D | 0.579 | 0.102 | 0.165 | 0.155 | 0.177 | 0.578 | 0.601 | 0.579 | 0.430 |
| Ensemble | 0.622 | 0.109 | 0.131 | 0.156 | 0.113 | 0.621 | 0.620 | 0.621 | 0.449 |
| *Combined* | | | | | | | | | |
| CNN1D | 0.646 | 0.182 | 0.188 | 0.121 | 0.419 | 0.644 | 0.674 | 0.646 | 0.526 |
| RNN (BiLSTM/GRU) | 0.669 | 0.222 | 0.278 | 0.302 | 0.258 | 0.667 | 0.698 | 0.669 | 0.559 |
| Transformer | 0.680 | 0.182 | 0.225 | 0.156 | 0.403 | 0.680 | 0.696 | 0.682 | 0.549 |
| Attn-CNN fusion | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.692 | 0.681 | 0.536 |
| Ensemble/Stack | **0.740** | 0.170 | 0.244 | 0.178 | 0.387 | 0.739 | 0.718 | 0.739 | 0.565 |

### Two-day horizon

| Model | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| *Clinical-only* | | | | | | | | | |
| MLP | 0.715 | 0.333 | 0.384 | 0.321 | 0.477 | 0.711 | 0.727 | 0.713 | 0.592 |
| *Machine-only* | | | | | | | | | |
| CNN1D | 0.610 | 0.219 | 0.252 | 0.181 | 0.413 | 0.610 | 0.638 | 0.610 | 0.480 |
| RNN (BiLSTM/GRU) | 0.590 | 0.179 | 0.190 | 0.180 | 0.202 | 0.591 | 0.594 | 0.590 | 0.425 |
| Transformer | 0.557 | 0.145 | 0.207 | 0.160 | 0.294 | 0.558 | 0.554 | 0.558 | 0.380 |
| Attention-CNN1D | 0.589 | 0.182 | 0.265 | 0.180 | 0.505 | 0.589 | 0.609 | 0.589 | 0.442 |
| Ensemble | 0.639 | 0.211 | 0.176 | 0.219 | 0.147 | 0.640 | 0.641 | 0.640 | 0.478 |
| *Combined* | | | | | | | | | |
| CNN1D | 0.677 | 0.298 | 0.278 | 0.177 | 0.651 | 0.681 | 0.699 | 0.678 | 0.553 |
| RNN (BiLSTM/GRU) | 0.661 | 0.303 | 0.279 | 0.219 | 0.385 | 0.664 | 0.691 | 0.662 | 0.546 |
| Transformer | 0.668 | 0.286 | 0.275 | 0.295 | 0.257 | 0.668 | 0.693 | 0.667 | 0.545 |
| Attn-CNN fusion | 0.664 | 0.235 | 0.279 | 0.214 | 0.404 | 0.666 | 0.660 | 0.665 | 0.499 |
| Ensemble/Stack | **0.718** | 0.279 | 0.331 | 0.256 | 0.468 | 0.720 | 0.715 | 0.717 | 0.564 |

*Source: `exp_4/results/table_vi_bestseed.csv`. Path A (MLP/CNN1D/RNN/Transformer) = best-val-seed rows from `results/dl_current_denoised.csv`; Path B (Attention-CNN1D, Ensemble, Attn-CNN fusion, Ensemble/Stack) = per-seed re-runs (`rerun_machine_bestseed.py`, `rerun_combined_bestseed.py`).*
