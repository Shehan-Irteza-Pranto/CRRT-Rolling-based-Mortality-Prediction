# Pipeline × encoder matrix (1-day, best-seed)

For **both pipelines** (machine-only, combined) and **four sequence encoders** — **CNN1D (mean pooling)**,
**CNN-attention**, **RNN (BiGRU, mean pooling)**, **patch-Transformer** — each is reported as the
**standalone DL** and as its **ensemble** = a logistic stack of the DL **+ top-3 classical** models
(machine: base/rich machine models; combined: combined + clinical models, by validation AUROC). All
models are built under the **same recipe** (window-reduced `[144×30]` sequence, 5 seeds, best-of-5 by
validation AUROC; threshold = max-F1 on validation), so rows are internally comparable. Balanced 1:1 /
1:2 = mean over 100 resamples. Best AUROC per section in **bold**.

## Machine-only pipeline
| Encoder | Variant | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **CNN1D (mean pool)** | DL | 0.590 | 0.111 | 0.206 | 0.172 | 0.258 | 0.590 | 0.615 | 0.591 | 0.449 |
| | Ensemble (DL + 3 classical) | 0.636 | 0.113 | 0.121 | 0.162 | 0.097 | 0.635 | 0.627 | 0.636 | 0.458 |
| **CNN-attention** | DL | 0.579 | 0.102 | 0.165 | 0.155 | 0.177 | 0.578 | 0.601 | 0.579 | 0.430 |
| | Ensemble (DL + 3 classical) | 0.621 | 0.109 | 0.104 | 0.147 | 0.081 | 0.619 | 0.619 | 0.620 | 0.449 |
| **RNN (BiGRU)** | DL | 0.566 | 0.114 | 0.161 | 0.101 | 0.387 | 0.564 | 0.602 | 0.566 | 0.440 |
| | **Ensemble (DL + 3 classical)** | **0.641** | 0.109 | 0.089 | 0.100 | 0.081 | 0.641 | 0.619 | 0.641 | 0.448 |
| **patch-Transformer** | DL | 0.618 | 0.098 | 0.168 | 0.106 | 0.403 | 0.614 | 0.591 | 0.618 | 0.421 |
| | Ensemble (DL + 3 classical) | 0.636 | 0.105 | 0.083 | 0.114 | 0.065 | 0.632 | 0.610 | 0.636 | 0.440 |

## Combined pipeline
| Encoder | Variant | AUROC | AUPRC | F1 | Prec. | Rec. | 1:1 AUROC | 1:1 AUPRC | 1:2 AUROC | 1:2 AUPRC |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **CNN1D (mean pool)** | Fusion-DL | 0.711 | 0.218 | 0.253 | 0.202 | 0.339 | 0.711 | 0.728 | 0.712 | 0.587 |
| | Ensemble (DL + 3 classical) | 0.719 | 0.184 | 0.261 | 0.197 | 0.387 | 0.720 | 0.723 | 0.720 | 0.575 |
| **CNN-attention** | Fusion-DL | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.692 | 0.681 | 0.536 |
| | Ensemble (DL + 3 classical) | 0.694 | 0.174 | 0.258 | 0.194 | 0.387 | 0.694 | 0.708 | 0.694 | 0.558 |
| **RNN (BiGRU)** | Fusion-DL | 0.664 | 0.185 | 0.204 | 0.132 | 0.452 | 0.663 | 0.690 | 0.664 | 0.541 |
| | Ensemble (DL + 3 classical) | 0.709 | 0.174 | 0.219 | 0.151 | 0.403 | 0.708 | 0.710 | 0.709 | 0.560 |
| **patch-Transformer** | **Fusion-DL** | **0.724** | 0.188 | 0.250 | 0.198 | 0.339 | 0.723 | 0.725 | 0.725 | 0.581 |
| | Ensemble (DL + 3 classical) | 0.708 | 0.188 | 0.249 | 0.180 | 0.403 | 0.704 | 0.716 | 0.707 | 0.570 |

## Reading the matrix
- **Combined ≫ machine-only** everywhere (adding the 40 clinical features lifts AUROC by ~0.10) — the
  clinical signal dominates, as throughout the study.
- **Encoder ranking (standalone):** patch-Transformer is the strongest sequence encoder in **both**
  pipelines (machine 0.618, combined 0.724); CNN1D mean-pool is a solid second in combined (0.711).
  CNN-attention and RNN trail as standalone models.
- **Ensembling raises AUROC but not the ceiling.** Machine-only ensembles (0.62–0.64) beat their DL
  (0.57–0.62); combined ensembles land ~0.69–0.72 — all pulled toward the classical members. Note the
  ensembles' **recall collapses** (machine 0.07–0.10) because the logistic stack of mostly-classical
  members produces conservative, compressed scores.
- **The best numbers are within noise of each other** (±0.06 CI, 62 positives): combined patch-Transformer
  standalone **0.724** and CNN1D-mean ensemble 0.719 are statistically tied.

## References (not rebuilt here, same protocol)
- **Tuned patch-Transformer** (HP-searched) combined: Fusion-DL 0.735 seed-avg (0.686 best-seed);
  ensemble (tf + 3 classical) **0.738** — see `BRANCH_REPORT_TABLE.md` / `hp_search_transformer.py`.
- **Classical-only Stack top-4** (no DL) combined = **0.740** — the overall best, statistically tied
  with the above.

*All numbers under the exp_3 recipe (window_agg 30-ch, 5-seed best-val); these differ from the Table VI
Path A numbers (patch-reduction, HP-searched) and are not directly comparable across pipelines. Sources:
`results/matrix_machine_1day.csv`, `results/matrix_combined_1day.csv`. Scripts: `matrix_machine_1day.py`,
`matrix_combined_1day.py`.*
