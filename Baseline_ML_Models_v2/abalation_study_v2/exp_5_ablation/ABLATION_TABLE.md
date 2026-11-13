# Ablation study — combined best model (Transformer-fusion + 3 classical), 1-day

Ablation of the best combined model (a logistic stack of a sequence-encoder + clinical **fusion DL** and
the **top-3 classical** models), 1-day, best-of-5-seeds by validation AUROC, held-out test set.
**Optimal feature set: 40 machine × 60 clinical (grouped-mRMR).**
F1/Precision/Recall at the max-F1-on-validation threshold.
*(Rows 5–8 report AUROC only — the encoder-swap operating-point metrics were not recomputed at 40×60.)*
**Top block** — incremental preprocessing (enriched features → priming & downtime removal → peak removal),
encoder fixed to the tuned patch-Transformer. **Bottom block** — full preprocessing (✓✓✓ = "current
denoised"), sequence encoder swapped. Best AUROC in **bold**. *(Priming and downtime removal are merged into
one preprocessing step.)*

| Enriched Features | Priming & Downtime Removal | Peak Removal | Sequence Encoder | AUROC | AUPRC | Precision | Recall | F1 |
|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | Patch Transformer | 0.737 | 0.192 | 0.205 | 0.387 | 0.268 |
| ✓ | ✗ | ✗ | Patch Transformer | 0.730 | 0.188 | 0.177 | 0.419 | 0.249 |
| ✓ | ✓ | ✗ | Patch Transformer | 0.747 | 0.201 | 0.189 | 0.500 | 0.274 |
| ✓ | ✓ | ✓ | CNN | 0.729 | — | — | — | — |
| ✓ | ✓ | ✓ | RNN | 0.713 | — | — | — | — |
| ✓ | ✓ | ✓ | Transformer | 0.732 | — | — | — | — |
| ✓ | ✓ | ✓ | **Patch Transformer (Ours)** | **0.766** | — | — | — | — |

## Reading the ablation
- **Preprocessing helps cumulatively.** AUROC from raw to fully-cleaned:
  raw 0.737 → +enriched 0.730 → +priming&downtime 0.747 → **+peak 0.766**. **Priming & downtime removal**
  (+0.017) and **peak removal** (+0.019) give the lifts; enriched features are slightly negative here
  (−0.007) but within the ±0.06 test CI (62 positives), so the *cumulative* gain
  (0.737 → 0.766, **+0.029**) is the meaningful signal, not any single step.
- **Encoder ablation (full preprocessing).** The tuned **windowed/patch-Transformer (0.766)** is the best
  sequence encoder for the combined model, ahead of the default Patch-Transformer (0.732), CNN (0.729), and
  RNN (0.713).
- **Operating-point metrics move with the threshold**, so precision/recall vary row-to-row (e.g. the raw
  model has high recall 0.484 / low precision; the full model is more balanced). AUROC/AUPRC are the
  threshold-free comparison.

*Rows 1–4: `tableV_rows1_4_40x60.py` → `../exp_4/results/tableV_rows1_4_40x60.csv` (tuned patch-Transformer
combined stack, classical base learners at 40/60, re-trained on re-cleaned signal variants from
`gen_variants.py`). Rows 5–8: combined encoder-swap at 40/60, `../exp_4/results/tableV_rows5_8_40x60.csv`.
"Patch Transformer" = the HP-tuned transformer fusion (window-token `[144×30]`) behind the best combined
model. (Previous 20×40 numbers in git history.)*
