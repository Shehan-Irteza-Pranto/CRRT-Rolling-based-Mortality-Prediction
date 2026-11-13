# Ablation — swapping the machine branch of the best ensemble (1-day)

**Question.** The best model is the combined **Ensemble/Stack** (0.740), whose DL component is a
CNN-fusion: an **attention-CNN1D** machine branch + a clinical-MLP branch. What happens if that machine
branch is a different sequence encoder?

**Design.** Rebuild the fusion with three machine encoders — **CNN-attention** (current, baseline),
**RNN (BiGRU)**, and a **patch-Transformer** — over the *same* window-reduced pressure sequence
`[144×30]`, keeping the **same** clinical branch (mRMR-40), training recipe (AdamW, class-balanced
sampler, early-stop on val AUROC, 5 seeds), and stacking. Best-of-5-seeds by **validation** AUROC;
threshold = max-F1 on validation. The combined-classical + clinical-classical stack members and the
logistic meta-learner are identical across branches, so any delta isolates the encoder. Everything is
1-day, held-out test. Source: `ablation_ensemble_branch.py` → `results/ablation_branch_1day.csv`.

Three views are reported per branch:
- **Fusion-DL** — the DL model alone (encoder + clinical branch).
- **Stack top-4** — the actual best method (top-4 components by val AUROC → logistic meta-learner).
- **Stack fusion+top-3** — force the DL fusion *into* the ensemble (fusion + top-3 classical), to expose
  the branch effect when the DL model is actually used.

## Results (held-out test, best-val seed)
| branch | component | AUROC | AUPRC | F1 | Prec | Rec | 1:1 AUROC | 1:2 AUROC |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **CNN-attention** (current) | Fusion-DL | 0.682 | 0.156 | 0.239 | 0.163 | 0.452 | 0.680 | 0.681 |
| | Stack top-4 *(best)* | **0.740** | 0.170 | 0.244 | 0.178 | 0.387 | 0.739 | 0.739 |
| | Stack fusion+top-3 | 0.694 | 0.174 | 0.258 | 0.194 | 0.387 | 0.694 | 0.694 |
| **RNN (BiGRU)** | Fusion-DL | 0.664 | 0.185 | 0.204 | 0.132 | 0.452 | 0.663 | 0.664 |
| | Stack top-4 *(best)* | **0.740** | 0.170 | 0.244 | 0.178 | 0.387 | 0.739 | 0.739 |
| | Stack fusion+top-3 | 0.709 | 0.174 | 0.219 | 0.151 | 0.403 | 0.708 | 0.709 |
| **patch-Transformer** | Fusion-DL | **0.724** | 0.188 | 0.250 | 0.198 | 0.339 | 0.723 | 0.725 |
| | Stack top-4 *(best)* | **0.740** | 0.170 | 0.244 | 0.178 | 0.387 | 0.739 | 0.739 |
| | Stack fusion+top-3 | 0.708 | 0.188 | 0.249 | 0.180 | 0.403 | 0.704 | 0.707 |

## What happened
1. **As a standalone DL model, the branch matters — and the patch-Transformer wins.** Fusion-DL AUROC:
   **patch-Transformer 0.724 > CNN-attention 0.682 > RNN 0.664** (Transformer also has the best AUPRC,
   0.188). Feeding the 144 window-tokens to a small self-attention encoder captures the day-level
   structure better than local convolutions or a recurrent pass.
2. **But the best Ensemble/Stack is unchanged (0.740) — because it never uses the DL branch.** The
   top-4-by-validation stack selects the strongest classical members (combined GradientBoosting/RF +
   clinical labs); the DL fusion's validation AUROC is below the 4th classical member, so
   `fusion_in_top4 = False` for **every** branch. The winning stack is therefore identical regardless of
   the machine encoder — the ensemble's strength comes from the classical models, not the sequence net.
3. **Forcing the DL branch into the ensemble *lowers* it** (0.69–0.71 < 0.740): the DL fusion is the
   weakest member, so including it displaces a stronger classical model. Among the forced variants the
   encoder barely matters (RNN 0.709 ≈ Transformer 0.708 > CNN 0.694).

## Takeaway
Swapping the CNN branch for an RNN or a patch-Transformer **improves the standalone DL fusion**
(Transformer best, +0.04 AUROC) but **does not change the best ensemble** — the Ensemble/Stack already
discards the DL fusion in favor of higher-validation classical models, so the machine-sequence encoder is
not the bottleneck for the headline 0.740. If the DL fusion is the model of interest, use the
patch-Transformer branch; if the Stack is the deliverable, the branch choice is immaterial.

*Reproduce: `python ablation_ensemble_branch.py --case 1day` (GPU). Numbers trace to
`results/ablation_branch_1day.csv`. Same best-seed / max-F1-val protocol as the exp_4 Table VI.*
