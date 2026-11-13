# exp_3 — pushing the machine-only 1-day AUROC toward 0.65–0.70

Goal: lift the **machine-only (pressure) AUROC** for the 1-day pipeline to 0.65–0.70, using
**information/representation** rather than sample-count tricks (MIL / fragment-augmentation were
already shown not to help — recombination can't raise effective N). All work is self-contained in
`exp_3/`; parent `abalation_study_v2/` and `exp_2/` are untouched. Pressures only; v2 patient splits;
leak-safe (impute/scale/mRMR/tune on TRAIN, threshold on VAL, AUROC on held-out TEST; bootstrap over
PATIENTS).

## What was built
- **Phase 1 — richer physiology features** (`features/`): added the circuit-instability signatures the
  parent 133-feature set lacked — spectral band-power + **dominant period** + spectral/permutation
  **entropy**, **end-of-day deterioration** (last-2h vs first-2h level & slope), sustained-excursion
  runs, cross-channel coupling (TMP–PFD) — 67 new features per circuit-day. Re-ran the parent mRMR +
  classical grid (5-seed).
- **Phase 2 — attention CNN1D** (`dl/`): plain per-day CNN1D on the window-reduced signal with
  **attention pooling** (weights the end-of-day windows) + 5-seed ensemble.
- **Phase 3/4 — ensemble** (`ensemble/`): logistic stack / mean of the rich-classical + CNN1D
  probabilities, evaluated on both the **1-day** and **first-2-day** outcome horizons.

## Headline numbers (held-out test, patient-bootstrap 95% CI)
Most robust estimate = `ensemble:mean` (no noisy val-based model selection):

| cohort | model | test AUROC | 95% CI |
|---|---|---|---|
| baseline (parent) 1day classical | best | 0.594 | — |
| baseline (parent) 1day CNN1D | — | 0.606 | — |
| **exp_3** 1day rich-classical (GB/RF) | single | 0.636 | [0.573, 0.706] |
| **exp_3** 1day attn-CNN1D (5-seed) | ensemble | 0.617 | [0.542, 0.692] |
| **exp_3** 1day **ensemble mean** | rich-cls + CNN | **0.649** | [0.576, 0.723] |
| baseline (parent) 2day classical RF / CNN1D | — | 0.662 / 0.666 | — |
| **exp_3** 2day rich-RF | single | 0.651 | [0.592, 0.714] |
| **exp_3** 2day **ensemble mean** | rich-cls + CNN | 0.643 | [0.576, 0.718] |

## What worked, what didn't
- **The richer features are real signal on 1-day**: +0.03–0.05 over the parent baseline
  (0.594 → 0.636 single / 0.649 ensembled). mRMR kept **12 of 30** machine features from the new set —
  dominant spectral period, end-of-day slope/delta, permutation entropy, sustained-excursion runs.
  Circuit-deterioration physiology genuinely carries prognostic signal.
- **They do NOT help the 2-day label**: the higher-prevalence 2-day outcome is already well-captured by
  marginal stats; rich features add nothing there (parent RF 0.662 stays the best 2-day model).
- **Attention + multi-seed**: confirmed the parent's single-seed 0.606 was typical (per-seed
  0.605 ± 0.020), and ensembling nudged it to 0.617 — modest.

## Honest conclusion vs the 0.65–0.70 target
- **1-day machine-only tops out at ~0.64–0.65** (point 0.649, CI [0.576, 0.723]). The target's lower
  bound is **grazed, not comfortably cleared** — the 95% CI includes both 0.65 and 0.58, so 0.65 is not
  a reliable floor for machine-only day-1 data.
- **2-day machine-only reaches ~0.66** (parent RF 0.662 / CNN 0.666), which **does** meet the lower end
  of 0.65–0.70 — i.e. predicting **2-day** mortality from day-1 circuit data clears 0.65.
- A stable **0.65–0.70 floor for the strict 1-day machine-only** model is **at/just beyond the data's
  signal ceiling**. The reliable routes above it are: (a) the **2-day horizon**, (b) **fusing clinical
  features** (clinical alone 0.71–0.73; combined 0.73–0.75), or (c) **more patients** (effective N,
  not method, is the binding constraint). Val-based selection can occasionally print a 0.65+ single
  number, but with ~30–50 val positives it does not replicate (e.g. 2-day rich-LogReg: val 0.633 →
  test 0.573) and is not reported as a result.

## Reproduce
```bash
conda activate crrt
cd Baseline_ML_Models_v2/abalation_study_v2/exp_3
python features/build_rich_features.py --case 1day      # 67 new features -> data/rich/
python features/classical_rich.py      --case 1day      # base vs rich classical, saves probs
python dl/train_eval.py                --case 1day      # attention CNN1D, 5 seeds, saves probs
python ensemble/ensemble.py            --case 1day      # stack -> final gate
# repeat with --case 2day for the 2-day horizon
```
