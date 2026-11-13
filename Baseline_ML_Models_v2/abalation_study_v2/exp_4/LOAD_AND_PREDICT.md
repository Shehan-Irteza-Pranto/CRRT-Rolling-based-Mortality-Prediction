# exp_4/models — saved models for external validation

All 11 Table VI models × 2 horizons = **22 bundles** in `models/` (index: `MANIFEST.csv`,
`pathA_bundles_index.csv`). Each is the **best-of-5-seeds by validation AUROC** with its
**max-F1-on-validation threshold**. Load with `joblib.load(...)`.

## What's inside each bundle
| kind | models | contents | self-contained? |
|---|---|---|---|
| `dl_seq_attn` | Attention-CNN1D | AttnCNN1D `state_dict` + arch + channel z-score `norm_mean/std` + threshold | fully* |
| `ensemble_mean` | Ensemble (machine) | kept classical members (fitted sklearn estimator + imputer/scaler/mRMR idx each) + attn-CNN weights + norm + threshold | fully* |
| `dl_fusion` | Attn-CNN fusion | FusionModel `state_dict` + arch + clinical-branch imputer/scaler/mRMR-40 idx + threshold | fully* |
| `logistic_stack` | Ensemble/Stack | top-4 members (fitted estimators/prep + the fusion weights) + fitted `meta_learner` (LogisticRegression) + threshold | fully* |
| `pathA_benchmark` | MLP, CNN1D, RNN, Transformer | best-seed `.pt` (`state_dict`+arch) + threshold + tabular prep (clinical/combined branches embedded; machine-seq via repo loader) | weights+thr embedded |

\* "fully" = tabular branches are embedded; the **machine sequence branch** of every sequence model
(Path A *and* B) still requires the raw signal to be converted to the reduced `[N, 144, 30]` window
representation by the **repo's signal pipeline** — inherent to any waveform model, not a missing artifact.

## Round-trip proof
`python predict.py --verify` reconstructs each Path B model from its bundle and reproduces the reported
test AUROC to **±0.0005** (all 8 = OK). This confirms the bundles are functional.

## Applying a model to an EXTERNAL cohort
1. **Feature-extract the external cohort with the same pipeline** into the project's table/sequence
   formats (see `paper_materials/DATA_PROCESSING.md`): the per-day clinical table, the base/rich machine
   feature tables, and the window-reduced pressure sequence `[N, 6, 1440] → [N, 144, 30]`.
2. **Build a context** and call `predict.score(bundle_path, ctx)` (see `predict.py::_test_ctx` for the
   exact contract). It returns `(probabilities, threshold)`; predict death = `prob >= threshold`.
   - `dl_seq_attn` / `dl_fusion`: need the reduced sequence `Xte,[mask]` (z-scored with the bundle's
     `norm_mean/std`) and, for fusion, the combined clinical table.
   - `ensemble_mean`: needs the base+rich machine tables + reduced sequence.
   - `logistic_stack`: needs the combined + clinical tables + reduced sequence.
   - `pathA_benchmark`: rebuild the reduced sequence via `models/AggDataset` (machine/combined) or the
     tabular features via `models/TabularClinicalDataset` (clinical), load `checkpoint['state_dict']`.
3. **Preprocessing is fit on the ORIGINAL training data** — every imputer/scaler/mRMR selection and the
   sequence z-score stats are the *training* ones (embedded or cached), never re-fit on external data.

## Notes
- Thresholds are model-specific (in each bundle + `MANIFEST.csv`); AUROC/AUPRC are threshold-free.
- Best-seed selection is on **validation** AUROC (leak-safe), so a model's test AUROC is honest, not
  cherry-picked. See `../README.md` for the best-seed vs seed-mean caveats.
- Bundles total ~65 MB (`models/`). Parent checkpoints (`../results/saved_models/dl/*.pt`) are untouched.
