# abalation_study_v2 — enriched CRRT machine features

A self-contained re-run of the CRRT mortality study with a **richer machine feature
representation**, built to test whether better features let the pressure waveforms add
prognostic value beyond clinical data (the v1 finding was that combined ≈ clinical).

Everything — data processing, processed data (`./data`), models, results (`./results`) —
lives **inside this folder**. It reuses the same resampled-evaluation protocol, the same
models, and the same mRMR selection as `Abalation_study`.

**Splits (v2-specific):** v2 builds its OWN patient-level **75 : 5 : 20** train/val/test
split (`preprocessing/build_splits.py` → `data/splits/pid_splits_v2.json`), **stratified so
the positive rate matches across splits** (stratum = each patient's positive-circuit-day
count). A patient's circuit-days never straddle splits (no leakage). Result: 731/49/196
PIDs, positive rate ≈ 0.061/0.054/0.064 (1day) and 0.106/0.092/0.113 (2day); the 20% test
set (~1002 circuit-days, ~64 deaths) also makes the subpopulation analysis more evaluable.

---

## What changed vs v1 (the only difference)

v1 machine features = **13 marginal-distribution stats × 4 raw channels = 52**, each
channel summarized in isolation and order-invariant. That discards the CRRT-specific
physiology that signals circuit deterioration. v2 adds two families:

### 1. Cross-channel CRRT-physiology channels (per timestep, then the same 13 stats)
| Channel | Formula | Captures |
|---|---|---|
| **TMP** | (FILTER + RETURN)/2 − EFFLUENT | transmembrane pressure → membrane fouling |
| **PFD** | FILTER − RETURN | pressure drop across the filter → clotting |
| **ARD** | RETURN − ACCESS | circuit pressure span / access instability |

### 2. Within-day temporal dynamics (time-ordered series, per channel)
`slope`, `trend_r2` (trend strength), `mad` (mean |Δ|), `max_jump` (max |Δ|),
`ac1` (lag-1 autocorrelation), `exc_frac` (fraction of minutes with |z| > 2).
These encode *trajectories* (e.g. filter/TMP rising = clotting) that distribution stats erase.

**Result:** ~**133 machine features** per circuit-day =
`(4 raw + 3 derived) × 13 stats (91) + (4 raw + 3 derived) × 6 dynamics (42)`.
The DL sequence cache likewise grows from `[T,4]` to **`[T,7]`** (raw + derived channels).

Knobs that follow from this (`abl_config.py`): `DERIVED_CHANNELS`, `ALL_CHANNELS`,
`DYN_STATS`; `is_machine_col` now matches derived prefixes; mRMR machine budgets raised
20 → 30 (`N_MRMR`, `N_MRMR_MACHINE`) to sample the richer set. Clinical features, splits,
eval protocol, and models are **unchanged**.

---

## How to run

```bash
conda activate crrt
cd Baseline_ML_Models_v2/abalation_study_v2

# 0) BUILD SPLITS  (stratified 75:5:20 patient-level; writes data/splits/pid_splits_v2.json)
python preprocessing/build_splits.py

# 1) PROCESS DATA  (writes ./data/abalation/current_<peak>/...; ~5–15 min)
python refresh_data.py --variants current --peaks denoised noised --cases 1day 2day --force
#   per (peak, case) builds: machine_only, combined (machine+clinical), their *_agg
#   expanding-day versions, clinical_only_agg, and the DL seq_cache/*.npz ([T,7]).
#   add --skip-seq for tables only (no DL cache).

# 2) CLASSICAL benchmark (8 models × clinical/machine/combined × per_day/agg)
python run_benchmark.py --variant current --peak denoised --scopes per_day agg
python run_benchmark.py --variant current --peak noised   --scopes per_day agg

# 3) DEEP-LEARNING benchmark (dual-GPU; machine_only, combined, + clinical MLP)
python run_dl_benchmark.py --gpus 0 1 --variant current --peak denoised --seeds 5

# 4) SUBPOPULATION results (Ward clustering of test patients on clinical features ->
#    4 subgroups; per-subpop AUROC for every saved classical + DL model, on the 20% test set)
python subpopulation_eval.py --case 1day
python subpopulation_eval.py --case 2day
```

Outputs land in `./results/` (one row per model config with `full_test_*`,
`r1to1_*_mean/std`, `r1to2_*_mean/std`), saved models under `results/saved_models/`, and
subpopulation tables/figures under `results/subpopulation/<case>/`.

### Compare against v1
The point of v2 is the **delta**. After step 2–3, build the same best-per-pipeline
metric table as v1 (`summary_all_models_denoised.csv`) and compare machine_only and
combined AUROC (full / 1:1 / 1:2) old vs new; a paired DeLong test / bootstrap CI of
ΔAUROC tells you whether the enriched features produce a real gain or fall within the
~0.028 resampling noise (the cohort has only ~53 deaths, so small gains are hard to detect).

---

## The evaluation protocol (`eval/resampled_eval.py`)

Train once, score once on the full test set. Then: all test positives always included,
negatives subsampled to **1:1** and **1:2**, **100 seeded iterations** (`RandomState(SEED0+i)`,
shared across models → paired), reporting **mean ± std** of AUROC/AUPRC/F1/Precision/
Recall/Balanced-Accuracy. Threshold = validation F1-optimal, held fixed across resamples.

## DL model (`models/dl_aggregated.py`)
3-branch late fusion: current-day `[T,7]` stream encoder (cnn1d / rnn / transformer,
with `dl_reduce` window-aggregation or patching) ⊕ prior-day machine summary MLP ⊕
clinical summary MLP (combined only). `clinical_only` uses a plain tuned MLP
(`models/dl_clinical_mlp.py`). HP search keeps the best by validation AUROC; balanced
minibatches (`WeightedRandomSampler`), AdamW, `ReduceLROnPlateau`, early-stop on val AUROC,
reported mean ± std over 5 seeds.

## Layout
```
abl_config.py              paths (data inside ./data), channels, mRMR, eval protocol
preprocessing/
  build_splits.py          stratified 75:5:20 patient-level split -> data/splits/pid_splits_v2.json
  build_features.py        ENRICHED per-day machine table (derived channels + dynamics)
  build_combined.py        machine + stable clinical join
  build_agg.py             expanding-day {mean,min,max,median} aggregation
  build_sequences.py       DL [T,7] sequence cache (raw + derived channels)
  build_streams.py         priming/downtime/peak cleaning (for LOO ablation)
refresh_data.py            one-command data processing -> ./data
models/                    classical (8), feature_select (mRMR), dl_* (fusion, clinical MLP)
run_benchmark.py           classical benchmark
run_dl_benchmark.py        dual-GPU DL benchmark
run_ablation.py / run_loo_ablation.py   Th-sweep + leave-one-out (after picking winners)
subpopulation_eval.py      subpopulation distribution-shift evaluation
results/  data/            generated outputs (created on first run)
```
Clinical per-day lab/PTI features are denoising-independent and read from the stable
`Model_Ready/clinical_only` table (the only external read-only input besides the
pressure CSVs).
