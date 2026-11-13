# Signal-Processing Pipeline — data cleaning & ablation

How the raw CRRT machine export is turned into the clean per-minute pressure stream the models
consume, and exactly what was varied in the preprocessing **ablation**. Parameters are the real
values from the code (`preprocessing/build_streams.py`, `preprocessing/smoothing.py`,
`abl_config.py`, `exp_3/features/day_signal.py`). For the downstream **feature equations** see
`DATA_PROCESSING.md`; this document is the cleaning-pipeline + ablation reference.

The raw input is `machine_filtered_v1` — the per-minute device export after the clinical-time
anomaly filter and visit numbering — at **1-minute native resolution** (no resampling), with 4 raw
channels: `ACCESS_PRESSURE`, `FILTER_PRESSURE`, `EFFLUENT_PRESSURE`, `RETURN_PRESSURE` (+ `RUN_TIME`,
`EFFLUENT` counters used only for cleaning).

---

## 0. Pipeline overview

```
 raw per-minute export (machine_filtered_v1)
            │
   [1] priming removal            (RUN_TIME == 0 on a 1-min edge)
            │
   [2] downtime removal           (flat: effluent unchanged  |  dose: ΔEFFLUENT < Th·dose)
            │
   [3] day numbering + cap        (Day = dense-rank(date); keep Day ≤ 14)
            │
   [4] peak removal / smoothing   (40-min window z-score outliers → neighbour-mean)   → "denoised"
            │                                                         (skip [4] → "noised")
   per-minute cleaned pressure CSV  (PID, Visit_no, Day, Time, 4 channels)
            │
   [5] gap-aware day signal        (0–1439 grid; ≤10-min gaps interpolated, larger masked)
            │                       + derived channels TMP/PFD/ARD  →  X[6,1440] + mask
            ▼
        features / models  (see DATA_PROCESSING.md)
```

A "contiguous 1-min edge" = two successive rows exactly 1 minute apart (a >60 s gap breaks the edge,
so nothing is dropped across a real gap). Steps [1]–[4] are `clean_streams()`; step [5] is the
day-signal builder.

---

## 1. Priming removal
`build_streams._apply_priming_downtime`. The machine reports `RUN_TIME = 0` while priming (before
blood is circulating). Drop minute *i* when run-time is zero on a 1-min edge:

```
drop_i  ⇐  (Δt = 1 min)  ∧  (RUN_TIME_i = 0)  ∧  (RUN_TIME_{i−1} = 0)
```

---

## 2. Downtime removal
Downtime = minutes where the circuit is paused / not actually delivering therapy. Two selectable
rules:

- **flat** — used by the canonical **"current"** stream. Drop minute *i* if effluent did not advance
  (no delivery that minute):
  ```
  drop_i  ⇐  (Δt = 1 min)  ∧  (EFFLUENT_i = EFFLUENT_{i−1})
  ```

- **dose** — used by the **Th70 / Th80 / Th90** ablation streams (and the leave-one-out study's
  default). Drop minute *i* if the delivered effluent that minute falls below a fraction `Th` of the
  prescribed dose:
  ```
  ΔEFFLUENT_i = EFFLUENT_i − EFFLUENT_{i−1}
  threshold   = Th · (D · W) / 60      [mL/min],     D = 25 mL/kg/h (prescribed),  W = weight (kg)
  drop_i      ⇐  (Δt = 1 min)  ∧  (W known)  ∧  (ΔEFFLUENT_i < threshold)
  ```
  `Th ∈ {0.70, 0.80, 0.90}`; patients with unknown weight (~25 %) skip this rule (priming still
  removed). Approx. rows removed: **Th70 3.4 % · Th80 5.1 % · Th90 8.4 %**.

---

## 3. Peak removal / denoising (window-based outlier smoothing)
`smoothing.detect_and_smooth_outliers_skip_nan`, params `window="40min", threshold=2,
neighbor_count=4, min_periods=1, fill_nan=False`. Alarm-driven pressure spikes are detected and
locally smoothed:

1. **Windowing** — per (PID, Visit, channel), partition the series into **non-overlapping 40-minute
   windows** (`timestamp.floor("40min")` — *tumbling* windows, not a rolling/sliding window).
2. **Outlier flag** — standardize each value against its window stats and flag spikes:
   ```
   z_t = (x_t − μ_w) / σ_w ,   outlier_t  ⇐  (σ_w > 0)  ∧  (|z_t| > 2)  ∧  (x_t not NaN)
   ```
3. **Replacement** — each flagged value → **mean of its ≤4 nearest valid neighbours** (up to 2
   before + 2 after; the search radius expands until 4 are found). Fallback order if no neighbours:
   the (PID, Visit) channel mean, else keep the original. Missing values are **not** imputed here
   (`fill_nan=False`).

`denoised` = peaks smoothed (the primary stream, `CRRT_pressure_wo_peak.csv`); `noised` = step [3]
skipped, peaks kept (`CRRT_pressure_with_peak.csv`).

---

## 4. Day numbering & cohort cap
`build_streams._compute_day_cap14`: within each (PID, Visit_no), `Day = dense_rank(calendar date)`.
Rows with `Day > 14` are dropped at clean time. The **modelling cohort further restricts to
`Day ≤ 7`** (`MAX_TREATMENT_DAY = 7`, the first treatment week). Output columns:
`PID, Visit_no, Day, Time, {ACCESS, FILTER, EFFLUENT, RETURN}` in physical units.

---

## 5. Missing / gap handling (gap-aware day signal)
`day_signal._gap_fill_day` (downstream of cleaning). Each circuit-day is placed on a fixed 0–1439
minute grid; absent minutes are missing. For a run of `g` consecutive absent minutes between two
observed minutes *a*, *b*:

```
g ≤ 10            →  linear interpolation  x_t = x_a + (x_b − x_a)(t − a)/(b − a),   mask = 1
g > 10, or leading / trailing absence      →  left missing (NaN),                    mask = 0
```

So short dropouts (≤10 min, e.g. brief sensor gaps) are filled and counted valid; longer gaps and
the unobserved head/tail of the day are masked out and ignored by the masked encoders/pooling.
Derived channels `TMP = (F+R)/2 − E`, `PFD = F − R`, `ARD = R − A` are computed from the filled
signal → `X[6,1440] + mask`. Per-channel z-score is fit on **TRAIN valid minutes only**; masked
minutes are set to 0. (Equations: `DATA_PROCESSING.md` §B.)

---

## 6. Ablation — what was varied

### 6a. Denoise-variant sweep (`abl_config.DENOISE_VARIANTS`)
Downtime strategy × peak handling → **8 streams**:

| variant | downtime rule | × peak | streams |
|---|---|---|---|
| current | flat (effluent unchanged) | denoised / noised | 2 |
| Th70 | dose, Th = 0.70 | denoised / noised | 2 |
| Th80 | dose, Th = 0.80 | denoised / noised | 2 |
| Th90 | dose, Th = 0.90 | denoised / noised | 2 |

Isolates: (i) how aggressively downtime is removed (flat vs dose at 70/80/90 %), and (ii) whether
smoothing alarm peaks helps (denoised vs noised).

### 6b. Leave-one-out (`abl_config.LOO_CONFIGS`)
Independently toggle the three cleaning steps (downtime via the dose rule, Th = 0.8):

| config | priming | downtime | peak | tests |
|---|---|---|---|---|
| all3 | removed | removed | removed | baseline (all cleaning on) |
| no_priming | kept | removed | removed | contribution of priming removal |
| no_downtime | removed | kept | removed | contribution of downtime removal |
| no_peak | removed | removed | kept | contribution of peak smoothing |

### 6c. Models & findings
The ablation is run on the best machine-only models — **DecisionTree (1-day)** and
**RandomForest (2-day)**. Headline outcomes: *denoised ≥ noised* in general; the best machine-only
configuration is **RF 2-day, current_denoised (AUROC 0.662)**; in the leave-one-out, **removing all
three** cleaning steps (all3) is best — each of priming, downtime, and peak removal contributes.
Full numbers: `RESULTS_COMPENDIUM.md` **Table 4** (denoise sweep) and **Table 5** (leave-one-out).

The same two sweeps were also run on the **best machine-only model — the ensemble** (rich-classical +
attention-CNN1D): it is robust to preprocessing (1-day 0.59–0.63, 2-day 0.62–0.66 across all 12
configs), peak removal helps most at 1-day, and `current_denoised` is well-justified. See
`RESULTS_COMPENDIUM.md` **Table 5b** and `exp_3/ablation/README.md`.

---

## 7. Implementation pointers
| step | code |
|---|---|
| priming + downtime + day-cap + peak orchestration | `preprocessing/build_streams.py` → `clean_streams()` |
| window-based peak smoothing | `preprocessing/smoothing.py` → `detect_and_smooth_outliers_skip_nan()` |
| ablation knobs (variants, LOO, dose, day cap) | `abl_config.py` → `DENOISE_VARIANTS`, `LOO_CONFIGS`, `EFFLUENT_DOSE_ML_KG_HR`, `MAX_TREATMENT_DAY` |
| gap-aware day signal + derived channels | `exp_3/features/day_signal.py` |
| feature equations (downstream) | `DATA_PROCESSING.md` |

*Parameter quick-reference:* 1-min grid · priming RUN_TIME=0 edge · downtime flat (current) / dose
Th 0.7–0.9 (dose 25 mL/kg/h) · peak 40-min tumbling window, |z|>2, 4 neighbours · Day-cap 14 →
modelling Day ≤ 7 · gap ≤10 min interpolate / >10 mask · z-score on train valid minutes.
