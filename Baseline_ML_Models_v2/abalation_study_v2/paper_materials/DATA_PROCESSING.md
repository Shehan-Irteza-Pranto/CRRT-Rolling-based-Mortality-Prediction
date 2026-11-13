# Data & Signal Processing — full specification with equations

Every preprocessing step from the raw CRRT device export to the model-ready representations, written
for the paper's Methods section. Each transform is given as an equation where one applies, with the
exact parameter values and source file. This is the **equation-complete companion** to
`../exp_3/ARCHITECTURE.md` (block diagram) and `DL_METHODS.md` (model layers).

Notation: a circuit-day is one CRRT circuit on one calendar day for one patient. Within a day, time
is the minute index `t ∈ {0,…,1439}`. Raw channels are A=ACCESS, F=FILTER, E=EFFLUENT, R=RETURN
pressure. `m_t ∈ {0,1}` is a validity mask.

---

## A. Raw acquisition → cleaned ("denoised") stream
Source: `preprocessing/build_streams.py`, `preprocessing/smoothing.py`, `abl_config.py`.
> For the cleaning pipeline as stages and the full **ablation matrix** (denoise variants +
> leave-one-out), see **`SIGNAL_PROCESSING_PIPELINE.md`**; this section gives the same steps with the
> equations.

Input is already on a **1-minute grid** (4 raw pressure channels + `RUN_TIME`, `EFFLUENT` counters);
no resampling is performed. A "contiguous 1-min edge" means two successive rows exactly 1 minute
apart (a gap >60 s breaks the edge, so nothing is dropped across a gap).

**A.1 Priming removal.** Drop minute *i* (machine still priming) when run-time is zero on a 1-min edge:

  drop_i = (Δt = 1 min) ∧ (RUN_TIME_i = 0) ∧ (RUN_TIME_{i−1} = 0)

**A.2 Downtime removal.** Two rules; the canonical "current" stream uses the *flat* rule, the
ablation variants Th70/Th80/Th90 use the *dose-based* rule.

- *Flat* (current/default): drop minute *i* if no effluent was delivered in that minute:

    drop_i = (Δt = 1 min) ∧ (E_i = E_{i−1})

- *Dose-based* (ablation Th ∈ {0.70, 0.80, 0.90}): drop minute *i* if the delivered effluent falls
  below a fraction `Th` of the prescribed dose,

    ΔE_i = E_i − E_{i−1},   threshold = Th · (D · W) / 60   [mL/min],
    drop_i = (Δt = 1 min) ∧ (W known) ∧ (ΔE_i < threshold)

  with prescribed dose **D = 25 mL/kg/h** and patient weight **W** (kg). Patients with unknown W
  (~25 %) skip this rule (priming still removed). Rows removed: Th70 ≈ 3.4 %, Th80 ≈ 5.1 %,
  Th90 ≈ 8.4 %. Default `BEST_TH = 0.80`.

**A.3 Peak denoising / smoothing** (`detect_and_smooth_outliers_skip_nan`; window = 40 min,
z-threshold = 2, neighbour_count = 4). For each (PID, Visit, channel) and each non-overlapping
40-minute window `w`, standardize each value against its window statistics and flag outliers:

  z_t = (x_t − μ_w) / σ_w,   outlier_t = (σ_w > 0) ∧ (|z_t| > 2) ∧ (x_t not NaN)

Each flagged value is replaced by the mean of its nearest valid neighbours (up to 2 before + 2
after; search radius expands until 4 are found):

  x_t ← mean( {x_j : j ∈ nearest valid neighbours, |neighbours| ≤ 4} )

Fallback order if no neighbours: the (PID, Visit) channel mean, else the original value. Missing
values are **not** imputed here (`fill_nan = False`). This yields the **denoised** stream
(`CRRT_pressure_wo_peak.csv`); keeping the peaks gives the **noised** stream
(`CRRT_pressure_with_peak.csv`).

**A.4 Day numbering & cohort cap.** Within (PID, Visit), `Day = dense_rank(calendar date)`. At clean
time rows with `Day > 14` are dropped; the **modelling cohort further restricts to Day ≤ 7**
(`MAX_TREATMENT_DAY = 7`). Output columns: `PID, Visit_no, Day, Time, {A,F,E,R}` in physical units.

---

## B. Per-circuit-day gap-aware signal
Source: `exp_3/features/day_signal.py` (physical units, Branch A) and `dl/dl_data.py` (z-scored, Branch B).

**B.1 Calendar-minute grid.** Place each day's cleaned rows on `t ∈ {0,…,1439}` → matrix `[4,1440]`;
absent minutes are NaN with presence `p_t`.

**B.2 Gap rule.** Let two observed minutes `a < b` bracket a run of `g = b − a − 1` absent minutes.

  if g ≤ 10:   x_t = x_a + (x_b − x_a)·(t − a)/(b − a),  m_t = 1,   for a < t < b   (linear interp.)
  if g > 10, or t before first / after last observed minute:   x_t = NaN,  m_t = 0   (masked)

with `GAP_MAX_MIN = 10`.

**B.3 Derived physiology channels** (per minute):

  TMP_t = (F_t + R_t)/2 − E_t      (transmembrane pressure — membrane fouling)
  PFD_t = F_t − R_t                (filter pressure drop — clotting)
  ARD_t = R_t − A_t                (access–return span — access instability; Branch A only)

The model signal is `X = [A, F, E, R, TMP, PFD] ∈ ℝ^{6×1440}` with mask `m ∈ {0,1}^{1440}`.

**B.4 Normalization.**
- *Branch A* (engineered features) keeps **physical units** (thresholds, band-power, end-of-day
  deltas need the real scale); NaN minutes are ignored per-feature.
- *Branch B* (CNN/RNN) applies **per-channel z-score** with statistics from **TRAIN valid minutes
  only**:  x̃_{c,t} = (x_{c,t} − μ_c)/σ_c, with masked minutes set to 0 (= channel mean).

---

## C. Branch A — engineered features (200 per circuit-day), with equations
Source: `preprocessing/build_features.py` (133 parent), `exp_3/features/build_rich_features.py` (67 new).
Let `x = (x_1,…,x_n)` be a channel's valid (non-NaN) values, `x̄` its mean, `σ` its std.

**C.1 Distribution statistics — 13 per channel** (7 channels: A, F, E, R, TMP, PFD, ARD → 91):

| feature | equation |
|---|---|
| mean | x̄ = (1/n) Σ x_i |
| std | σ = √[(1/n) Σ (x_i − x̄)²] |
| min / max | min_i x_i  /  max_i x_i |
| median | median(x) |
| q25 / q75 | 25th / 75th percentile |
| iqr | q75 − q25 |
| range | max − min |
| skewness | E[(x − x̄)³] / σ³  (Fisher; 0 if n ≤ 2) |
| kurtosis | E[(x − x̄)⁴] / σ⁴ − 3  (excess; 0 if n ≤ 3) |
| cv | σ / x̄  (0 if x̄ = 0) |
| n_valid | n |

**C.2 Temporal-dynamics statistics — 6 per channel** (7 channels → 42). With time index
`τ = (0,1,…,n−1)` over the ordered valid samples and successive differences `Δx_i = x_{i+1} − x_i`:

| feature | equation |
|---|---|
| slope | OLS β in `x_i = α + β·τ_i + ε_i` |
| trend_r2 | 1 − SS_res/SS_tot = 1 − Σ(x_i − x̂_i)² / Σ(x_i − x̄)²  (0 if SS_tot = 0) |
| mad | (1/(n−1)) Σ \|Δx_i\| |
| max_jump | max_i \|Δx_i\| |
| ac1 | ρ(x_{1:n−1}, x_{2:n})  (lag-1 autocorrelation; 0 if σ = 0) |
| exc_frac | (1/n) Σ 𝟙[ \|z_i\| > 2 ],  z_i = (x_i − x̄)/σ  (0 if σ = 0) |

**C.3 New physiology features — 67** (per channel over 6 base + ARD = 63, plus 4 coupling). Let the
day's continuous segment (small gaps interpolated, §B.2) be `s`, sampled at 1/min.

*Spectral* — Welch power spectral density `P(f)`, dropping DC; normalized `p_f = P(f)/Σ_f P(f)`:

  spec_entropy = −Σ_f p_f ln p_f / ln N_f          (N_f = #frequency bins; ∈ [0,1])
  bandpow_lo   = Σ_{f < 1/30} p_f                   (fraction of power at period > 30 min)
  bandpow_hi   = Σ_{f > 1/10} p_f                   (fraction of power at period < 10 min)
  dom_period   = 1 / argmax_f P(f)                  (period of the dominant frequency, min)

*Complexity* — permutation entropy of order `d = 3`. For each ordinal pattern π of `d` consecutive
samples with empirical probability `q_π`:

  perm_entropy = −Σ_π q_π ln q_π / ln(d!)           (∈ [0,1])

*Excursion* — with daily median `x̃` and indicator `b_t = 𝟙[x_t > x̃]`:

  cross_rate    = (#{t : b_t ≠ b_{t−1}}) / n        (median-crossings per minute)
  max_run_above = (longest run of consecutive b_t = 1) / n

*End-of-day deterioration* — over the last/first 120 min of the segment:

  eod_delta = mean(s_{last 2h}) − mean(s_{first 2h})
  eod_slope = OLS slope over the last 2 h

*Cross-channel coupling* (over jointly-valid minutes):

  corr(TMP,PFD),  corr(A,R),  corr(F,E)             (Pearson r)
  cotrend = sign(β_TMP · β_PFD)                     (β = §C.2 slope; +1 if TMP & PFD trend together)

**C.4 Classical model preprocessing** (leak-safe, fit on TRAIN only): median imputation of any NaN
feature → standardization (zero mean, unit variance) → **mRMR** selection of the top **30** features
→ classifier. (mRMR = minimum-Redundancy Maximum-Relevance.)

---

## D. Branch B — sequence reduction before the encoder
Source: `models/dl_reduce.py`. The z-scored signal `[1440, C]` is reduced to a short sequence.

**D.1 Window aggregation** (CNN1D, RNN; window `k = 10` → 144 windows). For each window and channel,
over the valid samples `i` in the window (mask `m_i`, in-window time `t_i ∈ {0,…,k−1}`, count
`N = Σ m_i`):

  mean = (Σ m_i x_i)/N,   min = min_{m_i=1} x_i,   max = max_{m_i=1} x_i,
  std  = √[ (Σ m_i (x_i − mean)²)/N ],
  slope (mask-weighted OLS):
        t̄ = (Σ m_i t_i)/N,
        β_w = Σ m_i (t_i − t̄)(x_i − mean) / Σ m_i (t_i − t̄)²

Each window → 5 statistics per channel (feature dim = C·5; e.g. 6 channels → 30). An empty window
(N = 0) yields all-zero stats and window-mask = 0.

**D.2 Patchify** (Transformer; patch `P = 60`): non-overlapping 60-min patches are flattened
(feature dim = C·P) and linearly embedded by the encoder's input projection; a patch is valid if it
contains ≥1 valid minute.

---

## E. Cohort, label, and split
- **Prediction unit:** one circuit-day (one score per (PID, Visit, Day)).
- **Label horizon (rolling, gap-based).** For a circuit-day on calendar date *D*, let
  `dud = days_until_death` (calendar days from *D* to the death date). A **1-day blanking gap** (*D+1*)
  is always excluded so the task is genuine *early* warning, not trivial imminent-death detection:
  - **1-day:** survivor → 0; `dud ∈ {0,1}` (death day *D* + gap day *D+1*) → **excluded**;
    `dud = 2` → **1** (predict death on *D+2*); `dud ≥ 3` → 0.
  - **2-day:** `dud ∈ {0,1}` → **excluded**; `dud ∈ {2,3}` → **1** (predict death on *D+2* or *D+3*);
    `dud ≥ 4` → 0.
  Same input signal (day *D*); only the prediction horizon — and hence the positive prevalence
  (~6–7 % for 1-day vs ~11–12 % for 2-day) — differs. (Source: `Code/label_processing/
  label_1day_2day_7_day.ipynb`.)
- **Split:** patient-level **stratified 70:10:20** (positive-rate matched), `pid_splits_v2.json` —
  no patient spans two splits. Sizes: train 3236 / val 462 / test 887 circuit-days; 1-day positives
  201 / 30 / 62.
- **Evaluation:** held-out test only; AUROC/AUPRC/F1–F3; 95% CI by bootstrapping over **patients**
  (1000 resamples); decision threshold = max-F1 on validation.

---

*Parameter quick-reference:* gap ≤ 10 min interpolate · dose D = 25 mL/kg/h, Th ∈ {0.7,0.8,0.9} ·
smoothing 40-min window, |z|>2, 4 neighbours · derived TMP/PFD/ARD · z-score on train valid minutes ·
features 133 + 67 = 200 · mRMR top-30 · window k = 10 (5 stats) · patch P = 60 · Day ≤ 7 · split 70:10:20.
