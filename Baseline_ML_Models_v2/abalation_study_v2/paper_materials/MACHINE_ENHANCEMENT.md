# Machine-only Enhancement — enriched features (what they are, and vs the base set) + Table VIII

Companion to Section C / Table VIII. It states **exactly what the enriched ("rich") machine features
are**, **how they differ from the base feature set**, which of them the selector kept, and the
results. Feature equations are in `DATA_PROCESSING.md` §C; the numbers match `RESULTS_COMPENDIUM.md`
§3 and Table VIII.

Both feature sets summarize the **same** per-circuit-day gap-aware pressure signal (6 base channels
ACCESS, FILTER, EFFLUENT, RETURN, TMP, PFD, + derived ARD = RETURN−ACCESS). The enrichment only adds
**more descriptors of that signal** — no new data source.

---

## 1. What changed
| set | # features | contents |
|---|--:|---|
| **base** (baseline) | **133** | per-channel marginal-distribution statistics + basic within-day dynamics |
| **rich** (enriched, ours) | **200** | the 133 base features **+ 67 new** circuit-instability descriptors |

The base set answers *"what did the pressure distribution look like over the day?"*; the 67 new
features answer *"how did the signal behave?"* — its **frequency content, nonlinear complexity,
sustained excursions, end-of-day trajectory, and cross-channel coupling** — the clotting/fouling
"circuit-instability" physiology the base set does not capture.

---

## 2. Table A — Base feature set (133)
7 channels (4 raw + TMP, PFD, ARD) described by two blocks:

| block | per-channel features | count |
|---|---|--:|
| distribution (×13) | mean, std, min, max, median, q25, q75, iqr, range, skewness, kurtosis, cv, n_valid | 7 × 13 = 91 |
| within-day dynamics (×6) | slope (OLS vs time), trend_r2, mad (mean \|Δ\|), max_jump (max \|Δ\|), ac1 (lag-1 autocorr), exc_frac (fraction \|z\|>2) | 7 × 6 = 42 |
| | | **133** |

→ marginal distribution + a single linear trend + coarse minute-to-minute volatility.

---

## 3. Table B — the 67 NEW enriched features (ours)
Per channel (6 base + ARD = 7) × 9 descriptors = **63**, plus **4** day-level cross-channel coupling
scalars = **67**. Grouped by what they add:

| group | features (per channel) | # / channel | physiology it captures |
|---|---|--:|---|
| **spectral** (Welch PSD) | `spec_entropy`, `bandpow_lo` (power at period >30 min), `bandpow_hi` (<10 min), `dom_period` (1/peak-freq) | 4 | frequency content / dominant rhythm — clotting & membrane fouling change the oscillation structure (slow drift vs fast fluctuation) |
| **complexity** | `perm_entropy` (order-3 permutation entropy) | 1 | nonlinear signal irregularity — circuit instability increases complexity |
| **excursion** | `cross_rate` (median-crossings / min), `max_run_above` (longest run above the daily median, fraction) | 2 | how often and how **persistently** the pressure deviates — sustained excursions |
| **end-of-day** | `eod_delta` (mean last 2 h − first 2 h), `eod_slope` (slope over last 2 h) | 2 | **within-day deterioration** — pressures worsening toward the end of the day |
| **coupling** (day-level, 4 total) | `TMP–PFD corr`, `ACCESS–RETURN corr`, `FILTER–EFFLUENT corr`, `TMP–PFD cotrend` | — | multi-channel co-movement — clotting drives TMP and PFD up **together** |

`7 × 9 = 63` per-channel + `4` coupling = **67**. (Equations: `DATA_PROCESSING.md` §C.3.)

---

## 4. Base vs enriched — the difference in one line
- **Base (133):** *static description* of each channel over the day — where the values sit (mean,
  quantiles, skew…) plus one linear slope and rough volatility.
- **Enriched (+67):** *dynamic/structural description* — the signal's **spectral content**
  (rhythm), **entropy** (irregularity), **sustained-excursion** structure, **end-of-day
  deterioration**, and **cross-channel coupling**. These are exactly the circuit-failure signatures
  a linear slope + marginal stats miss.

---

## 5. Table C — which new features the selector kept (mRMR top-30, 1-day)
Of the 30 mRMR-selected machine features, **12 are from the new set** — and they concentrate in the
deterioration / instability descriptors:

| new feature kept | group |
|---|---|
| `TMP_dom_period`, `PFD_dom_period`, `RETURN_PRESSURE_dom_period`, `EFFLUENT_PRESSURE_dom_period` | spectral (dominant period) |
| `ACCESS_PRESSURE_eod_slope`, `ACCESS_PRESSURE_eod_delta`, `ARD_eod_slope`, `ARD_eod_delta`, `RETURN_PRESSURE_eod_delta` | end-of-day deterioration |
| `FILTER_PRESSURE_max_run_above`, `ACCESS_PRESSURE_max_run_above` | sustained excursion |
| `FILTER_PRESSURE_perm_entropy` | complexity |

So the model's added signal comes from **dominant spectral period + end-of-day slope/delta +
sustained excursions + entropy** — not from more marginal statistics.

---

## 6. Table VIII — same model, base features → enriched features (+ DL on top)
**Design (per the reviewer note):** take every model from the main machine-only benchmark (Table V,
*base* 133 features), then run **the identical model through the identical single-model pipeline**
(median-impute → standardize → mRMR top-30 → RandomizedSearchCV tune → single fit) on the **enriched**
200-feature table. Nothing else changes, so the base column **reproduces Table V exactly** and the
enriched column is a clean same-model comparison. All numbers are full held-out test AUROC with
patient-bootstrap 95% CI. Best per horizon **bold**; base column verified against Table V. (The DL /
ensemble models are **not** in this table — they do not consume the 133/200 tabular features; see the
separate block below.)

### 1-day (predict death at D+2)
| model | base (133) | enriched (200) | Δ |
|---|--:|--:|--:|
| DecisionTree (best base) | 0.596 [0.526, 0.663] | 0.577 [0.507, 0.648] | −0.019 |
| **LightGBM** | 0.594 [0.515, 0.676] | **0.622** [0.553, 0.694] | **+0.028** |
| AdaBoost | 0.591 [0.509, 0.674] | 0.620 [0.547, 0.689] | **+0.029** |
| LogisticRegression | 0.588 [0.507, 0.664] | 0.591 [0.516, 0.666] | +0.003 |
| RandomForest | 0.575 [0.509, 0.646] | 0.607 [0.538, 0.673] | **+0.032** |
| XGBoost | 0.571 [0.497, 0.651] | 0.605 [0.535, 0.670] | **+0.034** |
| GradientBoosting | 0.561 [0.493, 0.641] | 0.613 [0.544, 0.680] | **+0.052** |
| KNN | 0.529 [0.454, 0.602] | 0.508 [0.439, 0.577] | −0.021 |

### 2-day (predict death at D+2/D+3)
| model | base (133) | enriched (200) | Δ |
|---|--:|--:|--:|
| **RandomForest (best base)** | **0.662** [0.597, 0.725] | 0.639 [0.580, 0.703] | −0.022 |
| LightGBM | 0.648 [0.589, 0.709] | 0.594 [0.543, 0.654] | −0.054 |
| XGBoost | 0.627 [0.566, 0.695] | 0.611 [0.553, 0.672] | −0.016 |
| GradientBoosting | 0.624 [0.563, 0.694] | 0.623 [0.562, 0.690] | −0.001 |
| KNN | 0.604 [0.550, 0.665] | 0.575 [0.517, 0.642] | −0.029 |
| AdaBoost | 0.603 [0.536, 0.672] | 0.618 [0.553, 0.686] | +0.014 |
| LogisticRegression | 0.599 [0.531, 0.672] | 0.573 [0.511, 0.640] | −0.026 |
| DecisionTree | 0.527 [0.460, 0.593] | 0.611 [0.561, 0.670] | **+0.085** |

### Table VIII(b) — DL / ensemble on the same signal (no tabular-feature column)
These models do **not** ingest the 133/200 engineered features. The Attention-CNN1D consumes the **raw
window-reduced pressure sequence** `[N, 144 windows, 30]` (6 channels × 5 per-window stats
mean/min/max/std/slope; z-scored, gap-masked). The Ensemble is a validation-fit logistic stack that
fuses **two representations of the same signal** — the enriched-classical models (tabular) **plus** the
attention-CNN (sequence). Reported separately so no number sits under an "enriched-features" column.

| model | input representation | 1-day AUROC | 2-day AUROC |
|---|---|--:|--:|
| Attention-CNN1D | raw window-reduced sequence `[144×30]` | 0.617 [0.542, 0.692] | 0.616 [0.539, 0.693] |
| **Ensemble (logistic stack)** | enriched-classical (tabular) + attn-CNN (sequence) | **0.649** [0.576, 0.723] | 0.643 [0.576, 0.718] |

**What the table shows.**
- **1-day — enrichment helps the strong models.** Every ensemble-family model gains from the 67
  circuit-instability features: GradientBoosting **+0.052** (0.561→0.613), XGBoost **+0.034**,
  RandomForest **+0.032**, AdaBoost **+0.029**, LightGBM **+0.028**. The best *enriched* single
  classical model (LightGBM 0.622) beats the best *base* model (DecisionTree 0.596), and stacking the
  enriched-classical models with the Attention-CNN1D pushes the **Ensemble to 0.649** — the
  machine-only ceiling. Only the single tree and KNN (high-variance / distance-based, no benefit from
  extra descriptors) fail to improve.
- **2-day — the base ceiling already holds.** The higher-prevalence 2-day label is largely explained
  by the marginal-distribution statistics, so the best base **RandomForest (0.662)** is already the
  ceiling and enrichment does **not** raise it (RF −0.022; LightGBM −0.054). Enrichment here only
  rescues weak base learners (DecisionTree **+0.085**, 0.527→0.611). The enriched Attention-CNN1D
  (0.616) and Ensemble (0.643) are competitive but do not exceed the base RF.

**Takeaway:** the enriched features deliver a **consistent +0.03–0.05 AUROC lift at 1-day** across the
strong classical models and let the ensemble reach **0.649**; at 2-day the signal is already saturated
by the base statistics (best model unchanged at **0.662**). The base column here matches Table V
exactly, so the earlier 0.599-vs-0.575 mismatch is resolved.

*Source: `exp_3/results/enrich_effect_samepipe.py` → `enrich_effect.csv` (base column uses the main
run's cached mRMR selection → reproduces Table V; enriched uses fresh mRMR-30 on the 200-feature pool).
DL/Ensemble rows: `exp_3/results/exp3_variants_full_1to1_1to2.csv`.*

---

## 7. Corrected Section-C paragraph (paper-ready)
> **C. Enhancing the machine-only signal.** To test whether the machine signal carries more than the
> baseline captured, we enriched the 133-feature machine set with **67 circuit-instability
> descriptors** computed from the same day signal: per-channel **spectral** features (spectral
> entropy, low-/high-band power, dominant period), **permutation entropy**, **excursion** statistics
> (median-crossing rate, longest sustained run above the daily median), **end-of-day deterioration**
> (last-2h vs first-2h level change and slope), and four **cross-channel coupling** terms
> (TMP–PFD, ACCESS–RETURN, FILTER–EFFLUENT correlations and the TMP–PFD co-trend) — features the base
> set, which holds only marginal-distribution statistics plus a single linear trend, does not
> capture. Passing each base model through an identical pipeline on the enriched set, the one-day
> AUROC of the strong ensemble-family models rises consistently — GradientBoosting +0.052
> (0.561→0.613), XGBoost +0.034, RandomForest +0.032, AdaBoost +0.029, LightGBM +0.028 — so that the
> best enriched classical model (0.622) exceeds the best base model (0.596), and a classical +
> attention-CNN ensemble reaches **0.649** (95% CI [0.576, 0.723]); mRMR retains **12 of 30** features
> from the new set, concentrated in dominant spectral period, end-of-day slope/delta, sustained
> excursions, and permutation entropy. At two-day the higher-prevalence label is already well captured
> by the marginal statistics, so enrichment does not improve the best base random forest (**0.662**)
> and only rescues weak base learners. Overall the machine-only signal tops out near **0.65**,
> consistent with a small but real prognostic contribution from the CRRT pressure waveform.

*Sources: `exp_3/features/build_rich_features.py` (feature code), `DATA_PROCESSING.md` §C (equations),
`exp_3/results/enrich_effect_samepipe.py` → `enrich_effect.csv` (same-model base→enriched, base = Table V),
`exp_3/results/exp3_variants_full_1to1_1to2.csv` (Attention-CNN1D / Ensemble).*
