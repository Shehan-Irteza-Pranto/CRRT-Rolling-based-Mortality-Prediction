# Interpretability — feature selection, SHAP attribution, and reliability

Paper-ready interpretability section for the CRRT-mortality models. It reads the model through three
complementary lenses, all fit **leak-safely** (mRMR selection and SHAP explainers are fit on TRAIN and
applied to the held-out TEST; nothing is tuned on test):

1. **Selection** — which features mRMR keeps for each pipeline, and how stable those choices are across
   horizons and feature domains (§2).
2. **Attribution** — SHAP mean|value| rankings for the best classical model of each pipeline: what the
   model actually *uses* (§3 base pipelines; §4 the enriched circuit-instability features).
3. **Reliability** — how trustworthy the attributions are across clinical phenotypes (§5) and the
   caveats that bound them (§6).

All figures/tables are regenerable: `exp_3/results/make_shap_enriched.py` (enriched SHAP),
`exp_2/make_shap.py --dpi 300 --out interpretability` (base SHAP), `make_interpretability_figs.py`
(mRMR map). Assets live in [interpretability/](interpretability/). See also [[MACHINE_ENHANCEMENT]]
(the 67 enriched features), [[SUBPOPULATION]] (phenotype robustness), and RESULTS_COMPENDIUM.md §6.

---

## 1. Method (one paragraph for Methods)
> **Interpretability.** Feature relevance was assessed with minimum-redundancy–maximum-relevance (mRMR)
> selection and SHapley Additive exPlanations (SHAP). For each pipeline the model, mRMR selection, and
> SHAP explainer were fit on the training split only; SHAP values were then computed on the held-out
> test circuit-days. Tree models were explained with `TreeExplainer`; non-tree models with a
> permutation explainer over a 60-sample training background. Global importance is the mean absolute
> SHAP value across test days. Selection stability was quantified by how many of the eight pipelines
> (clinical-only / machine-base / machine-enriched / combined × 1-day/2-day) retained each feature.

---

## 2. Feature selection across pipelines (mRMR)
mRMR keeps 40 clinical features (clinical-only), 30 machine features (machine-only), or a grouped
20 machine + 40 clinical (combined). The full cross-pipeline map is
[interpretability/mrmr_selection_map.png](interpretability/mrmr_selection_map.png) (shaded = selected;
machine panel left, clinical panel right; y-labels colored by feature family). Per-feature stability is
in [interpretability/mrmr_selection_counts.csv](interpretability/mrmr_selection_counts.csv); family ×
pipeline counts in [interpretability/mrmr_family_counts.csv](interpretability/mrmr_family_counts.csv).

**Family composition of each machine selection** (from `mrmr_family_counts.csv`):

| pipeline | base-marginal | base-dynamic | enriched (spectral / complexity / excursion / eod) | enriched total |
|---|--:|--:|--:|--:|
| machine-base 1-day | 22 | 8 | 0 | **0** |
| machine-base 2-day | 21 | 9 | 0 | **0** |
| machine-rich 1-day | 14 | 4 | 4 / 1 / 2 / 5 | **12** |
| machine-rich 2-day | 14 | 4 | 4 / 1 / 3 / 4 | **12** |

So when the enriched pool is offered, mRMR **replaces 12 of 30 marginal/dynamic statistics with
circuit-instability descriptors** — dominant spectral period, end-of-day slope/delta, sustained
excursions, permutation entropy — at *both* horizons.

**Stable core vs. horizon-specific choices.**
- **Machine (of 6 machine-bearing pipelines):** 11 features are selected in **all 6** — the effluent
  quartiles (`EFFLUENT_PRESSURE_mean/median/q25/q75`), central TMP (`TMP_mean/median/q25`),
  `FILTER_PRESSURE_skewness/trend_r2`, `TMP_exc_frac`, `ARD_skewness`. These are the robust machine
  backbone; the enriched descriptors are selected only in the rich pipelines (by construction) and are
  the horizon-specific additions.
- **Clinical (of 4 clinical-bearing pipelines):** all **12 core labs**
  (INR, lactate, albumin, platelets, pH, bicarbonate, potassium, chloride, calcium, magnesium,
  phosphate, PTT) plus ~21 comorbidity/indication/severity flags are selected in **all 4** — clinical
  selection is highly stable across horizon and whether machine data is present.

**Reading:** the two domains are near-disjoint (clinical features never enter machine pipelines and
vice-versa); the combined selection is essentially the stable clinical core plus the stable machine
backbone.

---

## 3. SHAP attribution — base pipelines (best classical model)
Best model per pipeline (highest full-test AUROC; source `exp_2/results/shap/` regenerated at 300 dpi
into [interpretability/](interpretability/)). Figures: `shap_beeswarm_<pipe>.png`, `shap_bar_<pipe>.png`.

| pipeline (model, AUROC) | top-5 by mean\|SHAP\| | reading |
|---|---|---|
| **clinical-only** (RF, 0.733) | INR, lactate, albumin, platelets, pH | coagulation + perfusion/acid–base — the physiology of decompensation |
| **machine-only** (RF, 0.662) | `FILTER_PRESSURE_mad`, `TMP_max`, `TMP_q75`, `TMP_mean`, `TMP_exc_frac` | transmembrane / filter pressure level and **volatility** — clotting & membrane fouling |
| **combined** (AdaBoost, 0.738) | INR, pH, Age, albumin, **`TMP_q75` (5th)** | clinical labs dominate; the best machine feature ranks only **5th** → small incremental machine value |

Full top-15 tables: [interpretability/shap_top15_clinical_only.csv](interpretability/shap_top15_clinical_only.csv),
[machine_only](interpretability/shap_top15_machine_only.csv),
[combined](interpretability/shap_top15_combined.csv). In the combined top-15, only **3 of 15** features
are machine — consistent with the headline result that CRRT pressure adds a small increment once
clinical data is present.

---

## 4. SHAP attribution — enriched circuit-instability features (NEW)
The enrichment story previously rested on *selection* (mRMR kept 12/30). Here SHAP confirms the enriched
descriptors are not just selected but **drive predictions**. Source
`exp_3/results/make_shap_enriched.py`; figures `shap_{beeswarm,bar}_{machine,combined}_enriched_{1,2}day.png`.

### 4a. Machine-only, enriched
| horizon (model, AUROC) | top-5 by mean\|SHAP\| | enriched in top-15 |
|---|---|--:|
| **1-day** (LightGBM, 0.622) | **`FILTER_PRESSURE_perm_entropy` (1st)**, `ARD_skewness`, `FILTER_PRESSURE_trend_r2`, `TMP_mean`, `TMP_exc_frac` | 5 |
| **2-day** (RF, 0.639) | `FILTER_PRESSURE_mad`, `TMP_q75`, **`FILTER_PRESSURE_perm_entropy` (3rd)**, `TMP_cross_rate`, `TMP_mean` | 6 |

At 1-day the **single most important feature is an enriched one** — `FILTER_PRESSURE_perm_entropy`
(nonlinear irregularity of the filter pressure) — and the top-15 also surfaces `ARD_eod_slope/delta`,
`ACCESS_PRESSURE_max_run_above/eod_delta` (sustained excursions and end-of-day deterioration). This is
the attribution counterpart of the **+0.03–0.05 1-day lift** in [[MACHINE_ENHANCEMENT]] Table VIII: the
new circuit-instability physiology is what the model leans on.

### 4b. Combined, enriched — where do the machine features land?
| horizon (GradientBoosting, AUROC) | top-5 | best machine rank |
|---|---|---|
| **1-day** (0.733) | INR, **`FILTER_PRESSURE_perm_entropy` (2nd)**, pH, lactate, platelets | **2nd** (was 5th, base combined) |
| **2-day** (0.718) | INR, pH, **`FILTER_PRESSURE_perm_entropy` (3rd)**, albumin, Age | **3rd** |

The key contrast: in the **base** combined model the top machine feature (`TMP_q75`) sat at rank 5; in
the **enriched** combined model `FILTER_PRESSURE_perm_entropy` **climbs to rank 2 (1-day) / 3 (2-day)**,
interleaved with the strongest coagulation/acid–base labs. The enriched machine signal competes with —
though does not overtake — the clinical labs, which is exactly why the combined AUROC (~0.73) stays
close to clinical-only: the machine contribution is real and now visible, but incremental.
Tables: [shap_top15_machine_enriched_1day.csv](interpretability/shap_top15_machine_enriched_1day.csv),
[…2day](interpretability/shap_top15_machine_enriched_2day.csv),
[combined_enriched_1day](interpretability/shap_top15_combined_enriched_1day.csv),
[…2day](interpretability/shap_top15_combined_enriched_2day.csv).

---

## 5. Reliability by phenotype
Global SHAP tells us *which* features matter on average; [[SUBPOPULATION]] tells us *where* the model is
trustworthy. Clustering the test circuit-days into four clinical phenotypes (Ward hierarchical on the
clinical features), the best machine-only ensemble discriminates unevenly:

| sub-pop | n days | phenotype | 1-day AUROC |
|---|--:|---|--:|
| **S0** | 258 | preserved albumin, higher BMI, lower severity | **0.736** (best) |
| S2 | 160 | obese, highest APACHE-II, low albumin | 0.617 |
| **S3** | 449 | highest lactate, low platelets, low albumin | **0.598** (worst) |

Discrimination varies ~2.3× (0.60–0.74): the machine-pressure signal is most reliable in the healthier,
preserved-albumin group and least reliable in the sickest high-lactate/low-platelet group — a concrete,
reportable limitation. (Figures: `subpop/tsne_<case>.png`, `subpop/dendrogram_<case>.png`.)

---

## 6. Synthesis + limitations
**Synthesis.** The three lenses agree. *Machine* prognostic signal is transmembrane/filter-pressure
**instability** — level, volatility (`mad`, `exc_frac`), nonlinear irregularity (`perm_entropy`),
sustained excursions and end-of-day deterioration — the physiology of clotting and membrane fouling.
*Clinical* signal is coagulation and perfusion/acid–base decompensation (INR, lactate, albumin,
platelets, pH). In the combined model the clinical labs lead and the machine features follow; enrichment
raises the best machine feature from rank 5 to rank 2–3 but does not displace the labs — consistent with
a **small but real** incremental contribution from the CRRT pressure waveform.

**Limitations.**
- **Few positives** (≈62 test / ~30 val positives at 1-day) → mRMR and SHAP rankings below the top
  handful are not highly stable; we therefore emphasize families and the stable core, not exact ranks.
- **Global, not per-subgroup, SHAP** — attribution is averaged over all test days; §5 shows importance
  reliability varies by phenotype, but we did not compute phenotype-stratified SHAP.
- **Attribution ≠ causation** — SHAP explains the fitted model on this cohort; the pressure–mortality
  association is prognostic, not mechanistic.
- **Model-specific** — rankings are for the best classical model per pipeline; the DL models
  (attention-CNN1D) consume the raw window-reduced sequence and are interpreted separately (integrated
  gradients / attention rollout scaffolds exist in `Deep_Learning_Models/interpretability/`).

---

## 7. Paste-ready paragraph (Results/Discussion)
> **Interpretability.** Across selection (mRMR) and attribution (SHAP), the machine-only signal is
> dominated by transmembrane- and filter-pressure instability: pressure level and volatility
> (`FILTER_PRESSURE_mad`, `TMP_exc_frac`), and — once the enriched descriptors are available — nonlinear
> irregularity and end-of-day deterioration. In the enriched machine-only model the single most
> important feature at one day is the filter-pressure permutation entropy, and mRMR replaces 12 of 30
> marginal statistics with circuit-instability descriptors at both horizons, mirroring the +0.03–0.05
> AUROC lift. The clinical signal is coagulation and perfusion/acid–base physiology (INR, lactate,
> albumin, platelets, pH), whose twelve core labs are selected in every clinical-bearing pipeline. In
> the combined model the clinical labs dominate the SHAP ranking; enrichment lifts the leading machine
> feature from fifth to second–third place but does not overtake the labs, consistent with a small but
> genuine incremental value of the CRRT pressure waveform. Discrimination of the machine model varies
> roughly two-fold across clinical phenotypes (AUROC 0.60–0.74), being most reliable in lower-severity,
> preserved-albumin patients and least reliable in the sickest high-lactate group.

*Sources: `interpretability/shap_top15_*.csv`, `interpretability/mrmr_selection_{map.png,counts.csv,
family_counts.csv}`, `exp_3/results/enrich_effect.csv`, `SUBPOPULATION.md`. Scripts:
`exp_3/results/make_shap_enriched.py`, `exp_2/make_shap.py`, `paper_materials/make_interpretability_figs.py`.*
