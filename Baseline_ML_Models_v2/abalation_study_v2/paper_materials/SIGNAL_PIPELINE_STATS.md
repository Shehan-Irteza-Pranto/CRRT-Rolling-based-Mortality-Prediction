# Signal-processing pipeline — data provenance / attrition (results)

How much CRRT pressure data survives each step of the signal-processing pipeline, from the raw device
export to the model-ready cohort. Figure: **`figures/signal_pipeline_pie.png`** (donut — disposition of the
6.60 M raw minute-rows: 92.4% retained vs. removed by each step). All counts computed by
`signal_pipeline_stats.py` (→ `cohort/signal_pipeline_stats.csv`, `signal_pipeline_derived.csv`);
`draw_pipeline_pie.py` renders the figure.

## Attrition table
A **minute-row** = one minute of one circuit; a **circuit** = PID×Visit; a **circuit-day** = PID×Visit×Day.

| # | Stage | Minute-rows | Circuit-days | Circuits | Patients | Removed / changed |
|---|---|--:|--:|--:|--:|:--|
| 0 | Raw export (`machine_filtered_v1`) | 6,603,507 | — | 1,256 | 1,076 | — |
| 1 | Day numbering + cap (Day ≤ 14) | 6,236,988 | 5,698 | 1,256 | 1,076 | −366,519 rows (Day > 14) |
| 2 | Priming removal (RUN_TIME = 0) | 6,215,753 | 5,698 | 1,256 | 1,076 | −21,235 rows |
| 3 | Downtime removal (no effluent) | 6,102,758 | 5,696 | 1,256 | 1,076 | −112,995 rows; −2 circuit-days |
| 4 | Peak denoising (40-min, \|z\|>2) | 6,102,758 | 5,696 | 1,256 | 1,076 | 0 dropped; **764,322 minutes (12.5%) smoothed** |
| 5 | Gap-aware reconstruction (interp ≤10 min / mask >10) | 1440-min grid | 5,696 | — | — | **75.1% of minutes valid/day** (median 94.0%) |
| 6 | Day ≤ 7 cap + label merge → **cohort** | 4,585 | 4,585 | — | 976 | **293 deaths** |

## Cohort (Day ≤ 7, patient-disjoint split)
| Split | Circuit-days | Patients | Deaths |
|---|--:|--:|--:|
| Train | 3,236 | 682 | 201 |
| Validation | 462 | 98 | 30 |
| Test | 887 | 196 | 62 |
| **Total** | **4,585** | **976** | **293** |

## Key numbers per step (paste-ready)
- **Raw export:** 6.60 M minute-rows from **1,076 patients / 1,256 circuits**.
- **Day cap (≤14):** removes late-treatment tails → −366,519 rows (5.5%).
- **Priming removal** (drop RUN_TIME = 0 startup minutes): −21,235 rows (0.3%); no circuit-days lost.
- **Downtime removal** (drop minutes with no effluent delivered): −112,995 rows (1.8%); 2 circuit-days emptied.
- **Peak denoising** drops **no** rows — it smooths outliers in place: **764,322 minutes (12.5%)** had ≥1
  channel corrected, i.e. **5.0% of all channel-cells** replaced by the local-neighbour mean.
- **Gap-aware reconstruction:** gaps ≤10 min are linearly interpolated (kept valid), gaps >10 min and the
  leading/trailing minutes are masked → on average **75.1%** of the 1440 minutes per circuit-day are valid
  (median 94.0%; 74.3% of circuit-days are ≥50% complete).
- **Model-ready cohort (Day ≤ 7 + mortality labels):** **4,585 circuit-days, 976 patients, 293 deaths**
  (6.4% circuit-day prevalence), split patient-disjointly into 3,236 / 462 / 887.
- **Branches:** *A* = 200 engineered features/circuit-day (133 base + 67 enriched) → mRMR → classical;
  *B* = window-reduced sequence `[144 windows × 30]` (6 channels × 5 stats over 10-min windows) → DL.

*Notes: the pipeline uses the flat-downtime headline recipe (matches the production `CRRT_pressure_wo_peak.csv`,
6,102,758 rows). The Day≤14 cap is applied on the stream; the clinical Day≤7 cap at feature-build. The
drop from 5,696 circuit-days (signal) to 4,585 (cohort) is the Day≤7 cap plus the inner-join to circuit-days
with a valid mortality label. Source: `signal_pipeline_stats.py`.*
