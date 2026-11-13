# Data processing

This document lists the input files the pipeline expects, their column schemas, and what each processing stage does
to them. No data ship with the repository; the `Data/` tree is an empty skeleton with the expected folder names.

All paths are relative to the repository root. Every script locates `Data/` from its own file location
(`<repo>/Baseline_ML_Models_v2/...` goes two or three levels up), so keep the layout and do not move folders.

## 1. Input files (you provide these)

### 1.1 Machine stream (`Data/Intermediate/`)

| file | one row per | required columns |
|---|---|---|
| `extracted_raw.csv` | machine minute | `PID, Time, RUN_TIME, POST_INFUSION, PRE_INFUSION, DIALYSATE, EFFLUENT, PREBLOOD_INF, SYRINGE_INF, EXCESS_PT_FLUID, PUMP1..PUMP4, ACCESS_PRESSURE, FILTER_PRESSURE, EFFLUENT_PRESSURE, RETURN_PRESSURE` |
| `Events_date_time.csv` | patient | `PID, HospitalAdmitDate, IcuAdmitDate, CrrtStartDate, CrrtIndependence, IcuLastDate, HospitalDischargeDate, IcuDeath, IcuDeathDate, Death_Date, HospitalDeath, CRRTStartTime_Machine, CRRTEndTime_Machine, Mortality` |
| `mortality_label.csv` | patient | `PID, Mortality, Death_Date` |

`PID` is a study identifier (integer), `Time` is a timestamp at one-minute resolution, pressures are in mmHg as
exported by the device, and the fluid columns are the device's cumulative counters (mL). `extracted_raw.csv` is the
device export after vendor-specific parsing; that parser is not part of this release.

### 1.2 EHR tables (`Data/EHR/`)

| file | one row per | required columns |
|---|---|---|
| `Lab/lab_transformed_longitudinal.csv` | lab panel | `PID, FormDataId, DailyLaboratoryDataDate, TimeOfDay24Hour`, then one column per analyte (`WhiteCellCountUs, PhosphateUs, HemoglobinUs, MagnesiumUs, HematocritUs, CalciumTotalUs, PlateletsUs, CalciumIonizedUs, INRUs, AlbuminUs, PTTUs, BilirubinUs, SodiumUs, pHUs, PotassiumUs, FiO2Us, BicarbonateUs, PaO2Us, ChlorideUs, PaCO2Us, UreaUs, CreatinineUs, LactateUs, SaO2Us`) |
| `Demographics/pti_transformed_cleaned.csv` | patient | `PID, FormDataId, Age, Gender_Male, HospitalWeight_kg, IcuWeight_kg, CrrtInitiationWeight_kg, HospitalAdmitDate, IcuAdmitDate, CrrtStartDate, IcuHeight_cm, IcuBmi_kg_m2, CrrtStartTime_24HourFormat, CrrtStartTime_FractionOfDay`, the Charlson comorbidity flags (`DiabetesMellitus ... MetastaticTumor`), `CharlsonCoMorbidityScore, AdmissionSourceInsideHospital_presence` |

Analyte columns are numeric in the units used at the source; the `Us` suffix is a naming convention and is kept by the
feature code.

## 2. Processing stages

### Stage 0a. Signal alignment and cleaning
`Code/Denoising/CRRT_stream_denoising.ipynb` (set `parent_dir` to `../../Data`, the default in this release)

1. **Temporal alignment.** Machine minutes are kept only inside the charted CRRT window from `Events_date_time.csv`,
   widened by a 12-hour buffer on each side (Eq. 1 of the paper). Minutes outside any charted window are dropped.
2. **Episode and day indexing.** A gap of more than 24 h between machine minutes starts a new circuit episode
   (`Visit_no`); `Day` is the rank of the calendar date within the episode. Only the first 14 treatment days are kept.
3. **Priming removal.** Minutes before blood flow is established (prime / self-test phase, Eq. 2) are removed.
4. **Downtime removal.** Minutes with the machine stopped or recirculating (run-time counter flat, no effluent
   production, or low delivered dose; Eq. 3) are removed. `CRRT_stream_denoising_abalation.ipynb` produces the
   Th70 / Th80 / Th90 downtime-threshold variants used in the ablation.
5. **Peak detection and smoothing.** Within a rolling 40-minute window a minute whose pressure deviates by more than
   2 standard deviations from the window is flagged as an artefact and replaced by the mean of its neighbours (Eq. 4).
   Both the smoothed and the unsmoothed streams are saved.

Outputs (`Data/Processed/Machine/`):

| file | columns |
|---|---|
| `pressure/denoised/CRRT_pressure_wo_peak.csv` | `PID, Visit_no, Day, Time, ACCESS_PRESSURE, FILTER_PRESSURE, EFFLUENT_PRESSURE, RETURN_PRESSURE` |
| `pressure/noised/CRRT_pressure_with_peak.csv` | same columns, before peak smoothing |
| `fluid/CRRT_fluid.csv` | fluid counters on the same minutes (not used by the paper's models) |

The same operations are available as importable functions for the ablations in
`Baseline_ML_Models_v2/abalation_study_v2/preprocessing/build_streams.py` and `smoothing.py`.
`paper_materials/signal_pipeline_stats.py` reproduces the attrition numbers of Fig. 2.

### Stage 0b. Rolling labels
`Code/label_processing/label_1day_2day_7_day.ipynb` (set `base_dir` to `../../Data`)

For each circuit day *t* the label is death within the target window after a gap: `label_1day` = death on day *t+2*
(input day *t*, gap day *t+1*), `label_2day` likewise with a two-day window, `label_7day` for the seven-day horizon.
Days after death are dropped.

Output: `Data/Processed/Labels/labels_{1day,2day,7day}.csv` with
`PID, Visit_no, Day, Date, Mortality, Death_Date, days_until_death, label_<horizon>`.

### Stage 0c. Clinical tables
`Baseline_ML_Models_v2/Clinical_Only_Models/prepare_clinical_only_data.ipynb` (resolves the project root as `../..`)

Laboratory values are aggregated per calendar day and forward-filled within a patient; demographics and comorbidities
are static per patient. The notebook writes `Data/Processed/Model_Ready/clinical_only/{1day,2day}/`, which
`abalation_study_v2/preprocessing/build_clinical.py` turns into the study's clinical feature table.

### Stage 1. Study inputs (`Baseline_ML_Models_v2/abalation_study_v2/`)

| script | what it builds |
|---|---|
| `preprocessing/build_splits.py` | patient-level stratified split 70 / 10 / 20 (train / validation / test) with a fixed seed; a patient's circuit days never straddle splits. Written to `data/splits/pid_splits_v2.json` (not shipped; regenerate). |
| `refresh_data.py --variants current --peaks denoised noised --cases 1day 2day` | gap-aware reconstruction on a 1440-minute daily grid (gaps up to 10 minutes interpolated, longer gaps masked), derived channels TMP = (FILTER + RETURN)/2 − EFFLUENT, PFD = FILTER − RETURN, ARD = RETURN − ACCESS, the 133 base machine features per circuit day (13 distribution + 6 temporal statistics × 7 channels) and the deep-learning sequence cache. |
| `exp_3/features/build_rich_features.py --case 1day` | 67 enriched features (spectral, permutation entropy, excursion, end-of-day, cross-channel) giving the 200-feature machine table. The feature definitions are described in `paper_materials/MACHINE_ENHANCEMENT.md`. |
| `preprocessing/build_clinical.py` | clinical feature table aligned to circuit days. |
| `models/feature_select.py` | grouped mRMR: a fixed budget per modality (paper: 40 machine + 60 clinical) selected on the training split only. |
| `models/dl_reduce.py` | sequence view for the transformer: 144 ten-minute windows × 5 statistics × 6 channels, gap windows masked. |

Generated tables and caches live under `abalation_study_v2/data/` and `exp_*/data/`; they are ignored by git.

## 3. Privacy

The study identifiers in the original cohort are pseudonymous integers; no dates of birth, names or medical record
numbers are used anywhere in the code. Notebook outputs were cleared before release. If you adapt the pipeline to
your own data, keep patient-level tables outside version control (the shipped `.gitignore` excludes `Data/`, all
generated `data/`, `results/` and model folders).
