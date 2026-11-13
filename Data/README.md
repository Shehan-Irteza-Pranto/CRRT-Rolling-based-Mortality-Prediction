# Data/ (empty skeleton)

Place the input files here; nothing in this folder is tracked by git except this file and the `.gitkeep` markers.

```
Data/
├── Intermediate/
│   ├── extracted_raw.csv          per-minute machine export (PID, Time, fluid counters, 4 pressures)
│   ├── Events_date_time.csv       per-patient CRRT / ICU / hospital dates and outcome
│   └── mortality_label.csv        PID, Mortality, Death_Date
├── EHR/
│   ├── Lab/lab_transformed_longitudinal.csv
│   └── Demographics/pti_transformed_cleaned.csv
└── Processed/                     written by the stage-0 notebooks
    ├── Machine/pressure/{denoised,noised}/
    ├── Labels/labels_{1day,2day,7day}.csv
    └── Model_Ready/clinical_only/{1day,2day}/
```

Column schemas and the processing stages are documented in `../docs/DATA_PROCESSING.md`.
