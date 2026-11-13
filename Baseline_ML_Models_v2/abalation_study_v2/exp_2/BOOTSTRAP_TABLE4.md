# Table IV with patient-clustered bootstrap 95% CIs (test: 887 days, 196 patients, 62 deaths; B=2000, 2000 valid draws, seed 0)

| pipeline | model | AUROC | 95% CI | AUPRC | 95% CI |
|---|---|--:|--:|--:|--:|
| Clinical-only | LogReg | 0.710 | [0.641, 0.780] | 0.222 | [0.149, 0.322] |
| Clinical-only | DecisionTree | 0.617 | [0.545, 0.687] | 0.117 | [0.081, 0.176] |
| Clinical-only | RandomForest | 0.711 | [0.641, 0.778] | 0.233 | [0.153, 0.325] |
| Clinical-only | GradBoost | 0.717 | [0.649, 0.784] | 0.209 | [0.139, 0.308] |
| Clinical-only | AdaBoost | 0.702 | [0.632, 0.771] | 0.186 | [0.128, 0.276] |
| Clinical-only | KNN | 0.590 | [0.513, 0.672] | 0.151 | [0.095, 0.244] |
| Clinical-only | LightGBM | 0.716 | [0.644, 0.788] | 0.209 | [0.146, 0.316] |
| Clinical-only | XGBoost | 0.685 | [0.610, 0.757] | 0.217 | [0.143, 0.318] |
| Machine-only | LogReg | 0.589 | [0.508, 0.666] | 0.109 | [0.075, 0.176] |
| Machine-only | DecisionTree | 0.552 | [0.489, 0.614] | 0.085 | [0.063, 0.128] |
| Machine-only | RandomForest | 0.584 | [0.511, 0.656] | 0.109 | [0.071, 0.179] |
| Machine-only | GradBoost | 0.598 | [0.527, 0.672] | 0.101 | [0.074, 0.156] |
| Machine-only | AdaBoost | 0.608 | [0.535, 0.682] | 0.095 | [0.069, 0.137] |
| Machine-only | KNN | 0.559 | [0.485, 0.635] | 0.089 | [0.063, 0.133] |
| Machine-only | LightGBM | 0.603 | [0.534, 0.674] | 0.107 | [0.074, 0.172] |
| Machine-only | XGBoost | 0.587 | [0.511, 0.669] | 0.105 | [0.075, 0.167] |
| Combined | LogReg | 0.725 | [0.656, 0.792] | 0.218 | [0.148, 0.324] |
| Combined | DecisionTree | 0.535 | [0.448, 0.617] | 0.096 | [0.068, 0.156] |
| Combined | RandomForest | 0.728 | [0.665, 0.786] | 0.188 | [0.128, 0.273] |
| Combined | GradBoost | 0.718 | [0.653, 0.783] | 0.173 | [0.123, 0.260] |
| Combined | AdaBoost | 0.698 | [0.624, 0.772] | 0.159 | [0.112, 0.235] |
| Combined | KNN | 0.568 | [0.500, 0.642] | 0.102 | [0.071, 0.163] |
| Combined | LightGBM | 0.694 | [0.629, 0.764] | 0.179 | [0.124, 0.271] |
| Combined | XGBoost | 0.720 | [0.653, 0.790] | 0.182 | [0.130, 0.283] |
