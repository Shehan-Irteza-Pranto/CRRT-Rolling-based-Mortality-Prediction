# Patient-clustered bootstrap (test: 887 days, 196 patients, 62 deaths; B=2000, 2000 valid draws, seed 0)

## 95% percentile CIs

| model | metric | point | 95% CI | half-width |
|---|---|--:|--:|--:|
| Full stack (Transformer fusion + top-3 combined classical) | AUROC | 0.766 | [0.705, 0.825] | ±0.060 |
| Full stack (Transformer fusion + top-3 combined classical) | AUPRC | 0.216 | [0.152, 0.325] | ±0.087 |
| EHR-only stack (clinical MLP-60 + top-3 clinical classical) | AUROC | 0.738 | [0.674, 0.800] | ±0.063 |
| EHR-only stack (clinical MLP-60 + top-3 clinical classical) | AUPRC | 0.225 | [0.152, 0.315] | ±0.082 |
| Classical-only combined stack (top-3, no Transformer) | AUROC | 0.758 | [0.695, 0.819] | ±0.062 |
| Classical-only combined stack (top-3, no Transformer) | AUPRC | 0.208 | [0.147, 0.312] | ±0.083 |
| Clinical MLP-60 (Table III clinical-only) | AUROC | 0.710 | [0.642, 0.776] | ±0.067 |
| Clinical MLP-60 (Table III clinical-only) | AUPRC | 0.177 | [0.123, 0.270] | ±0.073 |
| Machine-only Windowed-Transformer Ensemble (mRMR-40) | AUROC | 0.625 | [0.559, 0.694] | ±0.068 |
| Machine-only Windowed-Transformer Ensemble (mRMR-40) | AUPRC | 0.109 | [0.078, 0.171] | ±0.046 |
| Machine-only CNN1D | AUROC | 0.579 | [0.505, 0.657] | ±0.076 |
| Machine-only CNN1D | AUPRC | 0.095 | [0.069, 0.140] | ±0.035 |
| Machine-only RNN (BiLSTM/GRU) | AUROC | 0.584 | [0.508, 0.660] | ±0.076 |
| Machine-only RNN (BiLSTM/GRU) | AUPRC | 0.094 | [0.070, 0.134] | ±0.032 |
| Machine-only Patch-Transformer | AUROC | 0.581 | [0.514, 0.647] | ±0.066 |
| Machine-only Patch-Transformer | AUPRC | 0.086 | [0.065, 0.121] | ±0.028 |
| Machine-only Attention-CNN1D | AUROC | 0.578 | [0.501, 0.650] | ±0.075 |
| Machine-only Attention-CNN1D | AUPRC | 0.103 | [0.074, 0.155] | ±0.040 |
| Combined CNN1D fusion | AUROC | 0.646 | [0.566, 0.722] | ±0.078 |
| Combined CNN1D fusion | AUPRC | 0.182 | [0.117, 0.273] | ±0.078 |
| Combined RNN (BiLSTM/GRU) fusion | AUROC | 0.669 | [0.593, 0.741] | ±0.074 |
| Combined RNN (BiLSTM/GRU) fusion | AUPRC | 0.222 | [0.148, 0.312] | ±0.082 |
| Combined Patch-Transformer fusion | AUROC | 0.680 | [0.608, 0.754] | ±0.073 |
| Combined Patch-Transformer fusion | AUPRC | 0.182 | [0.122, 0.268] | ±0.073 |
| Combined Attention-CNN1D fusion | AUROC | 0.682 | [0.603, 0.756] | ±0.077 |
| Combined Attention-CNN1D fusion | AUPRC | 0.156 | [0.109, 0.250] | ±0.071 |

## Paired differences (identical patient draws)

| comparison | metric | Δ | 95% CI | one-sided bootstrap p (Δ≤0) |
|---|---|--:|--:|--:|
| Full stack (Transformer fusion + top-3 combined classical)  −  EHR-only stack (clinical MLP-60 + top-3 clinical classical) | AUROC | +0.028 | [-0.012, +0.072] | 0.0845 |
| Full stack (Transformer fusion + top-3 combined classical)  −  EHR-only stack (clinical MLP-60 + top-3 clinical classical) | AUPRC | -0.009 | [-0.040, +0.044] | 0.5175 |
| Full stack (Transformer fusion + top-3 combined classical)  −  Classical-only combined stack (top-3, no Transformer) | AUROC | +0.008 | [-0.009, +0.024] | 0.1910 |
| Full stack (Transformer fusion + top-3 combined classical)  −  Classical-only combined stack (top-3, no Transformer) | AUPRC | +0.008 | [-0.004, +0.030] | 0.0985 |
| Full stack (Transformer fusion + top-3 combined classical)  −  Clinical MLP-60 (Table III clinical-only) | AUROC | +0.056 | [+0.008, +0.111] | 0.0110 |
| Full stack (Transformer fusion + top-3 combined classical)  −  Clinical MLP-60 (Table III clinical-only) | AUPRC | +0.039 | [-0.002, +0.093] | 0.0295 |
| EHR-only stack (clinical MLP-60 + top-3 clinical classical)  −  Clinical MLP-60 (Table III clinical-only) | AUROC | +0.028 | [+0.001, +0.064] | 0.0210 |
| EHR-only stack (clinical MLP-60 + top-3 clinical classical)  −  Clinical MLP-60 (Table III clinical-only) | AUPRC | +0.048 | [+0.008, +0.076] | 0.0095 |
| Machine-only Windowed-Transformer Ensemble (mRMR-40)  −  0.5 (chance) | AUROC | +0.125 | [+0.059, +0.194] | 0.0000 |

## Table III rows with patient-clustered 95% CIs

| row | AUROC | 95% CI | AUPRC | 95% CI |
|---|--:|--:|--:|--:|
| Clinical-only: Clinical MLP | 0.710 | [0.642, 0.776] | 0.177 | [0.123, 0.270] |
| Clinical-only: Clinical Fusion-free Stack (ours) | 0.738 | [0.674, 0.800] | 0.225 | [0.152, 0.315] |
| Machine-only: CNN1D | 0.579 | [0.505, 0.657] | 0.095 | [0.069, 0.140] |
| Machine-only: RNN (BiLSTM/GRU) | 0.584 | [0.508, 0.660] | 0.094 | [0.070, 0.134] |
| Machine-only: Patch-Transformer | 0.581 | [0.514, 0.647] | 0.086 | [0.065, 0.121] |
| Machine-only: Attention-CNN1D | 0.578 | [0.501, 0.650] | 0.103 | [0.074, 0.155] |
| Machine-only: Windowed-Transformer Ensemble (ours) | 0.625 | [0.559, 0.694] | 0.109 | [0.078, 0.171] |
| Combined: CNN1D fusion | 0.646 | [0.566, 0.722] | 0.182 | [0.117, 0.273] |
| Combined: RNN (BiLSTM/GRU) fusion | 0.669 | [0.593, 0.741] | 0.222 | [0.148, 0.312] |
| Combined: Patch-Transformer fusion | 0.680 | [0.608, 0.754] | 0.182 | [0.122, 0.268] |
| Combined: Attention-CNN1D fusion | 0.682 | [0.603, 0.756] | 0.156 | [0.109, 0.250] |
| Combined: Windowed-Transformer Fusion Stack (ours) | 0.766 | [0.705, 0.825] | 0.216 | [0.152, 0.325] |
