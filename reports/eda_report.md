# SECOM Dataset EDA Report

## 1. Dataset Dimensions
- Total wafers (samples): 1567
- Total sensor readings (features): 590

## 2. Yield Class Distribution (Imbalance)
- **Pass (0):** 1463 samples (93.36%)
- **Fail (1):** 104 samples (6.64%)
- **Imbalance Ratio:** 14.07:1

> [!IMPORTANT]
> Extreme class imbalance detected. Precision-Recall AUC, F1-Score, and Recall should be used as evaluation metrics instead of Accuracy. Resampling (SMOTE) or algorithmic weighting will be required during modeling.

## 3. Missing Value Diagnostics
- Total features with missing values: 538 / 590 (91.2%)
- Features with > 10% missing: 52
- Features with > 50% missing: 28

### Top 10 Features with Most Missing Values
| Feature | Missing Count | Missing Percentage (%)
| --- | --- | ---
| sensor_157 | 1429 | 91.19%
| sensor_292 | 1429 | 91.19%
| sensor_293 | 1429 | 91.19%
| sensor_158 | 1429 | 91.19%
| sensor_492 | 1341 | 85.58%
| sensor_358 | 1341 | 85.58%
| sensor_85 | 1341 | 85.58%
| sensor_220 | 1341 | 85.58%
| sensor_246 | 1018 | 64.96%
| sensor_109 | 1018 | 64.96%

## 4. Zero Variance Features (Constants)
- **Count:** 116 constant features identified.
- Constant features are non-informative and should be dropped during preprocessing.
- List of constant features: `sensor_5, sensor_13, sensor_42, sensor_49, sensor_52, sensor_69, sensor_97, sensor_141, sensor_149, sensor_178, sensor_179, sensor_186, sensor_189, sensor_190, sensor_191, sensor_192, sensor_193, sensor_194, sensor_226, sensor_229, sensor_230, sensor_231, sensor_232, sensor_233, sensor_234, sensor_235, sensor_236, sensor_237, sensor_240, sensor_241...`

## 5. Timeline Analysis
- **Data Collection Period:** 2008-07-19 11:55:00 to 2008-10-17 06:07:00
