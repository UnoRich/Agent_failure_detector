# Root Cause Analysis (XAI / SHAP) Report

## 1. Global Feature Analysis
This section shows the top 10 sensors that have the strongest global impact on driving wafers to fail.

| Rank | Sensor (Feature) | Mean Abs SHAP Value |
| --- | --- | --- |
| 1 | sensor_59 | 0.788098 |
| 2 | sensor_460 | 0.206072 |
| 3 | sensor_307 | 0.158013 |
| 4 | sensor_573 | 0.156364 |
| 5 | sensor_33 | 0.150516 |
| 6 | sensor_333 | 0.125050 |
| 7 | sensor_11 | 0.111750 |
| 8 | sensor_65 | 0.100233 |
| 9 | sensor_0 | 0.097461 |
| 10 | sensor_121 | 0.095546 |

![Global SHAP Importance Graph](file:///Users/wonya/AI/Hynix/reports/global_shap_importance.png)

## 2. Local Root Cause Analysis for Flagged Wafers
The model analyzed 314 test wafers and flagged **79** anomalies (Yield Failure Risk >= 0.050).
Below are details for the top 5 failed wafers showing their candidate root cause sensors:

### Lot: LOT_0051 | Wafer: 6
- **Yield Fail Probability:** 5.35%
- **Actual Yield Status:** PASS
- **Operator & Recipe:** OP_E | RECIPE_DRAM_DIFF_1z
- **Top 5 Root Cause Candidates (SHAP Contribution):**

| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |
| --- | --- | --- | --- |
| 1 | sensor_99 | 0.0828 | 0.2635 |
| 2 | sensor_333 | 7.2632 | 0.1991 |
| 3 | sensor_9 | -0.0364 | 0.1688 |
| 4 | sensor_460 | 47.8616 | 0.1262 |
| 5 | sensor_40 | 82.4000 | 0.1129 |

---

### Lot: LOT_0051 | Wafer: 7
- **Yield Fail Probability:** 21.44%
- **Actual Yield Status:** PASS
- **Operator & Recipe:** OP_D | RECIPE_HBM_BOND_3D
- **Top 5 Root Cause Candidates (SHAP Contribution):**

| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |
| --- | --- | --- | --- |
| 1 | sensor_573 | 0.0956 | 1.4058 |
| 2 | sensor_0 | 2914.0400 | 0.1730 |
| 3 | sensor_307 | 0.1433 | 0.1179 |
| 4 | sensor_71 | 183.2095 | 0.0842 |
| 5 | sensor_201 | 7.7300 | 0.0679 |

---

### Lot: LOT_0051 | Wafer: 11
- **Yield Fail Probability:** 5.99%
- **Actual Yield Status:** PASS
- **Operator & Recipe:** OP_E | RECIPE_NAND_ETCH_20nm
- **Top 5 Root Cause Candidates (SHAP Contribution):**

| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |
| --- | --- | --- | --- |
| 1 | sensor_0 | 2953.6300 | 0.1477 |
| 2 | sensor_423 | 108.6813 | 0.1334 |
| 3 | sensor_491 | 4.4743 | 0.1129 |
| 4 | sensor_460 | 32.6879 | 0.1103 |
| 5 | sensor_21 | -5242.7500 | 0.0930 |

---

### Lot: LOT_0051 | Wafer: 13
- **Yield Fail Probability:** 6.58%
- **Actual Yield Status:** PASS
- **Operator & Recipe:** OP_B | RECIPE_DRAM_DIFF_1z
- **Top 5 Root Cause Candidates (SHAP Contribution):**

| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |
| --- | --- | --- | --- |
| 1 | sensor_573 | 0.1474 | 0.3380 |
| 2 | sensor_425 | 8.6787 | 0.1809 |
| 3 | sensor_460 | 29.3356 | 0.1519 |
| 4 | sensor_21 | -5294.5000 | 0.1255 |
| 5 | sensor_90 | 8612.4800 | 0.0928 |

---

### Lot: LOT_0051 | Wafer: 18
- **Yield Fail Probability:** 14.27%
- **Actual Yield Status:** PASS
- **Operator & Recipe:** OP_E | RECIPE_HBM_BOND_3D
- **Top 5 Root Cause Candidates (SHAP Contribution):**

| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |
| --- | --- | --- | --- |
| 1 | sensor_65 | 28.1684 | 0.4505 |
| 2 | sensor_173 | 0.8643 | 0.2483 |
| 3 | sensor_460 | 33.7616 | 0.1386 |
| 4 | sensor_102 | -0.0331 | 0.1202 |
| 5 | sensor_584 | 0.0066 | 0.1100 |

---

