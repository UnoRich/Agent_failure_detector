# Model Training & Evaluation Report

## 1. Split Strategy
- **Method:** Chronological Time-Series Split (80% Train, 20% Test)
- **Train Samples:** 1253 (Failures: 87)
- **Test Samples:** 314 (Failures: 17)

## 2. Preprocessing & Feature Engineering
- **Total Raw Features:** 590
- **Features Dropped (Zero Variance):** 116
- **Features Dropped (>50% Missing):** 28
- **Final Features Count:** 446
- **Imputation Strategy:** Median value attribution of training set
- **Scaling Strategy:** StandardScaler

## 3. Model Performance Comparison
| Model | Precision | Recall | F1-Score | PR-AUC | Optimal Threshold |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 0.1000 | 0.0588 | 0.0741 | 0.0493 | 0.070 |
| XGBoost | 0.0759 | 0.3529 | 0.1250 | 0.0595 | 0.050 |

### Selected Model: **XGBoost**
- **Optimal Decision Threshold:** 0.050
- **Criteria:** Maximizing Precision-Recall Area Under Curve (PR-AUC) under extreme class imbalance.
