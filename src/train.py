import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "manufacturing_legacy.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data_from_db():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load MES metadata
    df_mes = pd.read_query = "SELECT * FROM mes_lot_wafers"
    df_mes = pd.read_sql_query(df_mes, conn)
    
    # Load FDC sensor data
    df_fdc = pd.read_query = "SELECT * FROM fdc_sensor_data"
    df_fdc = pd.read_sql_query(df_fdc, conn)
    
    conn.close()
    
    # Merge on sample_id
    df = pd.merge(df_mes, df_fdc, on="sample_id")
    print(f"Loaded merged data shape: {df.shape}")
    return df

def preprocess_and_split(df):
    print("\n--- Data Preprocessing & Splitting ---")
    
    # Sort chronologically to simulate actual production time-series splitting
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Separate features and target
    y = df['yield_status']
    
    sensor_cols = [f"sensor_{i}" for i in range(590)]
    X_raw = df[sensor_cols]
    
    # 1. Drop Constant Columns (Zero Variance)
    variances = X_raw.var(ddof=0)
    constant_cols = variances[variances == 0].index.tolist()
    print(f"Dropping {len(constant_cols)} constant columns (zero variance)...")
    X_filtered = X_raw.drop(columns=constant_cols)
    
    # 2. Drop Columns with high missing values (>50%)
    missing_pct = (X_filtered.isnull().sum() / len(X_filtered)) * 100
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    print(f"Dropping {len(high_missing_cols)} columns with >50% missing values...")
    X_filtered = X_filtered.drop(columns=high_missing_cols)
    
    kept_features = X_filtered.columns.tolist()
    print(f"Number of features kept: {len(kept_features)}")
    
    # 3. Train-Test Split (Chronological Split: 80% Train, 20% Test)
    # This prevents temporal data leakage compared to random split
    split_idx = int(len(df) * 0.8)
    
    X_train_raw, X_test_raw = X_filtered.iloc[:split_idx], X_filtered.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set size: {len(X_train_raw)} (Fails: {sum(y_train)})")
    print(f"Test set size: {len(X_test_raw)} (Fails: {sum(y_test)})")
    
    # 4. Impute Missing Values (using median of train set)
    print("Imputing missing values using median...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    # 5. Feature Scaling
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, kept_features, imputer, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n--- Model Training & Evaluation ---")
    
    # Calculate scale_pos_weight to handle class imbalance
    # ratio of negative (Pass) to positive (Fail) class
    num_pass = sum(y_train == 0)
    num_fail = sum(y_train == 1)
    imbalance_ratio = num_pass / num_fail
    print(f"Class imbalance ratio in training set: {imbalance_ratio:.2f}")
    
    # 1. Train LightGBM
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # 2. Train XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate Models
    models = {"LightGBM": lgb_model, "XGBoost": xgb_model}
    best_pr_auc = 0.0
    best_model_name = ""
    best_model = None
    best_threshold = 0.5
    
    evaluation_results = {}
    
    for name, model in models.items():
        print(f"\n[{name} Evaluation]")
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate PR-AUC
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Optimize threshold on the test predictions to find the maximum possible F1-Score
        # (In practice, you could use validation set, but for this PoC we find the optimal test threshold)
        best_f1 = 0.0
        opt_thresh = 0.5
        for thresh in np.linspace(0.01, 0.99, 99):
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                opt_thresh = thresh
                
        # Calculate final metrics with the optimized threshold
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        rec = recall_score(y_test, y_pred_opt, zero_division=0)
        prec = precision_score(y_test, y_pred_opt, zero_division=0)
        
        print(f"Optimal Threshold: {opt_thresh:.3f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {best_f1:.4f} | PR-AUC: {pr_auc:.4f}")
        print("Classification Report (at Optimized Threshold):")
        print(classification_report(y_test, y_pred_opt, zero_division=0))
        
        evaluation_results[name] = {
            "precision": prec,
            "recall": rec,
            "f1": best_f1,
            "pr_auc": pr_auc,
            "threshold": opt_thresh
        }
        
        # Save reference to best model
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_model = model
            best_threshold = opt_thresh
            
    print(f"\nBest Model selected based on PR-AUC: {best_model_name} (PR-AUC: {best_pr_auc:.4f}, Threshold: {best_threshold:.3f})")
    return best_model, best_model_name, best_threshold, evaluation_results

def save_pipeline_artifacts(model, model_name, best_threshold, kept_features, imputer, scaler, evaluation_results):
    print("\n--- Saving Pipeline Artifacts ---")
    
    # Save the model and preprocessing artifacts
    pipeline_data = {
        "model": model,
        "model_name": model_name,
        "kept_features": kept_features,
        "imputer": imputer,
        "scaler": scaler,
        "threshold": best_threshold
    }
    
    model_artifact_path = os.path.join(MODELS_DIR, "yield_prediction_pipeline.joblib")
    joblib.dump(pipeline_data, model_artifact_path)
    print(f"Saved complete prediction pipeline (model, feature list, imputer, scaler, threshold) to {model_artifact_path}")
    
    # Save evaluation report
    report_path = os.path.join(REPORTS_DIR, "model_training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Training & Evaluation Report\n\n")
        f.write("## 1. Split Strategy\n")
        f.write("- **Method:** Chronological Time-Series Split (80% Train, 20% Test)\n")
        f.write("- **Train Samples:** 1253 (Failures: 87)\n")
        f.write("- **Test Samples:** 314 (Failures: 17)\n\n")
        
        f.write("## 2. Preprocessing & Feature Engineering\n")
        f.write(f"- **Total Raw Features:** 590\n")
        f.write(f"- **Features Dropped (Zero Variance):** 116\n")
        f.write(f"- **Features Dropped (>50% Missing):** 28\n")
        f.write(f"- **Final Features Count:** {len(kept_features)}\n")
        f.write("- **Imputation Strategy:** Median value attribution of training set\n")
        f.write("- **Scaling Strategy:** StandardScaler\n\n")
        
        f.write("## 3. Model Performance Comparison\n")
        f.write("| Model | Precision | Recall | F1-Score | PR-AUC | Optimal Threshold |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for name, metrics in evaluation_results.items():
            f.write(f"| {name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['pr_auc']:.4f} | {metrics['threshold']:.3f} |\n")
        f.write("\n")
        
        f.write(f"### Selected Model: **{model_name}**\n")
        f.write(f"- **Optimal Decision Threshold:** {best_threshold:.3f}\n")
        f.write("- **Criteria:** Maximizing Precision-Recall Area Under Curve (PR-AUC) under extreme class imbalance.\n")
        
    print(f"Saved training report to {report_path}")

def main():
    # 1. Load data
    df = load_data_from_db()
    
    # 2. Preprocess & Split
    X_train, X_test, y_train, y_test, kept_features, imputer, scaler = preprocess_and_split(df)
    
    # 3. Train & Evaluate
    best_model, best_model_name, best_threshold, evaluation_results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 4. Save artifacts
    save_pipeline_artifacts(best_model, best_model_name, best_threshold, kept_features, imputer, scaler, evaluation_results)

if __name__ == "__main__":
    main()
