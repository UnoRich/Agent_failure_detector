import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "manufacturing_legacy.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

PIPELINE_PATH = os.path.join(MODELS_DIR, "yield_prediction_pipeline.joblib")

def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file not found at {PIPELINE_PATH}. Run train.py first.")
    print(f"Loading pipeline from {PIPELINE_PATH}...")
    return joblib.load(PIPELINE_PATH)

def load_data_from_db():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df_mes = pd.read_sql_query("SELECT * FROM mes_lot_wafers", conn)
    df_fdc = pd.read_sql_query("SELECT * FROM fdc_sensor_data", conn)
    conn.close()
    
    df = pd.merge(df_mes, df_fdc, on="sample_id")
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def main():
    # 1. Load pipeline and data
    pipeline = load_pipeline()
    model = pipeline["model"]
    model_name = pipeline["model_name"]
    kept_features = pipeline["kept_features"]
    imputer = pipeline["imputer"]
    scaler = pipeline["scaler"]
    threshold = pipeline["threshold"]
    
    df = load_data_from_db()
    total_samples = len(df)
    split_idx = int(total_samples * 0.8)
    
    # Slice the test set
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    
    # 2. Extract features and preprocess
    X_test_raw = df_test[kept_features]
    X_test_imputed = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 3. Get predictions and filter predicted fails
    print(f"Applying model ({model_name}) with decision threshold: {threshold:.3f}...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    df_test['pred_prob'] = y_prob
    df_test['pred_fail'] = (y_prob >= threshold).astype(int)
    
    failed_wafers = df_test[df_test['pred_fail'] == 1]
    print(f"Total test samples: {len(df_test)}")
    print(f"Wafers predicted to fail (alarm triggered): {len(failed_wafers)}")
    
    if len(failed_wafers) == 0:
        print("No failures predicted in the test set. Lower threshold or retrain model.")
        return

    # 4. Initialize SHAP Explainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values for the test set...")
    shap_values = explainer.shap_values(X_test_scaled)
    
    # In some versions of SHAP and depending on binary/multiclass classification, 
    # shap_values can be a list [shap_values_class0, shap_values_class1] or a single array.
    # XGBoost/LightGBM binary classification usually returns a single array or 2D array.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1] # Use positive class contribution
        
    print(f"SHAP values computed. Shape: {shap_values.shape}")
    
    # 5. Global Feature Importance (Top 15 features across test set)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    global_importance = pd.DataFrame({
        'feature': kept_features,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Plot Global Feature Importance
    plt.figure(figsize=(10, 6))
    top_features = global_importance.head(15)
    sns_plot = plt.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='steelblue')
    plt.title('Global Feature Importance (Top 15 Sensors driving Yield Failures)')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.tight_layout()
    global_plot_path = os.path.join(REPORTS_DIR, "global_shap_importance.png")
    plt.savefig(global_plot_path)
    plt.close()
    print(f"Saved global SHAP importance plot to {global_plot_path}")

    # 6. Local Root Cause Analysis for predicted fails
    root_cause_records = []
    
    # Find indices of predicted failed wafers in df_test
    failed_indices = failed_wafers.index.tolist()
    
    print("\nExtracting local root causes for flagged wafers...")
    for idx in failed_indices:
        wafer_info = df_test.loc[idx]
        sample_id = int(wafer_info['sample_id'])
        lot_id = wafer_info['lot_id']
        wafer_num = int(wafer_info['wafer_id'])
        prob = float(wafer_info['pred_prob'])
        operator = wafer_info['operator_id']
        recipe = wafer_info['recipe']
        actual_status = int(wafer_info['yield_status'])
        
        # Get SHAP values for this sample
        sample_shap = shap_values[idx]
        
        # Zip features with their SHAP values and raw values
        sample_features_analysis = []
        for i, feat in enumerate(kept_features):
            raw_val = X_test_raw.loc[idx, feat]
            # Replace NaN in raw values with "Missing" or median for reporting
            if pd.isna(raw_val):
                raw_val_str = "NaN (Missing)"
            else:
                raw_val_str = f"{raw_val:.4f}"
                
            sample_features_analysis.append({
                "feature": feat,
                "shap_val": float(sample_shap[i]),
                "raw_value": raw_val_str
            })
            
        # Sort by SHAP value descending (top contributors to failure prediction)
        sample_features_analysis = sorted(sample_features_analysis, key=lambda x: x["shap_val"], reverse=True)
        top_5_contributors = sample_features_analysis[:5]
        
        root_cause_records.append({
            "sample_id": sample_id,
            "lot_id": lot_id,
            "wafer_id": wafer_num,
            "prediction_probability": prob,
            "actual_status": "FAIL" if actual_status == 1 else "PASS",
            "operator_id": operator,
            "recipe": recipe,
            "top_root_cause_sensors": top_5_contributors
        })
        
    # Save root cause records to JSON
    json_path = os.path.join(REPORTS_DIR, "root_cause_data.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(root_cause_records, jf, indent=4, ensure_ascii=False)
    print(f"Saved local root cause analysis data to {json_path}")
    
    # 7. Write Markdown Report
    report_path = os.path.join(REPORTS_DIR, "root_cause_analysis.md")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Root Cause Analysis (XAI / SHAP) Report\n\n")
        rf.write("## 1. Global Feature Analysis\n")
        rf.write("This section shows the top 10 sensors that have the strongest global impact on driving wafers to fail.\n\n")
        rf.write("| Rank | Sensor (Feature) | Mean Abs SHAP Value |\n")
        rf.write("| --- | --- | --- |\n")
        for i in range(10):
            row = global_importance.iloc[i]
            rf.write(f"| {i+1} | {row['feature']} | {row['importance']:.6f} |\n")
        rf.write("\n")
        rf.write(f"![Global SHAP Importance Graph](file://{global_plot_path})\n\n")
        
        rf.write("## 2. Local Root Cause Analysis for Flagged Wafers\n")
        rf.write(f"The model analyzed {len(df_test)} test wafers and flagged **{len(failed_wafers)}** anomalies (Yield Failure Risk >= {threshold:.3f}).\n")
        rf.write("Below are details for the top 5 failed wafers showing their candidate root cause sensors:\n\n")
        
        # Display top 5 examples in report
        for record in root_cause_records[:5]:
            rf.write(f"### Lot: {record['lot_id']} | Wafer: {record['wafer_id']}\n")
            rf.write(f"- **Yield Fail Probability:** {record['prediction_probability']*100:.2f}%\n")
            rf.write(f"- **Actual Yield Status:** {record['actual_status']}\n")
            rf.write(f"- **Operator & Recipe:** {record['operator_id']} | {record['recipe']}\n")
            rf.write("- **Top 5 Root Cause Candidates (SHAP Contribution):**\n\n")
            
            rf.write("| Rank | Sensor | Raw Value | SHAP Contribution (Log-Odds Impact) |\n")
            rf.write("| --- | --- | --- | --- |\n")
            for r, sensor in enumerate(record['top_root_cause_sensors']):
                rf.write(f"| {r+1} | {sensor['feature']} | {sensor['raw_value']} | {sensor['shap_val']:.4f} |\n")
            rf.write("\n---\n\n")
            
    print(f"Saved markdown root cause analysis report to {report_path}")
    print("Explainability step completed successfully.")

if __name__ == "__main__":
    main()
