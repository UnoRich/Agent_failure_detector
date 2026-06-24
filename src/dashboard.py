import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import run_agent

# Page configuration
st.set_page_config(
    page_title="Semiconductor Yield Excursion Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "manufacturing_legacy.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

PIPELINE_PATH = os.path.join(MODELS_DIR, "yield_prediction_pipeline.joblib")
ROOT_CAUSE_JSON_PATH = os.path.join(REPORTS_DIR, "root_cause_data.json")

# Custom Styling (Dark Mode / Sleek Manufacturing Theme)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }
    .metric-card h3 {
        color: #e5e7eb !important;
        font-size: 15px;
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .metric-card h2 {
        color: #ffffff !important;
        font-size: 32px;
        margin: 0;
        font-weight: 700;
    }
    .alert-header {
        background-color: #991b1b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_data_from_db():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    df_mes = pd.read_sql_query("SELECT * FROM mes_lot_wafers", conn)
    df_fdc = pd.read_sql_query("SELECT * FROM fdc_sensor_data", conn)
    conn.close()
    
    df = pd.merge(df_mes, df_fdc, on="sample_id")
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

@st.cache_data
def load_root_causes():
    if not os.path.exists(ROOT_CAUSE_JSON_PATH):
        return []
    with open(ROOT_CAUSE_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    st.title("🏭 Semiconductor Yield Excursion Copilot")
    st.subheader("On-Premise AI-assisted Manufacturing Yield Monitoring & Diagnostics")
    
    # Load assets
    pipeline = load_pipeline()
    df_all = load_data_from_db()
    root_causes = load_root_causes()
    
    if pipeline is None or df_all is None:
        st.error("Error: Project database or trained model pipeline missing. Please complete the modeling step first.")
        return

    model = pipeline["model"]
    kept_features = pipeline["kept_features"]
    imputer = pipeline["imputer"]
    scaler = pipeline["scaler"]
    threshold = pipeline["threshold"]

    # Slicing the test set to display production data
    total_samples = len(df_all)
    split_idx = int(total_samples * 0.8)
    df_test = df_all.iloc[split_idx:].reset_index(drop=True)

    # Preprocess and score on-the-fly to ensure consistency
    X_test_raw = df_test[kept_features]
    X_test_imputed = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_imputed)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    df_test['pred_prob'] = y_prob
    df_test['pred_fail'] = (y_prob >= threshold).astype(int)

    # 1. Sidebar - Navigation & Filters
    st.sidebar.header("🔍 Fab Monitoring Filters")
    
    # Recipes & Operators Filter
    selected_recipe = st.sidebar.selectbox("Filter by Process Recipe", ["All"] + list(df_test['recipe'].unique()))
    selected_operator = st.sidebar.selectbox("Filter by Operator", ["All"] + list(df_test['operator_id'].unique()))
    
    df_filtered = df_test.copy()
    if selected_recipe != "All":
        df_filtered = df_filtered[df_filtered['recipe'] == selected_recipe]
    if selected_operator != "All":
        df_filtered = df_filtered[df_filtered['operator_id'] == selected_operator]

    # 2. Main KPIs
    total_wafers = len(df_filtered)
    flagged_fails = sum(df_filtered['pred_fail'] == 1)
    actual_fails = sum(df_filtered['yield_status'] == 1)
    pass_wafers = total_wafers - flagged_fails
    predicted_yield = (pass_wafers / total_wafers) * 100 if total_wafers > 0 else 100.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='metric-card'><h3>Total Wafers</h3><h2>{total_wafers}</h2></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='metric-card'><h3>Yield Failure Alarms</h3><h2 style='color:#f87171;'>{flagged_fails}</h2></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-card'><h3>Actual Failures (GT)</h3><h2>{actual_fails}</h2></div>", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"<div class='metric-card'><h3>Predicted Yield Rate</h3><h2>{predicted_yield:.2f}%</h2></div>", unsafe_allow_html=True)

    # Layout columns
    col_left, col_right = st.columns([1, 1])

    # Left Column: Lot Yield Status
    with col_left:
        st.subheader("📦 Lot Yield Monitoring Status")
        
        # Group by Lot
        lot_grouped = df_filtered.groupby('lot_id').agg(
            total_wafers=('sample_id', 'count'),
            alarms=('pred_fail', 'sum'),
            max_risk=('pred_prob', 'max'),
            actual_fails=('yield_status', 'sum')
        ).reset_index()
        
        # Add risk severity status
        def assign_status(row):
            if row['alarms'] > 3:
                return "🚨 CRITICAL"
            elif row['alarms'] > 0:
                return "⚠️ WARNING"
            return "✅ NORMAL"
            
        lot_grouped['status'] = lot_grouped.apply(assign_status, axis=1)
        lot_grouped['max_risk'] = (lot_grouped['max_risk'] * 100).round(2).astype(str) + "%"
        
        st.dataframe(
            lot_grouped[['lot_id', 'total_wafers', 'alarms', 'actual_fails', 'max_risk', 'status']],
            use_container_width=True,
            column_config={
                "lot_id": "Lot ID",
                "total_wafers": "Total Wafers",
                "alarms": "Yield Alarms",
                "actual_fails": "Actual Fails",
                "max_risk": "Max Wafer Risk",
                "status": "Lot Status"
            }
        )

    # Right Column: Global Model Insights
    with col_right:
        st.subheader("📊 Global Yield Failure Factors (SHAP)")
        global_img_path = os.path.join(REPORTS_DIR, "global_shap_importance.png")
        if os.path.exists(global_img_path):
            st.image(global_img_path, caption="Top Global Sensor Features Contributing to Failures", use_container_width=True)
        else:
            st.info("Global SHAP feature importance chart not generated.")

    st.markdown("---")

    # 3. Interactive Diagnostics Section
    st.header("🔍 Wafer Level Root-Cause Diagnostics (Local XAI)")
    
    diag_col1, diag_col2 = st.columns([1, 2])
    
    with diag_col1:
        # Select Lot
        available_lots = sorted(df_filtered['lot_id'].unique())
        selected_lot = st.selectbox("Select Lot for Inspection", available_lots)
        
        # Wafers in that lot
        df_lot = df_filtered[df_filtered['lot_id'] == selected_lot].copy()
        
        st.write(f"Wafers in **{selected_lot}**:")
        
        # Format table helper
        def style_wafer_row(val):
            color = 'red' if val == 1 else 'green'
            return f'color: {color}'
            
        st.dataframe(
            df_lot[['wafer_id', 'pred_prob', 'pred_fail', 'yield_status', 'operator_id', 'recipe']],
            use_container_width=True,
            column_config={
                "wafer_id": "Wafer #",
                "pred_prob": "Fail Risk (Prob)",
                "pred_fail": "Alarm Status (1=Fail)",
                "yield_status": "Actual Status (1=Fail)",
                "operator_id": "Operator",
                "recipe": "Recipe"
            }
        )
        
        # Select specific wafer for SHAP analysis
        wafer_options = sorted(df_lot['wafer_id'].unique())
        selected_wafer = st.selectbox("Select Wafer to Diagnose", wafer_options)
        
    with diag_col2:
        # Load local details for selected wafer
        wafer_row = df_lot[df_lot['wafer_id'] == selected_wafer].iloc[0]
        sample_id_val = int(wafer_row['sample_id'])
        pred_prob_val = float(wafer_row['pred_prob'])
        pred_fail_val = int(wafer_row['pred_fail'])
        actual_val = int(wafer_row['yield_status'])
        
        st.subheader(f"Diagnosis Details: Lot {selected_lot} | Wafer {selected_wafer}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Failure Risk", f"{pred_prob_val*100:.2f}%")
        c2.metric("Alarm Status", "🚨 TRIGGERED" if pred_fail_val == 1 else "✅ NORMAL")
        c3.metric("Actual Yield Status", "FAIL" if actual_val == 1 else "PASS")
        
        # Display root cause sensors if predicted fail or if user requests
        # We find the matching entry in root_causes
        wafer_cause = next((item for item in root_causes if item["sample_id"] == sample_id_val), None)
        
        if wafer_cause:
            st.markdown("#### 🔬 Top 5 Root Cause Candidate Sensors:")
            
            top_sensors = wafer_cause["top_root_cause_sensors"]
            
            # Display sensor table
            sensors_df = pd.DataFrame(top_sensors)
            sensors_df.columns = ["Sensor", "SHAP Value (Contribution)", "Raw Value"]
            
            st.dataframe(sensors_df, use_container_width=True)
            
            # Plot local SHAP chart
            fig, ax = plt.subplots(figsize=(8, 3))
            names = [s["feature"] for s in top_sensors][::-1]
            values = [s["shap_val"] for s in top_sensors][::-1]
            
            # Color code contribution
            colors = ['#f87171' if v > 0 else '#60a5fa' for v in values]
            ax.barh(names, values, color=colors)
            ax.set_title("Local Sensor Contribution to Fail Risk (SHAP)")
            ax.set_xlabel("SHAP Value (Log-Odds)")
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
        else:
            if pred_fail_val == 0:
                st.success("Wafer is predicted normal. Sensor values are within control limits.")
            else:
                st.warning("No SHAP values found for this wafer. Re-run explanation analysis script.")

    st.markdown("---")
    
    # 4. Preview of Local Agent
    st.header("🤖 Secure Copilot Query Preview (Local LLM)")
    st.markdown("> *Below is a simulation of the Local LLM Agentic interaction. The agent uses Ollama and the local database to generate explanations.*")
    
    user_query = st.text_input(
        "Ask the Local AI Agent a question:",
        value=f"Why was Wafer {selected_wafer} in Lot {selected_lot} flagged for failure?"
    )
    
    if st.button("Query Local Copilot Agent"):
        with st.spinner("Local LLM (Ollama / Llama 3) generating secure analysis..."):
            agent_response = run_agent(user_query)
            st.markdown(agent_response)

if __name__ == "__main__":
    main()
