import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# URL references
SECOM_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABELS_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def main():
    secom_data_path = os.path.join(DATA_DIR, "secom.data")
    secom_labels_path = os.path.join(DATA_DIR, "secom_labels.data")

    # Step 1: Download Data
    if not os.path.exists(secom_data_path):
        download_file(SECOM_DATA_URL, secom_data_path)
    else:
        print("secom.data already exists.")

    if not os.path.exists(secom_labels_path):
        download_file(SECOM_LABELS_URL, secom_labels_path)
    else:
        print("secom_labels.data already exists.")

    # Step 2: Load Data
    print("Loading data into Pandas DataFrames...")
    # secom.data is space-separated
    df_features = pd.read_csv(secom_data_path, sep=r"\s+", header=None)
    # secom_labels.data is double-quoted or space-separated: "label" "timestamp"
    # It contains label and timestamp separated by spaces/quotes. Let's load it.
    df_labels = pd.read_csv(secom_labels_path, sep=r"\s+", header=None, names=["label", "timestamp"])

    print(f"Features shape: {df_features.shape}")
    print(f"Labels shape: {df_labels.shape}")

    # Merge features and labels
    # We rename feature columns to sensor_0, sensor_1, ...
    df_features.columns = [f"sensor_{i}" for i in range(df_features.shape[1])]
    df = pd.concat([df_features, df_labels], axis=1)
    
    # Save a merged copy as CSV for easier future use
    merged_csv_path = os.path.join(DATA_DIR, "secom_merged.csv")
    df.to_csv(merged_csv_path, index=False)
    print(f"Saved merged dataset to {merged_csv_path}")

    # Step 3: Perform EDA
    print("\n=== Exploratory Data Analysis (EDA) ===")
    
    # 3.1 Class Distribution
    # Note: In SECOM, -1 represents Pass, 1 represents Fail.
    # Let's map it to 0 (Pass) and 1 (Fail) for standard classification tasks
    df['target'] = df['label'].map({-1: 0, 1: 1})
    class_counts = df['target'].value_counts()
    pass_count = class_counts.get(0, 0)
    fail_count = class_counts.get(1, 0)
    total_count = len(df)
    fail_rate = (fail_count / total_count) * 100

    print(f"Total Samples: {total_count}")
    print(f"Pass (0): {pass_count} ({pass_count/total_count*100:.2f}%)")
    print(f"Fail (1): {fail_count} ({fail_count/total_count*100:.2f}%)")
    print(f"Imbalance Ratio (Pass/Fail): {pass_count/fail_count:.2f}")

    # Plot Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title('SECOM Yield Class Distribution (0: Pass, 1: Fail)')
    plt.ylabel('Count')
    plt.xlabel('Status')
    plt.xticks([0, 1], [f'Pass ({pass_count})', f'Fail ({fail_count})'])
    class_dist_plot_path = os.path.join(REPORTS_DIR, "class_distribution.png")
    plt.savefig(class_dist_plot_path)
    plt.close()
    print(f"Saved class distribution plot to {class_dist_plot_path}")

    # 3.2 Missing Values Analysis
    missing_counts = df_features.isnull().sum()
    missing_pct = (df_features.isnull().sum() / len(df_features)) * 100
    missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_percentage': missing_pct})
    missing_df = missing_df.sort_values(by='missing_percentage', ascending=False)

    print("\n--- Missing Values Summary ---")
    print(f"Total Features: {df_features.shape[1]}")
    print(f"Features with any missing values: {sum(missing_counts > 0)}")
    print(f"Features with > 10% missing: {sum(missing_pct > 10)}")
    print(f"Features with > 30% missing: {sum(missing_pct > 30)}")
    print(f"Features with > 50% missing: {sum(missing_pct > 50)}")
    
    # Top 10 features with most missing values
    print("\nTop 10 features with most missing values:")
    print(missing_df.head(10))

    # Plot histogram of missing values percentages
    plt.figure(figsize=(8, 4))
    sns.histplot(missing_pct, bins=30, kde=True, color='salmon')
    plt.title('Distribution of Missing Values Percentage per Sensor')
    plt.xlabel('Missing Percentage (%)')
    plt.ylabel('Number of Sensors')
    missing_plot_path = os.path.join(REPORTS_DIR, "missing_values_distribution.png")
    plt.savefig(missing_plot_path)
    plt.close()
    print(f"Saved missing values distribution plot to {missing_plot_path}")

    # 3.3 Zero Variance / Constant Features
    # Features with zero variance are constant and contain no useful information
    variances = df_features.var(ddof=0)
    constant_features = variances[variances == 0].index.tolist()
    print("\n--- Constant Features Summary ---")
    print(f"Number of constant features (variance=0): {len(constant_features)}")
    if len(constant_features) > 0:
        print(f"Example constant features: {constant_features[:10]}")

    # 3.4 Temporal analysis
    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values(by='timestamp')
    print("\n--- Time Range of Dataset ---")
    print(f"Start date: {df['timestamp'].min()}")
    print(f"End date: {df['timestamp'].max()}")

    # Write a detailed markdown report
    report_path = os.path.join(REPORTS_DIR, "eda_report.md")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# SECOM Dataset EDA Report\n\n")
        rf.write("## 1. Dataset Dimensions\n")
        rf.write(f"- Total wafers (samples): {total_count}\n")
        rf.write(f"- Total sensor readings (features): {df_features.shape[1]}\n\n")
        
        rf.write("## 2. Yield Class Distribution (Imbalance)\n")
        rf.write(f"- **Pass (0):** {pass_count} samples ({pass_count/total_count*100:.2f}%)\n")
        rf.write(f"- **Fail (1):** {fail_count} samples ({fail_count/total_count*100:.2f}%)\n")
        rf.write(f"- **Imbalance Ratio:** {pass_count/fail_count:.2f}:1\n\n")
        rf.write("> [!IMPORTANT]\n")
        rf.write("> Extreme class imbalance detected. Precision-Recall AUC, F1-Score, and Recall should be used as evaluation metrics instead of Accuracy. Resampling (SMOTE) or algorithmic weighting will be required during modeling.\n\n")
        
        rf.write("## 3. Missing Value Diagnostics\n")
        rf.write(f"- Total features with missing values: {sum(missing_counts > 0)} / {df_features.shape[1]} ({sum(missing_counts > 0)/df_features.shape[1]*100:.1f}%)\n")
        rf.write(f"- Features with > 10% missing: {sum(missing_pct > 10)}\n")
        rf.write(f"- Features with > 50% missing: {sum(missing_pct > 50)}\n\n")
        rf.write("### Top 10 Features with Most Missing Values\n")
        rf.write("| Feature | Missing Count | Missing Percentage (%)\n")
        rf.write("| --- | --- | ---\n")
        for idx, row in missing_df.head(10).iterrows():
            rf.write(f"| {idx} | {int(row['missing_count'])} | {row['missing_percentage']:.2f}%\n")
        rf.write("\n")
        
        rf.write("## 4. Zero Variance Features (Constants)\n")
        rf.write(f"- **Count:** {len(constant_features)} constant features identified.\n")
        if len(constant_features) > 0:
            rf.write("- Constant features are non-informative and should be dropped during preprocessing.\n")
            rf.write(f"- List of constant features: `{', '.join(constant_features[:30])}...`\n\n")

        rf.write("## 5. Timeline Analysis\n")
        rf.write(f"- **Data Collection Period:** {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
    print(f"\nSaved markdown report to {report_path}")
    print("EDA step completed successfully.")

if __name__ == "__main__":
    main()
