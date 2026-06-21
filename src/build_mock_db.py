import os
import sqlite3
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "manufacturing_legacy.db")
MERGED_CSV_PATH = os.path.join(DATA_DIR, "secom_merged.csv")

def main():
    if not os.path.exists(MERGED_CSV_PATH):
        print(f"Error: {MERGED_CSV_PATH} not found. Run download_and_eda.py first.")
        return

    print("Loading merged SECOM data...")
    df = pd.read_csv(MERGED_CSV_PATH)
    total_samples = len(df)

    # 1. Generate Mock MES Metadata
    print("Generating simulated MES metadata (Lots, Wafer IDs, Operators, Recipes)...")
    # A standard lot has 25 wafers
    wafers_per_lot = 25
    lot_numbers = [f"LOT_{i:04d}" for i in range(1, (total_samples // wafers_per_lot) + 2)]
    
    lots = []
    wafer_ids = []
    for i in range(total_samples):
        lot_idx = i // wafers_per_lot
        lots.append(lot_numbers[lot_idx])
        wafer_ids.append((i % wafers_per_lot) + 1)

    # Simulated operators and recipes
    np.random.seed(42)
    operators = [f"OP_{x}" for x in ["A", "B", "C", "D", "E"]]
    recipes = [f"RECIPE_{x}" for x in ["NAND_ETCH_20nm", "DRAM_DIFF_1z", "HBM_BOND_3D"]]

    df['lot_id'] = lots
    df['wafer_id'] = wafer_ids
    df['operator_id'] = np.random.choice(operators, size=total_samples)
    df['recipe'] = np.random.choice(recipes, size=total_samples)
    df['sample_id'] = range(total_samples)
    df['yield_status'] = df['label'].map({-1: 0, 1: 1}) # 0: Pass, 1: Fail

    # 2. Split into MES (Metadata) and FDC (Sensor Data) DataFrames
    df_mes = df[['sample_id', 'lot_id', 'wafer_id', 'timestamp', 'operator_id', 'recipe', 'yield_status']].copy()
    
    sensor_cols = [f"sensor_{i}" for i in range(590)]
    df_fdc = df[['sample_id'] + sensor_cols].copy()

    # 3. Create SQLite Database and Tables
    print(f"Connecting to SQLite database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create MES Table
    print("Creating 'mes_lot_wafers' table...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mes_lot_wafers (
        sample_id INTEGER PRIMARY KEY,
        lot_id TEXT NOT NULL,
        wafer_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        operator_id TEXT NOT NULL,
        recipe TEXT NOT NULL,
        yield_status INTEGER NOT NULL
    );
    """)

    # Create FDC Table (dynamically building schema for 590 sensors)
    print("Creating 'fdc_sensor_data' table...")
    sensor_schema_parts = [f"sensor_{i} REAL" for i in range(590)]
    fdc_table_sql = f"""
    CREATE TABLE IF NOT EXISTS fdc_sensor_data (
        sample_id INTEGER PRIMARY KEY,
        {', '.join(sensor_schema_parts)},
        FOREIGN KEY(sample_id) REFERENCES mes_lot_wafers(sample_id)
    );
    """
    cursor.execute(fdc_table_sql)
    conn.commit()

    # 4. Insert Data
    print("Inserting data into tables...")
    df_mes.to_sql('mes_lot_wafers', conn, if_exists='replace', index=False)
    df_fdc.to_sql('fdc_sensor_data', conn, if_exists='replace', index=False)
    conn.commit()

    # 5. Verify Database Integrity and Run Sample Queries
    print("\n=== Verification and Mock Database Queries ===")
    
    # Query 1: Total records count
    cursor.execute("SELECT COUNT(*) FROM mes_lot_wafers;")
    print(f"Total records in mes_lot_wafers: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM fdc_sensor_data;")
    print(f"Total records in fdc_sensor_data: {cursor.fetchone()[0]}")

    # Query 2: Retrieve fail count per Operator
    print("\nFail counts by Operator:")
    cursor.execute("""
        SELECT operator_id, COUNT(*) as total_wafers, SUM(yield_status) as fails, 
               ROUND(CAST(SUM(yield_status) AS REAL) / COUNT(*) * 100, 2) as fail_rate_pct
        FROM mes_lot_wafers
        GROUP BY operator_id;
    """)
    for row in cursor.fetchall():
        print(f"Operator {row[0]}: Total Wafers={row[1]}, Fails={row[2]} ({row[3]}%)")

    # Query 3: Join MES and FDC to retrieve specific sensor readings for failed wafers
    print("\nSample Join Query (Retriving first 5 failed wafers and sensor_0, sensor_1):")
    cursor.execute("""
        SELECT m.lot_id, m.wafer_id, m.yield_status, f.sensor_0, f.sensor_1
        FROM mes_lot_wafers m
        JOIN fdc_sensor_data f ON m.sample_id = f.sample_id
        WHERE m.yield_status = 1
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        print(f"Lot {row[0]} | Wafer {row[1]} | Status={row[2]} | Sensor_0={row[3]} | Sensor_1={row[4]}")

    conn.close()
    print("\nSQLite Database construction and data loading completed successfully.")

if __name__ == "__main__":
    main()
