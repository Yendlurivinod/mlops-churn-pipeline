import pandas as pd
import numpy as np
from datetime import datetime
import os

EXPECTED_COLUMNS = ['customer_id', 'transaction_id', 'last_purchase', 'amount', 'churned']

def validate_schema(df):
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")
    print(f"✓ Schema validated — {len(df)} records ingested")
    return df

def validate_data_quality(df):
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"⚠ Null values found:\n{null_counts[null_counts > 0]}")
    duplicates = df.duplicated(subset=['transaction_id']).sum()
    if duplicates > 0:
        print(f"⚠ {duplicates} duplicate transaction IDs found — dropping")
        df = df.drop_duplicates(subset=['transaction_id'])
    print(f"✓ Data quality check passed")
    return df

def compute_features(df):
    df['last_purchase'] = pd.to_datetime(df['last_purchase'])
    now = datetime.now()
    df['recency'] = (now - df['last_purchase']).dt.days
    freq = df.groupby('customer_id')['transaction_id'].count().reset_index()
    freq.columns = ['customer_id', 'frequency']
    mon = df.groupby('customer_id')['amount'].sum().reset_index()
    mon.columns = ['customer_id', 'monetary']
    df = df.merge(freq, on='customer_id', how='left')
    df = df.merge(mon, on='customer_id', how='left')
    df['avg_order_value'] = df['monetary'] / df['frequency']
    df['high_value'] = (df['monetary'] > df['monetary'].median()).astype(int)
    df['recency_score'] = pd.qcut(df['recency'], q=4, labels=[4, 3, 2, 1]).astype(int)
    print(f"✓ Features computed — {df.shape[1]} features ready")
    return df

def save_reference_data(df, output_dir):
    ref_path = os.path.join(output_dir, 'reference.csv')
    if not os.path.exists(ref_path):
        df.to_csv(ref_path, index=False)
        print(f"✓ Reference dataset saved to {ref_path}")

def run_etl(input_path="data/raw/transactions.csv",
            output_path="data/processed/features.csv"):
    print("\n--- Running ETL Pipeline ---")
    df = pd.read_csv(input_path)
    df = validate_schema(df)
    df = validate_data_quality(df)
    df = compute_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    save_reference_data(df, os.path.dirname(output_path))
    print(f"✓ ETL complete — saved to {output_path}")
    return df

if __name__ == "__main__":
    run_etl()
