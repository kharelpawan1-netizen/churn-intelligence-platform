# scripts/inspect_data.py
# First look at the raw churn dataset: shape, columns, and a sample of rows.

import pandas as pd

df = pd.read_csv("data/raw/telco_churn.csv")

print(f"Rows, columns: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())