# pipelines/clean_data.py
# Cleans the raw Telco churn dataset and saves a processed version.

import pandas as pd


def load_raw_data(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps to the raw churn data."""
    df = df.copy()

    # TotalCharges is read as text because a few rows have blank strings
    # instead of numbers (new customers with 0 tenure). Force to numeric;
    # anything that can't convert becomes NaN (missing).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Those blanks are customers with 0 months of tenure, so 0 total
    # charges makes logical sense — fill with 0 rather than dropping rows.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # customerID is just an identifier, not predictive — drop it.
    df = df.drop(columns=["customerID"])

    # Standardize the target column to a clean 0/1 integer instead of Yes/No text.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/telco_churn_clean.csv") -> None:
    """Save the cleaned DataFrame to disk."""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    print(f"Raw shape: {raw_df.shape}")
    print(f"Missing TotalCharges (as text issues): {raw_df['TotalCharges'].isna().sum()}")

    clean_df = clean_data(raw_df)
    print(f"\nCleaned shape: {clean_df.shape}")
    print(f"Churn value counts:\n{clean_df['Churn'].value_counts()}")
    print(f"\nAny remaining nulls?\n{clean_df.isnull().sum().sum()}")

    save_processed_data(clean_df)
    print("\nSaved to data/processed/telco_churn_clean.csv")