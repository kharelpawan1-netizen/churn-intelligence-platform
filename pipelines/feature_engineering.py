# pipelines/feature_engineering.py
# Transforms cleaned data into model-ready features.

import pandas as pd


def load_clean_data(path: str = "data/processed/telco_churn_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and encode categorical columns for modeling."""
    df = df.copy()

    # New feature: average monthly spend over the customer's whole lifetime.
    # Different from MonthlyCharges (current rate) — this captures billing history.
    # +1 avoids division by zero for tenure == 0 customers.
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # New feature: is this customer in their first 3 months? Onboarding risk window.
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    # Identify all remaining text (categorical) columns except the target.
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # One-hot encoding: turns e.g. Contract (3 text categories) into 3 separate
    # 0/1 columns, since ML models can't use raw text directly.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def save_features(df: pd.DataFrame, path: str = "data/processed/telco_churn_features.csv") -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Before feature engineering: {df.shape}")

    features_df = engineer_features(df)
    print(f"After feature engineering:  {features_df.shape}")
    print(f"\nNew columns added: avg_monthly_spend, is_new_customer")
    print(f"\nSample of final columns:\n{features_df.columns.tolist()[:10]} ...")

    save_features(features_df)
    print("\nSaved to data/processed/telco_churn_features.csv")