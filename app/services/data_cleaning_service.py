# app/services/data_cleaning_service.py
# Makes uploaded CSVs more robust: normalizes column names, handles missing
# values, and reports what was auto-corrected before prediction runs.

import pandas as pd
import numpy as np

# Maps common real-world column name variants to our expected schema.
# Keys are lowercase, whitespace/underscore-stripped versions for matching.
COLUMN_ALIASES = {
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "senior": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "tenuremonths": "tenure",
    "phoneservice": "PhoneService",
    "multiplelines": "MultipleLines",
    "internetservice": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "techsupport": "TechSupport",
    "streamingtv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "contract": "Contract",
    "contracttype": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "monthlycost": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "totalcost": "TotalCharges",
}

# Every column the model genuinely requires, and a sensible default when missing.
REQUIRED_COLUMNS_DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0,
}

MISSING_TOKENS = {"", "na", "n/a", "-", "null", "none", "nan"}


def _normalize_column_name(col: str) -> str:
    """Lowercase and strip whitespace/underscores for fuzzy matching."""
    return col.strip().lower().replace("_", "").replace(" ", "")


def clean_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Takes a raw, possibly-messy uploaded CSV and returns:
    - a cleaned DataFrame with exactly the columns the model expects
    - a report dict describing what was matched, defaulted, or ignored
    """
    report = {
        "matched_columns": [],
        "defaulted_columns": [],
        "ignored_columns": [],
        "rows_with_missing_values_filled": 0,
    }

    # Step 1: match uploaded columns to our known schema by normalized name.
    rename_map = {}
    matched_targets = set()
    for original_col in df.columns:
        normalized = _normalize_column_name(original_col)
        if normalized in COLUMN_ALIASES:
            target = COLUMN_ALIASES[normalized]
            rename_map[original_col] = target
            matched_targets.add(target)
        else:
            report["ignored_columns"].append(original_col)

    df = df.rename(columns=rename_map)
    report["matched_columns"] = list(matched_targets)

    # Step 2: treat common "missing" text tokens as real NaN, across all columns.
    df = df.replace(
        {col: {token: np.nan for token in MISSING_TOKENS}
         for col in df.columns if df[col].dtype == object}
    )
    # Also catch case-insensitive versions (e.g. "N/A" vs "n/a").
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in MISSING_TOKENS else v
        )

    rows_before_fill = df.isnull().any(axis=1).sum()

    # Step 3: add any missing required columns entirely, filled with defaults.
    for col, default in REQUIRED_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            report["defaulted_columns"].append(col)
        else:
            # Fill missing individual cell values within a present column.
            if pd.api.types.is_numeric_dtype(df[col]) or col in ("MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else default)
            else:
                df[col] = df[col].fillna(default)

    report["rows_with_missing_values_filled"] = int(rows_before_fill)

    # Step 4: keep only the columns the model actually needs, in a consistent set.
    df = df[list(REQUIRED_COLUMNS_DEFAULTS.keys())]

    return df, report