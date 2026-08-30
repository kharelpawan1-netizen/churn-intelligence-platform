# app/services/data_cleaning_service.py
# Makes uploaded CSVs more robust: normalizes column names (with fuzzy
# fallback matching, including per-segment matching for typo+compound
# names), normalizes common value variants, handles missing values, and
# reports what was auto-corrected before prediction runs.

import re
import difflib
import pandas as pd
import numpy as np
from typing import Optional

# Maps common real-world column name variants to our expected schema.
# Keys are lowercase, whitespace/underscore-stripped versions for matching.
# Expanded with more realistic synonyms a different exporter/CRM might use.
COLUMN_ALIASES = {
    "gender": "gender", "sex": "gender",
    "seniorcitizen": "SeniorCitizen", "senior": "SeniorCitizen",
    "issenior": "SeniorCitizen", "agegroup65": "SeniorCitizen",
    "partner": "Partner", "haspartner": "Partner", "married": "Partner",
    "dependents": "Dependents", "hasdependents": "Dependents", "haskids": "Dependents",
    "tenure": "tenure", "tenuremonths": "tenure", "monthsactive": "tenure",
    "customertenure": "tenure", "monthswithcompany": "tenure", "accountage": "tenure",
    "phoneservice": "PhoneService", "haspphone": "PhoneService", "phone": "PhoneService",
    "multiplelines": "MultipleLines", "multiline": "MultipleLines",
    "internetservice": "InternetService", "internettype": "InternetService", "internet": "InternetService",
    "onlinesecurity": "OnlineSecurity", "security": "OnlineSecurity",
    "onlinebackup": "OnlineBackup", "backup": "OnlineBackup",
    "deviceprotection": "DeviceProtection", "deviceprotect": "DeviceProtection",
    "techsupport": "TechSupport", "support": "TechSupport",
    "streamingtv": "StreamingTV", "tv": "StreamingTV",
    "streamingmovies": "StreamingMovies", "movies": "StreamingMovies",
    "contract": "Contract", "contracttype": "Contract", "plantype": "Contract", "billingcycle": "Contract",
    "paperlessbilling": "PaperlessBilling", "paperless": "PaperlessBilling", "ebilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod", "paytype": "PaymentMethod", "billingmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges", "monthlycost": "MonthlyCharges",
    "monthlyfee": "MonthlyCharges", "monthlyspend": "MonthlyCharges", "monthlybill": "MonthlyCharges",
    "totalcharges": "TotalCharges", "totalcost": "TotalCharges",
    "totalspend": "TotalCharges", "lifetimevalue": "TotalCharges", "totalbilled": "TotalCharges",
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

MISSING_TOKENS = {"", "na", "n/a", "-", "null", "none", "nan", "unknown", "?"}

# Value normalization: for a given target column, map common raw text
# variants to exactly what the model was trained on.
VALUE_ALIASES = {
    "Contract": {
        "m2m": "Month-to-month", "monthly": "Month-to-month", "month": "Month-to-month",
        "1yr": "One year", "1year": "One year", "annual": "One year", "yearly": "One year",
        "2yr": "Two year", "2year": "Two year", "biennial": "Two year",
    },
    "gender": {
        "m": "Male", "male": "Male", "man": "Male",
        "f": "Female", "female": "Female", "woman": "Female",
    },
    "InternetService": {
        "fiber": "Fiber optic", "fibre": "Fiber optic", "fiberoptic": "Fiber optic",
        "none": "No", "no internet": "No",
    },
}

# Columns that are conceptually yes/no, even if the raw data uses 1/0, Y/N, true/false.
YES_NO_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines",
]
YES_ALIASES = {"y", "yes", "true", "1", "t"}
NO_ALIASES = {"n", "no", "false", "0", "f"}


def _normalize_column_name(col: str) -> str:
    """Lowercase and strip whitespace/underscores for fuzzy matching."""
    return col.strip().lower().replace("_", "").replace(" ", "")


def _split_into_segments(col: str) -> list[str]:
    """
    Splits a raw column name into individual word segments, handling
    underscores, spaces, and camelCase boundaries — e.g. 'Custommer_Gendr'
    -> ['custommer', 'gendr'], 'monthlyCharges' -> ['monthly', 'charges'].
    """
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', col)
    segments = re.split(r'[_\s\-]+', spaced)
    return [s.lower() for s in segments if s]


def _fuzzy_match_column(normalized_col: str, original_col: str = "") -> Optional[str]:
    """
    Fallback for columns not in COLUMN_ALIASES. Tries, in order:
    1. Substring containment (reliable for compound names like 'tenure_months')
    2. Whole-string fuzzy similarity, strict cutoff (catches simple typos like 'contarct')
    3. Per-segment fuzzy matching, strict cutoff (catches typo+compound cases
       like 'Custommer_Gendr' -> segment 'gendr' -> 'gender')
    """
    candidates = list(COLUMN_ALIASES.keys())

    # Pass 1: substring containment.
    substring_hits = [alias for alias in candidates if alias in normalized_col and len(alias) >= 4]
    if substring_hits:
        best = max(substring_hits, key=len)
        return COLUMN_ALIASES[best]

    # Pass 2: whole-string similarity, strict cutoff.
    matches = difflib.get_close_matches(normalized_col, candidates, n=1, cutoff=0.85)
    if matches:
        return COLUMN_ALIASES[matches[0]]

    # Pass 3: per-segment fuzzy matching — split the ORIGINAL column name
    # (before underscore/space stripping) into words, and fuzzy-match each
    # segment individually. A typo confined to one word scores much higher
    # in isolation than as part of a long compound string.
    if original_col:
        segments = _split_into_segments(original_col)
        for segment in segments:
            if len(segment) < 4:
                continue  # too short to fuzzy-match reliably
            seg_matches = difflib.get_close_matches(segment, candidates, n=1, cutoff=0.8)
            if seg_matches:
                return COLUMN_ALIASES[seg_matches[0]]

    return None


def _normalize_value(col: str, value):
    """Map a raw cell value to what the model expects, for the given target column."""
    if pd.isna(value):
        return value

    text = str(value).strip().lower()

    if col in YES_NO_COLUMNS:
        if text in YES_ALIASES:
            return "Yes"
        if text in NO_ALIASES:
            return "No"

    if col in VALUE_ALIASES:
        collapsed = text.replace(" ", "").replace("-", "")
        if collapsed in VALUE_ALIASES[col]:
            return VALUE_ALIASES[col][collapsed]

    return value  # leave as-is if no known mapping applies


def clean_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Takes a raw, possibly-messy uploaded CSV and returns:
    - a cleaned DataFrame with exactly the columns the model expects
    - a report dict describing what was matched, fuzzy-matched, defaulted, or ignored
    """
    report = {
        "matched_columns": [],
        "fuzzy_matched_columns": [],  # e.g. "MonthlyFee -> MonthlyCharges"
        "defaulted_columns": [],
        "ignored_columns": [],
        "rows_with_missing_values_filled": 0,
    }

    # Step 1: match uploaded columns — exact alias first, fuzzy match as fallback.
    rename_map = {}
    matched_targets = set()
    for original_col in df.columns:
        normalized = _normalize_column_name(original_col)

        if normalized in COLUMN_ALIASES:
            target = COLUMN_ALIASES[normalized]
            rename_map[original_col] = target
            matched_targets.add(target)
        else:
            fuzzy_target = _fuzzy_match_column(normalized, original_col)
            if fuzzy_target and fuzzy_target not in matched_targets:
                rename_map[original_col] = fuzzy_target
                matched_targets.add(fuzzy_target)
                report["fuzzy_matched_columns"].append(f"{original_col} -> {fuzzy_target}")
            else:
                report["ignored_columns"].append(original_col)

    df = df.rename(columns=rename_map)
    report["matched_columns"] = [c for c in matched_targets]

    # Step 2: treat common "missing" text tokens as real NaN, across all columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in MISSING_TOKENS else v
        )

    # Step 3: normalize known value variants (Y/N, M2M, Fiber, etc.) in matched columns.
    for col in matched_targets:
        if col in VALUE_ALIASES or col in YES_NO_COLUMNS:
            df[col] = df[col].apply(lambda v: _normalize_value(col, v))

    rows_before_fill = df.isnull().any(axis=1).sum()

    # Step 4: add any missing required columns entirely, filled with defaults;
    # fill remaining missing cell values in columns that ARE present.
    for col, default in REQUIRED_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            report["defaulted_columns"].append(col)
        else:
            if col in ("MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else default)
            else:
                df[col] = df[col].fillna(default)

    report["rows_with_missing_values_filled"] = int(rows_before_fill)

    # Step 5: keep only the columns the model actually needs, in a consistent set.
    df = df[list(REQUIRED_COLUMNS_DEFAULTS.keys())]

    return df, report