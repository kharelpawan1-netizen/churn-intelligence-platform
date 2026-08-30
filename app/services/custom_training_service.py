# app/services/custom_training_service.py
# Trains a fresh, simple churn-style model on a user's own uploaded dataset.
# Distinct from the main production model — this is "bring your own data."

import uuid
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODELS_DIR = Path("models/custom")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS = 50          # below this, we refuse to train at all
LOW_CONFIDENCE_ROWS = 150  # below this, metrics are flagged low-confidence
CV_FOLDS = 5

TARGET_NAME_CANDIDATES = [
    "churn", "churned", "target", "label", "cancelled", "canceled",
    "attrition", "left", "exited", "is_churn", "will_churn",
]

POSITIVE_VALUE_ALIASES = {"yes", "y", "true", "1", "churn", "churned", "cancelled", "canceled", "left", "exited"}


class TrainingError(Exception):
    """Raised when the uploaded data can't be used to train a model, with a
    human-readable reason the API can pass straight back to the user."""
    pass


def _detect_target_column(df: pd.DataFrame) -> str:
    normalized_cols = {c.strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for candidate in TARGET_NAME_CANDIDATES:
        if candidate in normalized_cols:
            return normalized_cols[candidate]
    raise TrainingError(
        "Couldn't find a target/churn column. Please include a column named "
        "something like 'Churn', 'Target', or 'Cancelled' marking which rows churned."
    )


def _binarize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1}:
            return series.astype(int)
        raise TrainingError(
            f"Target column has numeric values other than 0/1: {unique_vals}. "
            "Target must be binary (0/1 or Yes/No)."
        )
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.apply(lambda v: 1 if v in POSITIVE_VALUE_ALIASES else 0)


def train_custom_model(df: pd.DataFrame) -> dict:
    """
    Trains a fresh model on the user's uploaded dataset, using k-fold
    cross-validation to report robust performance metrics (rather than a
    single small train/test split, which is unreliable on small datasets).
    """
    if len(df) < MIN_ROWS:
        raise TrainingError(
            f"Dataset too small ({len(df)} rows). Need at least {MIN_ROWS} rows "
            "to train and validate a meaningful model."
        )

    target_col = _detect_target_column(df)
    y = _binarize_target(df[target_col])

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise TrainingError("Target column has only one class present — need both churned and non-churned examples.")
    if class_counts.min() < 10:
        raise TrainingError(
            f"The minority class only has {class_counts.min()} examples — need at least 10 of "
            "each class to get a meaningful, non-noisy estimate of model performance."
        )

    X = df.drop(columns=[target_col])

    id_like_cols = [c for c in X.columns if X[c].nunique() == len(X)]
    X = X.drop(columns=id_like_cols)

    if X.shape[1] == 0:
        raise TrainingError("No usable feature columns remain after removing the target and ID-like columns.")

    X = X.dropna(thresh=int(X.shape[1] * 0.5))
    y = y.loc[X.index]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown")

    safe_categorical_cols = [c for c in categorical_cols if X[c].nunique() <= 30]
    dropped_high_cardinality = [c for c in categorical_cols if c not in safe_categorical_cols]
    X = X.drop(columns=dropped_high_cardinality)

    X_encoded = pd.get_dummies(X, columns=safe_categorical_cols, drop_first=True)

    if X_encoded.shape[1] == 0:
        raise TrainingError("No usable features remained after cleaning and encoding.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # --- Cross-validated metrics: the honest, robust performance estimate ---
    # Instead of one lucky/unlucky train/test split, we rotate through 5
    # different splits and average — much less noisy on small datasets, and
    # it directly surfaces HOW variable the model's performance actually is.
    n_folds = min(CV_FOLDS, class_counts.min())  # can't have more folds than the smallest class
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    base_model = LogisticRegression(max_iter=1000)

    auc_scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring="roc_auc")
    precision_scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring="precision")
    recall_scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring="recall")

    # --- Final model: trained on ALL the data, for actual deployment/prediction use ---
    # (Cross-validation above is only for honestly measuring performance;
    # the model we actually save uses every available row.)
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_scaled, y)

    confidence = "low" if len(df) < LOW_CONFIDENCE_ROWS else "moderate" if len(df) < 500 else "good"

    model_id = str(uuid.uuid4())[:8]
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(
        {
            "feature_columns": X_encoded.columns.tolist(),
            "target_column": target_col,
            "dropped_id_columns": id_like_cols,
            "dropped_high_cardinality_columns": dropped_high_cardinality,
            "categorical_columns": safe_categorical_cols,
        },
        model_dir / "metadata.pkl",
    )

    return {
        "model_id": model_id,
        "rows_used": len(X_encoded),
        "features_used": X_encoded.shape[1],
        "target_column_detected": target_col,
        "dropped_id_columns": id_like_cols,
        "dropped_high_cardinality_columns": dropped_high_cardinality,
        "cv_folds_used": n_folds,
        "roc_auc_mean": round(float(np.mean(auc_scores)), 4),
        "roc_auc_std": round(float(np.std(auc_scores)), 4),
        "precision_mean": round(float(np.mean(precision_scores)), 4),
        "recall_mean": round(float(np.mean(recall_scores)), 4),
        "confidence": confidence,
    }


def predict_with_custom_model(model_id: str, df: pd.DataFrame) -> list[dict]:
    """Load a previously trained custom model and predict on new data."""
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise TrainingError(f"No trained model found with id '{model_id}'.")

    model = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    metadata = joblib.load(model_dir / "metadata.pkl")

    feature_columns = metadata["feature_columns"]
    categorical_columns = metadata["categorical_columns"]
    dropped_cols = (
        metadata["dropped_id_columns"]
        + metadata["dropped_high_cardinality_columns"]
        + [metadata["target_column"]]
    )

    X = df.drop(columns=[c for c in dropped_cols if c in df.columns], errors="ignore")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = X[col].fillna("Unknown")

    present_categorical = [c for c in categorical_columns if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=present_categorical, drop_first=True)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(X_encoded)
    probs = model.predict_proba(scaled)[:, 1]

    return [
        {"row_index": int(idx), "churn_probability": round(float(p), 4), "will_churn": bool(p >= 0.5)}
        for idx, p in zip(df.index, probs)
    ]