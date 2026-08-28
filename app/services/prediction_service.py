# app/services/prediction_service.py
# Loads the trained model and handles turning raw customer data into predictions.

import pandas as pd
import joblib
import shap
from sqlalchemy.orm import Session

from app.schemas.customer import (
    CustomerData,
    PredictionResponse,
    BatchPredictionResult,
    TopFactor,
)
from app.db.models import PredictionLog

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
metadata = joblib.load("models/model_metadata.pkl")
shap_background = joblib.load("models/shap_background.pkl")

THRESHOLD = metadata["threshold"]
FEATURE_COLUMNS = metadata["feature_columns"]
# The production model_metadata.pkl doesn't currently store a "version" key
# (only the timestamped models/versions/<ts>/model_metadata.pkl does) — fall
# back to "unknown" rather than crashing if it's missing.
MODEL_VERSION = metadata.get("version", "unknown")

# Built once at startup, reused for every prediction — this is what makes
# per-request SHAP explanations fast enough to use in a live API.
explainer = shap.LinearExplainer(model, shap_background)


def _encode_row(row: pd.Series) -> pd.DataFrame:
    """Shared encoding logic: raw customer row -> model-ready feature row."""
    row_df = pd.DataFrame([row])
    row_df["avg_monthly_spend"] = row_df["TotalCharges"] / (row_df["tenure"] + 1)
    row_df["is_new_customer"] = (row_df["tenure"] <= 3).astype(int)

    categorical_cols = row_df.select_dtypes(include="object").columns.tolist()
    encoded_df = pd.get_dummies(row_df, columns=categorical_cols, drop_first=True)
    return encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def _get_top_factors(scaled_row, top_n: int = 3) -> list[TopFactor]:
    """Compute the top N features driving this specific prediction, by |impact|."""
    shap_values = explainer(scaled_row)
    impacts = shap_values.values[0]  # one row -> 1D array of per-feature impacts

    factor_pairs = list(zip(FEATURE_COLUMNS, impacts))
    factor_pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)

    return [
        TopFactor(feature=name, impact=round(float(impact), 4))
        for name, impact in factor_pairs[:top_n]
    ]


def predict_churn(customer: CustomerData, db: Session) -> PredictionResponse:
    """Run the full pipeline: raw input -> engineered features -> prediction -> log to DB."""
    encoded_df = _encode_row(pd.Series(customer.model_dump()))
    scaled = scaler.transform(encoded_df)

    probability = model.predict_proba(scaled)[0, 1]
    will_churn = probability >= THRESHOLD

    if probability >= 0.5:
        risk_level = "high"
    elif probability >= THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    top_factors = _get_top_factors(scaled)

    log_entry = PredictionLog(
        churn_probability=round(float(probability), 4),
        will_churn=str(will_churn),
        risk_level=risk_level,
        contract_type=customer.Contract,
        tenure=customer.tenure,
        monthly_charges=customer.MonthlyCharges,
        model_version=MODEL_VERSION,
    )
    db.add(log_entry)
    db.commit()

    return PredictionResponse(
        churn_probability=round(float(probability), 4),
        will_churn=bool(will_churn),
        risk_level=risk_level,
        top_factors=top_factors,
    )


def predict_churn_batch(df: pd.DataFrame, db: Session) -> list[BatchPredictionResult]:
    """Run predictions for multiple customers at once, from an uploaded CSV."""
    results = []

    for idx, row in df.iterrows():
        encoded_df = _encode_row(row)
        scaled = scaler.transform(encoded_df)

        probability = model.predict_proba(scaled)[0, 1]
        will_churn = probability >= THRESHOLD

        if probability >= 0.5:
            risk_level = "high"
        elif probability >= THRESHOLD:
            risk_level = "medium"
        else:
            risk_level = "low"

        log_entry = PredictionLog(
            churn_probability=round(float(probability), 4),
            will_churn=str(will_churn),
            risk_level=risk_level,
            contract_type=row.get("Contract", "unknown"),
            tenure=int(row.get("tenure", 0)),
            monthly_charges=float(row.get("MonthlyCharges", 0)),
            model_version=MODEL_VERSION,
        )
        db.add(log_entry)

        results.append(BatchPredictionResult(
            row_index=idx,
            churn_probability=round(float(probability), 4),
            will_churn=bool(will_churn),
            risk_level=risk_level,
        ))

    db.commit()
    return results