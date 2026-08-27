# app/services/prediction_service.py
# Loads the trained model and handles turning raw customer data into predictions.

import pandas as pd
import joblib
from sqlalchemy.orm import Session

from app.schemas.customer import CustomerData, PredictionResponse, BatchPredictionResult
from app.db.models import PredictionLog

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
metadata = joblib.load("models/model_metadata.pkl")

THRESHOLD = metadata["threshold"]
FEATURE_COLUMNS = metadata["feature_columns"]


def predict_churn(customer: CustomerData, db: Session) -> PredictionResponse:
    """Run the full pipeline: raw input -> engineered features -> prediction -> log to DB."""
    raw_df = pd.DataFrame([customer.model_dump()])

    raw_df["avg_monthly_spend"] = raw_df["TotalCharges"] / (raw_df["tenure"] + 1)
    raw_df["is_new_customer"] = (raw_df["tenure"] <= 3).astype(int)

    categorical_cols = raw_df.select_dtypes(include="object").columns.tolist()
    encoded_df = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)
    encoded_df = encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

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
        contract_type=customer.Contract,
        tenure=customer.tenure,
        monthly_charges=customer.MonthlyCharges,
    )
    db.add(log_entry)
    db.commit()

    return PredictionResponse(
        churn_probability=round(float(probability), 4),
        will_churn=bool(will_churn),
        risk_level=risk_level,
    )


def predict_churn_batch(df: pd.DataFrame, db: Session) -> list[BatchPredictionResult]:
    """Run predictions for multiple customers at once, from an uploaded CSV."""
    results = []

    for idx, row in df.iterrows():
        row_df = pd.DataFrame([row])

        row_df["avg_monthly_spend"] = row_df["TotalCharges"] / (row_df["tenure"] + 1)
        row_df["is_new_customer"] = (row_df["tenure"] <= 3).astype(int)

        categorical_cols = row_df.select_dtypes(include="object").columns.tolist()
        encoded_df = pd.get_dummies(row_df, columns=categorical_cols, drop_first=True)
        encoded_df = encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

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