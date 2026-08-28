# app/services/analytics_service.py
# Computes aggregate analytics for the dashboard: revenue at risk, cohort
# retention curves, and top model drivers.

import pandas as pd
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import PredictionLog

model = joblib.load("models/churn_model.pkl")
metadata = joblib.load("models/model_metadata.pkl")
FEATURE_COLUMNS = metadata["feature_columns"]

# Training data, used for cohort/contract analysis — this is historical,
# not live prediction data, so it doesn't change between requests.
_clean_df = pd.read_csv("data/processed/telco_churn_clean.csv")


def get_revenue_at_risk(db: Session) -> dict:
    """Sum of (churn_probability * monthly_charges) across all logged predictions —
    an estimate of total monthly revenue currently at risk of churning."""
    logs = db.query(PredictionLog.churn_probability, PredictionLog.monthly_charges).all()

    if not logs:
        return {"total_revenue_at_risk": 0.0, "predictions_counted": 0}

    total_at_risk = sum(prob * charges for prob, charges in logs)
    return {
        "total_revenue_at_risk": round(total_at_risk, 2),
        "predictions_counted": len(logs),
    }


def get_churn_by_contract() -> list[dict]:
    """Churn rate per contract type, from the historical training data."""
    grouped = _clean_df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    return [
        {"contract": contract, "churn_rate": round(rate, 4)}
        for contract, rate in grouped.items()
    ]


def get_retention_curve() -> list[dict]:
    """Churn rate by tenure bucket — shows the 'new customer risk window'."""
    df = _clean_df.copy()
    bins = [0, 3, 12, 24, 100]
    labels = ["0-3 months", "3-12 months", "12-24 months", "24+ months"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

    grouped = df.groupby("tenure_bucket", observed=True)["Churn"].mean()
    return [
        {"bucket": bucket, "churn_rate": round(rate, 4)}
        for bucket, rate in grouped.items()
    ]


def get_top_model_drivers(top_n: int = 8) -> list[dict]:
    """Global feature importance from the logistic regression's own coefficients —
    a fast, aggregate view (distinct from the per-prediction SHAP explanations)."""
    coefficients = model.coef_[0]
    pairs = list(zip(FEATURE_COLUMNS, coefficients))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    return [
        {"feature": name, "coefficient": round(float(coef), 4)}
        for name, coef in pairs[:top_n]
    ]


def get_dashboard_data(db: Session) -> dict:
    """Assemble all dashboard sections into one response."""
    total_predictions = db.query(func.count(PredictionLog.id)).scalar()

    risk_counts = (
        db.query(PredictionLog.risk_level, func.count(PredictionLog.id))
        .group_by(PredictionLog.risk_level)
        .all()
    )

    return {
        "total_predictions": total_predictions,
        "risk_breakdown": dict(risk_counts),
        "revenue_at_risk": get_revenue_at_risk(db),
        "churn_by_contract": get_churn_by_contract(),
        "retention_curve": get_retention_curve(),
        "top_model_drivers": get_top_model_drivers(),
    }