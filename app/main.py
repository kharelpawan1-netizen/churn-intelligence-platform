# app/main.py
# Entry point for the FastAPI backend.

import logging

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.core.security import verify_api_key
from app.core.logging_config import setup_logging
from app.schemas.customer import CustomerData, PredictionResponse
from app.services.prediction_service import predict_churn
from app.db.database import engine, Base, get_db
from app.db import models
from app.db.models import PredictionLog

setup_logging()
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


@app.get("/api/v1/health")
def health_check() -> dict:
    """Basic health check — confirms the API is running."""
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.environment,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(customer: CustomerData, db: Session = Depends(get_db)) -> PredictionResponse:
    """Predict churn risk for a single customer, and log the prediction."""
    try:
        logger.info(f"Prediction requested for customer with tenure={customer.tenure}, contract={customer.Contract}")
        result = predict_churn(customer, db)
        logger.info(f"Prediction result: probability={result.churn_probability}, risk={result.risk_level}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during prediction")


@app.get("/api/v1/monitoring/summary")
def monitoring_summary(db: Session = Depends(get_db)) -> dict:
    """Basic operational stats about predictions made so far."""
    total_predictions = db.query(func.count(PredictionLog.id)).scalar()

    if total_predictions == 0:
        return {"total_predictions": 0, "message": "No predictions logged yet."}

    avg_probability = db.query(func.avg(PredictionLog.churn_probability)).scalar()

    risk_counts = (
        db.query(PredictionLog.risk_level, func.count(PredictionLog.id))
        .group_by(PredictionLog.risk_level)
        .all()
    )

    return {
        "total_predictions": total_predictions,
        "average_churn_probability": round(avg_probability, 4),
        "risk_level_breakdown": dict(risk_counts),
    }