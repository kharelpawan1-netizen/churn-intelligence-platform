# app/main.py
# Entry point for the FastAPI backend.

import io
import logging

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.core.security import verify_api_key
from app.core.logging_config import setup_logging
from app.schemas.customer import (
    CustomerData,
    PredictionResponse,
    BatchPredictionResponse,
)
from app.services.prediction_service import predict_churn, predict_churn_batch
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

# Serve static frontend files (HTML/CSS/JS) under /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    """Serve the frontend UI at the root URL."""
    return FileResponse("static/index.html")


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


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)) -> BatchPredictionResponse:
    """Predict churn risk for multiple customers from an uploaded CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        logger.info(f"Batch prediction requested for {len(df)} customers")
        results = predict_churn_batch(df, db)
        logger.info(f"Batch prediction completed: {len(results)} results")
        return BatchPredictionResponse(total_processed=len(results), results=results)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during batch prediction")


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