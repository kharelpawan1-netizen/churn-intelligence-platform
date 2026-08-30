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
from app.schemas.custom_model import (
    TrainCustomModelResponse,
    CustomBatchPredictionResponse,
    CustomPredictionResult,
)
from app.services.prediction_service import predict_churn, predict_churn_batch
from app.services.analytics_service import get_dashboard_data
from app.services.custom_training_service import (
    train_custom_model,
    predict_with_custom_model,
    TrainingError,
)
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


@app.get("/dashboard")
def serve_dashboard():
    """Serve the analytics dashboard UI."""
    return FileResponse("static/dashboard.html")


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
    """Predict churn risk for multiple customers from an uploaded CSV.
    The CSV is automatically cleaned and normalized — tolerant of missing
    columns, reordered columns, common naming variants, and messy values."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    try:
        logger.info(f"Batch prediction requested for {len(df)} customers")
        results, cleaning_report = predict_churn_batch(df, db)
        logger.info(
            f"Batch prediction completed: {len(results)} results, "
            f"matched={len(cleaning_report['matched_columns'])}, "
            f"defaulted={len(cleaning_report['defaulted_columns'])}"
        )
        return BatchPredictionResponse(
            total_processed=len(results),
            cleaning_report=cleaning_report,
            results=results,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during batch prediction")


@app.post("/api/v1/train/custom", response_model=TrainCustomModelResponse, dependencies=[Depends(verify_api_key)])
async def train_custom(file: UploadFile = File(...)) -> TrainCustomModelResponse:
    """
    Train a fresh model on the user's own uploaded dataset. The CSV must
    include a target/churn column (e.g. 'Churn', 'Cancelled', 'Target').
    Returns a model_id to use with /predict/custom/{model_id}.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    try:
        logger.info(f"Custom model training requested, {len(df)} rows, {len(df.columns)} columns")
        result = train_custom_model(df)
        logger.info(
            f"Custom model trained: id={result['model_id']}, "
            f"auc={result['roc_auc_mean']}, confidence={result['confidence']}"
        )
        return TrainCustomModelResponse(**result)
    except TrainingError as e:
        logger.warning(f"Custom training rejected: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Custom training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during custom model training")


@app.post(
    "/api/v1/predict/custom/{model_id}",
    response_model=CustomBatchPredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict_custom(model_id: str, file: UploadFile = File(...)) -> CustomBatchPredictionResponse:
    """Predict using a previously trained custom model, on new uploaded rows."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    try:
        logger.info(f"Custom prediction requested for model {model_id}, {len(df)} rows")
        raw_results = predict_with_custom_model(model_id, df)
        results = [CustomPredictionResult(**r) for r in raw_results]
        return CustomBatchPredictionResponse(
            model_id=model_id,
            total_processed=len(results),
            results=results,
        )
    except TrainingError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Custom prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during custom prediction")


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


@app.get("/api/v1/analytics/dashboard")
def analytics_dashboard(db: Session = Depends(get_db)) -> dict:
    """Aggregate analytics for the dashboard: revenue at risk, retention curve,
    churn by contract, and top model drivers."""
    try:
        return get_dashboard_data(db)
    except Exception as e:
        logger.error(f"Dashboard data failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error building dashboard data")