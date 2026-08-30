# app/schemas/custom_model.py
# Request/response shapes for the bring-your-own-data training feature.

from pydantic import BaseModel
from typing import Optional


class TrainCustomModelResponse(BaseModel):
    """Returned after successfully training a model on a user's own CSV."""
    model_id: str
    rows_used: int
    features_used: int
    target_column_detected: str
    dropped_id_columns: list[str]
    dropped_high_cardinality_columns: list[str]
    cv_folds_used: int
    roc_auc_mean: float
    roc_auc_std: float
    precision_mean: float
    recall_mean: float
    confidence: str


class CustomPredictionResult(BaseModel):
    """One row's prediction from a custom-trained model."""
    row_index: int
    churn_probability: float
    will_churn: bool


class CustomBatchPredictionResponse(BaseModel):
    """Response for predictions made using a custom-trained model."""
    model_id: str
    total_processed: int
    results: list[CustomPredictionResult]