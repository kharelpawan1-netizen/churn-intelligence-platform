# app/schemas/customer.py
# Defines the shape of data the prediction API expects and returns.

from pydantic import BaseModel


class CustomerData(BaseModel):
    """Raw customer data, matching the original dataset's columns (minus Churn)."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    """What we return after making a prediction."""
    churn_probability: float
    will_churn: bool
    risk_level: str

# Add to app/schemas/customer.py

class BatchPredictionResult(BaseModel):
    """One customer's result within a batch."""
    row_index: int
    churn_probability: float
    will_churn: bool
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction request."""
    total_processed: int
    results: list[BatchPredictionResult]