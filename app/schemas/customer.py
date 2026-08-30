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


class TopFactor(BaseModel):
    """One feature's contribution to a specific prediction."""
    feature: str
    impact: float  # positive = pushed toward churn, negative = pushed away from it


class PredictionResponse(BaseModel):
    """What we return after making a prediction."""
    churn_probability: float
    will_churn: bool
    risk_level: str
    top_factors: list[TopFactor] = []


class BatchPredictionResult(BaseModel):
    """One customer's result within a batch."""
    row_index: int
    churn_probability: float
    will_churn: bool
    risk_level: str


class CleaningReport(BaseModel):
    """Describes what the auto-cleaning step did to the uploaded CSV."""
    matched_columns: list[str]
    defaulted_columns: list[str]
    ignored_columns: list[str]
    rows_with_missing_values_filled: int


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction request."""
    total_processed: int
    cleaning_report: CleaningReport
    results: list[BatchPredictionResult]