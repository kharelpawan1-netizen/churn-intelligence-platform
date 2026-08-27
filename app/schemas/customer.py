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