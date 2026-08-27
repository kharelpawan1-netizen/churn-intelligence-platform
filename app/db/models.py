# app/db/models.py
# Database table definitions.

from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime, timezone

from app.db.database import Base


class PredictionLog(Base):
    """One row per prediction made through the API."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    churn_probability = Column(Float)
    will_churn = Column(String)  # stored as string for simplicity
    risk_level = Column(String)
    contract_type = Column(String)
    tenure = Column(Integer)
    monthly_charges = Column(Float)