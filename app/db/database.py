# app/db/database.py
# Database connection setup.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite file will be created automatically at this path on first run.
DATABASE_URL = "sqlite:///./churn_platform.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Provides a database session, and guarantees it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()