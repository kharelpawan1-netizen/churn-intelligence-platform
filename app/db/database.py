# app/db/database.py
# Database connection setup.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# SQLite needs this special connect_arg; Postgres doesn't, so only apply it conditionally.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Provides a database session, and guarantees it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()