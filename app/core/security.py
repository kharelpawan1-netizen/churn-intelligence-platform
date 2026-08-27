# app/core/security.py
# API key authentication.

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Checks the X-API-Key request header against our configured secret.
    Raises 401 Unauthorized if missing or wrong.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )