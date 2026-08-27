# app/core/config.py
# Central configuration for the entire application.
# Every other module reads settings from here — never hardcode config values elsewhere.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    app_name: str = "Churn Intelligence Platform"
    environment: str = "development"
    debug: bool = True
    api_key: str = "changeme"  # overridden by .env in real use
    database_url: str = "sqlite:///./churn_platform.db"  # falls back to SQLite if unset

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# A single, shared instance other modules import and use.
settings = Settings()


if __name__ == "__main__":
    # Quick manual check: run this file directly to print loaded settings.
    print(f"App name:    {settings.app_name}")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode:  {settings.debug}")
    print(f"Database:    {settings.database_url}")