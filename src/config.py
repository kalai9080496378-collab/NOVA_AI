from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "NOVA AI"
    APP_ENV: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Database URLs
    # postgresql+asyncpg://... for async FastAPI operations
    # postgresql+psycopg://... for sync Alembic migrations
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nova_ai"
    DATABASE_URL_SYNC: str = "postgresql+psycopg://postgres:postgres@localhost:5432/nova_ai"

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Security & JWT
    JWT_SECRET_KEY: str = "nova-ai-development-super-secret-key-32-chars-min"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Rate Limiting
    LOGIN_RATE_LIMIT_MAX_ATTEMPTS: int = 5
    LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Video Provider
    VIDEO_PROVIDER: str = "runapi"  # "runapi" or "mock"
    RUNAPI_API_KEY: str = ""

    # LLM Provider
    LLM_PROVIDER: str = "gemini"  # "gemini" or "mock"
    LLM_API_KEY: str = ""

    # TTS Provider
    TTS_PROVIDER: str = "mock"

    # Storage
    STORAGE_DIR: str = "storage"

    # Cost Protection Quota
    DEFAULT_MONTHLY_QUOTA: int = 3


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
