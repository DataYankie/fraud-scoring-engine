"""Shared pytest fixtures for fraud-scoring-engine tests."""

import pytest
from sqlalchemy.engine import Engine

from fraud_scoring_engine.db.engine import create_db_engine

POSTGRES_ENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
)


@pytest.fixture(autouse=True)
def isolate_postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove postgres-related env vars so tests do not leak state."""
    for key in POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def postgres_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set canonical POSTGRES_* values for URL-building tests."""
    values = {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "testpass",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "fraud_db",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


@pytest.fixture
def sqlite_database_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Use in-memory SQLite so engine tests never hit Docker."""
    url = "sqlite:///:memory:"
    monkeypatch.setenv("DATABASE_URL", url)
    return url


@pytest.fixture
def engine(sqlite_database_url: str) -> Engine:
    """SQLAlchemy engine backed by in-memory SQLite."""
    return create_db_engine()
