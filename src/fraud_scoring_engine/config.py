"""Resolve database and MLflow configuration from environment variables.

This module centralizes environment-backed settings used across the project,
including PostgreSQL connection details, repository-relative paths, and MLflow
tracking and artifact locations.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus


def _require(name: str) -> str:
    """Read a required environment variable.

    Args:
        name: Environment variable name (e.g. ``POSTGRES_PASSWORD``).

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_postgres_user() -> str:
    """Return the PostgreSQL username from the environment.

    Reads ``POSTGRES_USER``. Falls back to ``postgres`` if unset.

    Returns:
        The database user name.
    """
    return os.environ.get("POSTGRES_USER", "postgres")


def get_postgres_password() -> str:
    """Return the PostgreSQL password from the environment.

    Reads ``POSTGRES_PASSWORD``. This variable is required.

    Returns:
        The database password.

    Raises:
        RuntimeError: If ``POSTGRES_PASSWORD`` is missing or empty.
    """
    return _require("POSTGRES_PASSWORD")


def get_postgres_host() -> str:
    """Return the PostgreSQL host from the environment.

    Reads ``POSTGRES_HOST``. Falls back to ``localhost`` if unset.

    Returns:
        The database host name or address.
    """
    return os.environ.get("POSTGRES_HOST", "localhost")


def get_postgres_port() -> str:
    """Return the PostgreSQL port from the environment.

    Reads ``POSTGRES_PORT``. Falls back to ``5432`` if unset.

    Returns:
        The database port as a string.
    """
    return os.environ.get("POSTGRES_PORT", "5432")


def get_postgres_db() -> str:
    """Return the PostgreSQL database name from the environment.

    Reads ``POSTGRES_DB``. Falls back to ``fraud_db`` if unset.

    Returns:
        The database name.
    """
    return os.environ.get("POSTGRES_DB", "fraud_db")


def get_mlflow_db() -> str:
    """Return the PostgreSQL database name used for MLflow tracking.

    Reads ``MLFLOW_DB``. Falls back to ``mlflow_db`` if unset.

    Returns:
        The MLflow tracking database name.
    """
    return os.environ.get("MLFLOW_DB", "mlflow_db")


def get_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking store URI.

    If ``MLFLOW_TRACKING_URI`` is set, it is returned unchanged. Otherwise
    builds a ``postgresql+psycopg://`` URL for :func:`get_mlflow_db` using
    the same ``POSTGRES_*`` connection settings as the application database.

    Returns:
        A connection URL suitable for MLflow tracking.

    Raises:
        RuntimeError: If ``POSTGRES_PASSWORD`` is missing when building the URL.
    """
    if url := os.environ.get("MLFLOW_TRACKING_URI"):
        return url
    password = quote_plus(get_postgres_password())
    return (
        f"postgresql+psycopg://{get_postgres_user()}:{password}"
        f"@{get_postgres_host()}:{get_postgres_port()}/{get_mlflow_db()}"
    )


def get_mlflow_artifact_root() -> Path:
    """Return the local directory used for MLflow artifact storage.

    Reads ``MLFLOW_ARTIFACT_ROOT``. Defaults to ``{repo_root}/mlartifacts``.

    Returns:
        Path to the artifact root directory.
    """
    if path := os.environ.get("MLFLOW_ARTIFACT_ROOT"):
        return Path(path)
    return Path(__file__).resolve().parents[2] / "mlartifacts"


def get_database_url() -> str:
    """Return the SQLAlchemy database URL for PostgreSQL.

    If ``DATABASE_URL`` is set, it is returned unchanged. Otherwise builds
    a ``postgresql+psycopg://`` URL from ``POSTGRES_*`` variables via the
    other getters, with the password URL-encoded via ``quote_plus``.

    Returns:
        A connection URL suitable for SQLAlchemy/psycopg.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is unset and ``POSTGRES_PASSWORD``
            is missing (via :func:`get_postgres_password`).
    """
    if url := os.environ.get("DATABASE_URL"):
        return url
    password = quote_plus(get_postgres_password())
    return (
        f"postgresql+psycopg://{get_postgres_user()}:{password}"
        f"@{get_postgres_host()}:{get_postgres_port()}/{get_postgres_db()}"
    )
