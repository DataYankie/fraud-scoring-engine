"""Tests for fraud_scoring_engine.config."""

from pathlib import Path

import pytest

from fraud_scoring_engine.config import (
    _require,
    get_database_url,
    get_mlflow_artifact_root,
    get_mlflow_db,
    get_mlflow_tracking_uri,
    get_postgres_db,
    get_postgres_host,
    get_postgres_password,
    get_postgres_port,
    get_postgres_user,
)


def test__require(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_REQUIRED_VAR", "value")
    assert _require("TEST_REQUIRED_VAR") == "value"

    monkeypatch.delenv("TEST_REQUIRED_VAR", raising=False)
    with pytest.raises(RuntimeError, match="Missing environment variable: TEST_REQUIRED_VAR"):
        _require("TEST_REQUIRED_VAR")

    monkeypatch.setenv("TEST_REQUIRED_VAR", "")
    with pytest.raises(RuntimeError, match="Missing environment variable: TEST_REQUIRED_VAR"):
        _require("TEST_REQUIRED_VAR")


def test_get_postgres_user() -> None:
    assert get_postgres_user() == "postgres"


def test_get_postgres_user_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "appuser")
    assert get_postgres_user() == "appuser"


def test_get_postgres_password(postgres_env: dict[str, str]) -> None:
    assert get_postgres_password() == postgres_env["POSTGRES_PASSWORD"]


def test_get_postgres_password_raises_when_unset() -> None:
    with pytest.raises(RuntimeError, match="Missing environment variable: POSTGRES_PASSWORD"):
        get_postgres_password()


def test_get_postgres_host() -> None:
    assert get_postgres_host() == "localhost"


def test_get_postgres_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    assert get_postgres_host() == "db.example.com"


def test_get_postgres_port() -> None:
    assert get_postgres_port() == "5432"


def test_get_postgres_port_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    assert get_postgres_port() == "5433"


def test_get_postgres_db() -> None:
    assert get_postgres_db() == "fraud_db"


def test_get_postgres_db_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DB", "other_db")
    assert get_postgres_db() == "other_db"


def test_get_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+psycopg://user:secret@host:5432/mydb"
    monkeypatch.setenv("DATABASE_URL", url)
    assert get_database_url() == url


def test_get_database_url_builds_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "fraud_db")

    url = get_database_url()

    assert url.startswith("postgresql+psycopg://postgres:p%40ss@localhost:5432/fraud_db")


def test_get_mlflow_db() -> None:
    assert get_mlflow_db() == "mlflow_db"


def test_get_mlflow_db_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_DB", "custom_mlflow")
    assert get_mlflow_db() == "custom_mlflow"


def test_get_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+psycopg://user:secret@host:5432/mlflow_db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", url)
    assert get_mlflow_tracking_uri() == url


def test_get_mlflow_tracking_uri_builds_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("MLFLOW_DB", "mlflow_db")

    url = get_mlflow_tracking_uri()

    assert url.startswith("postgresql+psycopg://postgres:p%40ss@localhost:5432/mlflow_db")


def test_get_mlflow_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(artifact_root))
    assert get_mlflow_artifact_root() == artifact_root
