"""Tests for fraud_scoring_engine.db.engine."""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from fraud_scoring_engine.db.engine import connect, create_db_engine


def test_create_db_engine(engine: Engine) -> None:
    assert engine.url.drivername == "sqlite"


def test_create_db_engine_echo(sqlite_database_url: str) -> None:
    engine = create_db_engine(echo=True)
    assert engine.echo is True


def test_create_db_engine_poolclass(sqlite_database_url: str) -> None:
    engine = create_db_engine(poolclass=NullPool)
    assert engine.pool.__class__.__name__ == "NullPool"


def test_connect(engine: Engine) -> None:
    with connect(engine) as connection:
        result = connection.execute(text("SELECT 1"))
        assert result.scalar() == 1
