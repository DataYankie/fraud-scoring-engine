"""SQLAlchemy engine and connection helpers."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.pool import Pool

from fraud_scoring_engine.config import get_database_url


def create_db_engine(
    *,
    poolclass: type[Pool] | None = None,
    echo: bool = False,
    **kwargs: Any,
) -> Engine:
    """Create a SQLAlchemy engine for the application database.

    Uses :func:`fraud_scoring_engine.config.get_database_url` for the connection
    URL. Pass ``poolclass=NullPool`` for short-lived workloads such as Alembic
    migrations.

    Args:
        poolclass: Optional pool implementation (e.g. :class:`~sqlalchemy.pool.NullPool`).
        echo: If True, log all SQL statements.
        **kwargs: Additional keyword arguments forwarded to :func:`sqlalchemy.create_engine`.

    Returns:
        A configured SQLAlchemy engine.
    """
    engine_kwargs: dict[str, Any] = {"echo": echo, **kwargs}
    if poolclass is not None:
        engine_kwargs["poolclass"] = poolclass
    return create_engine(get_database_url(), **engine_kwargs)


@contextmanager
def connect(engine: Engine) -> Generator[Connection, None, None]:
    """Open a database connection from an engine.

    Args:
        engine: The engine to connect with.

    Yields:
        An active SQLAlchemy connection, closed when the context exits.
    """
    with engine.connect() as connection:
        yield connection
