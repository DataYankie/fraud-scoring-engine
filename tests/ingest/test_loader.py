"""Tests for fraud_scoring_engine.ingest.loader."""

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.models import Base, Transaction, TransactionIdentity
from fraud_scoring_engine.ingest.loader import _ingest_to_postgres
from fraud_scoring_engine.ingest.transforms import generate_user_id_from_components


@pytest.fixture
def db_engine(engine: Engine) -> Engine:
    Base.metadata.create_all(engine)
    return engine


def _sample_merged() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TransactionID": 2987000,
                "isFraud": 0,
                "TransactionAmt": 68.5,
                "TransactionDT": 86400,
                "ProductCD": "W",
                "card1": 13926.0,
                "card2": None,
                "card3": 150.0,
                "card4": "discover",
                "card5": 142.0,
                "card6": "credit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": None,
                "addr1": 315.0,
                "addr2": 87.0,
                "dist1": 19.0,
                "dist2": None,
                "id_30": "Windows 10",
                "id_31": "chrome 63.0",
                "DeviceType": "desktop",
                "DeviceInfo": "Windows",
            },
            {
                "TransactionID": 2987001,
                "isFraud": 1,
                "TransactionAmt": 29.0,
                "TransactionDT": 86401,
                "ProductCD": "W",
                "card1": 2755.0,
                "card2": 404.0,
                "card3": 150.0,
                "card4": "mastercard",
                "card5": 102.0,
                "card6": "credit",
                "P_emaildomain": None,
                "R_emaildomain": None,
                "addr1": 325.0,
                "addr2": 87.0,
                "dist1": None,
                "dist2": None,
                "id_30": None,
                "id_31": None,
                "DeviceType": None,
                "DeviceInfo": None,
            },
        ]
    )


def test_ingest_to_postgres_inserts_transactions_and_identities(
    db_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fraud_scoring_engine.ingest.loader.create_db_engine",
        lambda: db_engine,
    )
    merged = _sample_merged()

    rows_inserted, rows_skipped, identities_inserted = _ingest_to_postgres(
        merged,
        batch_size=5000,
    )

    assert rows_inserted == 2
    assert rows_skipped == 0
    assert identities_inserted == 1

    with Session(db_engine) as session:
        transaction = session.get(Transaction, 2987000)
        assert transaction is not None
        assert transaction.is_fraud == 0
        assert transaction.transaction_at == datetime(2017, 12, 1)
        assert transaction.derived_user_id == generate_user_id_from_components(
            {
                "card1": 13926.0,
                "card2": None,
                "card3": 150.0,
                "card4": "discover",
                "card5": 142.0,
                "card6": "credit",
                "addr1": 315.0,
                "addr2": 87.0,
            }
        )

        identity = session.get(TransactionIdentity, 2987000)
        assert identity is not None
        assert identity.device_type == "desktop"

        missing_identity = session.get(TransactionIdentity, 2987001)
        assert missing_identity is None


def test_ingest_to_postgres_skips_existing_transactions(
    db_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fraud_scoring_engine.ingest.loader.create_db_engine",
        lambda: db_engine,
    )
    merged = _sample_merged()

    first_inserted, first_skipped, first_identities = _ingest_to_postgres(
        merged,
        batch_size=5000,
    )
    second_inserted, second_skipped, second_identities = _ingest_to_postgres(
        merged,
        batch_size=5000,
    )

    assert first_inserted == 2
    assert first_skipped == 0
    assert first_identities == 1
    assert second_inserted == 0
    assert second_skipped == 2
    assert second_identities == 0

    with Session(db_engine) as session:
        count = session.scalar(select(func.count()).select_from(Transaction))
        assert count == 2
