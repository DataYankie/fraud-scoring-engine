"""Tests for fraud_scoring_engine.features."""

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.models import Base, Transaction
from fraud_scoring_engine.features import (
    compute_avg_amount_ratio,
    compute_cumulative_spend,
    compute_transaction_features,
    compute_transaction_features_dataframe,
    compute_transaction_features_from_model,
    compute_velocity,
)

USER_A = "a" * 32
USER_B = "b" * 32
BASE_TIME = datetime(2017, 12, 1, 12, 0, 0)


@pytest.fixture
def db_engine(engine: Engine) -> Engine:
    Base.metadata.create_all(engine)
    return engine


def _insert_transaction(
    session: Session,
    *,
    transaction_id: int,
    derived_user_id: str | None,
    transaction_at: datetime,
    transaction_amt: float,
) -> Transaction:
    transaction = Transaction(
        transaction_id=transaction_id,
        derived_user_id=derived_user_id,
        is_fraud=0,
        transaction_amt=transaction_amt,
        product_cd="W",
        transaction_dt=0,
        transaction_at=transaction_at,
    )
    session.add(transaction)
    session.commit()
    return transaction


@pytest.fixture
def seeded_session(db_engine: Engine) -> Iterator[Session]:
    """Seed transactions for USER_A with controlled times and amounts.

    Timeline (all on 2017-12-01 unless noted):
    - id=1: BASE_TIME - 45d, $10  (outside 30d/90d ratio windows for late txns)
    - id=2: BASE_TIME - 20m, $20
    - id=3: BASE_TIME - 10m, $30
    - id=4: BASE_TIME, $2000 (spike txn under test)
    - id=5: BASE_TIME, $40 (same timestamp as id=4, lower id excluded first)
    """
    with Session(db_engine) as session:
        _insert_transaction(
            session,
            transaction_id=1,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME - timedelta(days=45),
            transaction_amt=10.0,
        )
        _insert_transaction(
            session,
            transaction_id=2,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME - timedelta(minutes=20),
            transaction_amt=20.0,
        )
        _insert_transaction(
            session,
            transaction_id=3,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME - timedelta(minutes=10),
            transaction_amt=30.0,
        )
        _insert_transaction(
            session,
            transaction_id=4,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            transaction_amt=2000.0,
        )
        _insert_transaction(
            session,
            transaction_id=5,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            transaction_amt=40.0,
        )
        yield session


def test_first_transaction_has_zero_velocity_and_spend(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _insert_transaction(
            session,
            transaction_id=100,
            derived_user_id=USER_B,
            transaction_at=BASE_TIME,
            transaction_amt=50.0,
        )

        assert (
            compute_velocity(
                session,
                derived_user_id=USER_B,
                transaction_at=BASE_TIME,
                window_hours=1,
                transaction_id=100,
            )
            == 0
        )
        assert (
            compute_cumulative_spend(
                session,
                derived_user_id=USER_B,
                transaction_at=BASE_TIME,
                window_hours=24,
                transaction_id=100,
            )
            == 0.0
        )
        assert (
            compute_avg_amount_ratio(
                session,
                derived_user_id=USER_B,
                transaction_at=BASE_TIME,
                transaction_amt=50.0,
                window_days=30,
                transaction_id=100,
            )
            is None
        )
        assert (
            compute_avg_amount_ratio(
                session,
                derived_user_id=USER_B,
                transaction_at=BASE_TIME,
                transaction_amt=50.0,
                window_days=90,
                transaction_id=100,
            )
            is None
        )


def test_velocity_counts_prior_transactions_in_last_hour(seeded_session: Session) -> None:
    assert (
        compute_velocity(
            seeded_session,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            window_hours=1,
            transaction_id=4,
        )
        == 2
    )


def test_cumulative_spend_sums_prior_24h_only(seeded_session: Session) -> None:
    assert (
        compute_cumulative_spend(
            seeded_session,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            window_hours=24,
            transaction_id=4,
        )
        == 50.0
    )


def test_avg_amount_ratio_spikes_for_large_current_amount(seeded_session: Session) -> None:
    ratio_30d = compute_avg_amount_ratio(
        seeded_session,
        derived_user_id=USER_A,
        transaction_at=BASE_TIME,
        transaction_amt=2000.0,
        window_days=30,
        transaction_id=4,
    )
    ratio_90d = compute_avg_amount_ratio(
        seeded_session,
        derived_user_id=USER_A,
        transaction_at=BASE_TIME,
        transaction_amt=2000.0,
        window_days=90,
        transaction_id=4,
    )

    assert ratio_30d == pytest.approx(2000.0 / 25.0)
    assert ratio_90d == pytest.approx(2000.0 / 20.0)


def test_null_derived_user_id_returns_safe_defaults(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _insert_transaction(
            session,
            transaction_id=200,
            derived_user_id=None,
            transaction_at=BASE_TIME,
            transaction_amt=100.0,
        )

        assert (
            compute_velocity(
                session,
                derived_user_id=None,
                transaction_at=BASE_TIME,
                window_hours=1,
                transaction_id=200,
            )
            == 0
        )
        assert (
            compute_cumulative_spend(
                session,
                derived_user_id=None,
                transaction_at=BASE_TIME,
                window_hours=24,
                transaction_id=200,
            )
            == 0.0
        )
        assert (
            compute_avg_amount_ratio(
                session,
                derived_user_id=None,
                transaction_at=BASE_TIME,
                transaction_amt=100.0,
                window_days=30,
                transaction_id=200,
            )
            is None
        )


def test_same_timestamp_excludes_current_transaction_id(seeded_session: Session) -> None:
    assert (
        compute_velocity(
            seeded_session,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            window_hours=1,
            transaction_id=5,
        )
        == 3
    )
    assert (
        compute_cumulative_spend(
            seeded_session,
            derived_user_id=USER_A,
            transaction_at=BASE_TIME,
            window_hours=24,
            transaction_id=5,
        )
        == 2050.0
    )


def test_compute_transaction_features_aggregator(seeded_session: Session) -> None:
    features = compute_transaction_features(
        seeded_session,
        derived_user_id=USER_A,
        transaction_at=BASE_TIME,
        transaction_amt=2000.0,
        transaction_id=4,
    )

    assert features.velocity_1h == 2
    assert features.cumulative_spend_24h == 50.0
    assert features.avg_amount_ratio_30d == pytest.approx(2000.0 / 25.0)
    assert features.avg_amount_ratio_90d == pytest.approx(2000.0 / 20.0)


def test_compute_transaction_features_from_model(seeded_session: Session) -> None:
    transaction = seeded_session.get(Transaction, 4)
    assert transaction is not None

    features = compute_transaction_features_from_model(seeded_session, transaction)

    assert features.velocity_1h == 2
    assert features.cumulative_spend_24h == 50.0
    assert features.avg_amount_ratio_30d == pytest.approx(2000.0 / 25.0)
    assert features.avg_amount_ratio_90d == pytest.approx(2000.0 / 20.0)


def test_compute_transaction_features_dataframe_matches_row_by_row(
    seeded_session: Session,
) -> None:
    dataframe = compute_transaction_features_dataframe(seeded_session, limit=None)

    assert len(dataframe) == 5

    for _, row in dataframe.iterrows():
        transaction = seeded_session.get(Transaction, int(row["transaction_id"]))
        assert transaction is not None
        expected = compute_transaction_features_from_model(seeded_session, transaction)
        assert row["velocity_1h"] == expected.velocity_1h
        assert row["cumulative_spend_24h"] == expected.cumulative_spend_24h
        if expected.avg_amount_ratio_30d is None:
            assert row["avg_amount_ratio_30d"] is None or pd.isna(row["avg_amount_ratio_30d"])
        else:
            assert row["avg_amount_ratio_30d"] == pytest.approx(expected.avg_amount_ratio_30d)
        if expected.avg_amount_ratio_90d is None:
            assert row["avg_amount_ratio_90d"] is None or pd.isna(row["avg_amount_ratio_90d"])
        else:
            assert row["avg_amount_ratio_90d"] == pytest.approx(expected.avg_amount_ratio_90d)


def test_compute_transaction_features_dataframe_respects_limit(seeded_session: Session) -> None:
    dataframe = compute_transaction_features_dataframe(seeded_session, limit=2)

    assert len(dataframe) == 2
    assert list(dataframe["transaction_id"]) == [1, 2]
