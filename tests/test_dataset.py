"""Tests for fraud_scoring_engine.training.dataset."""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.models import Base, Transaction, TransactionIdentity
from fraud_scoring_engine.training.dataset import (
    POSTGRES_MODEL_COLUMNS,
    POSTGRES_TRAINING_COLUMNS,
    build_training_frame,
    load_static_features,
    load_transaction_model_columns,
)

BASE_TIME = datetime(2017, 12, 1, 12, 0, 0)


@pytest.fixture
def db_engine(engine: Engine) -> Engine:
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def seeded_session(db_engine: Engine) -> Iterator[Session]:
    with Session(db_engine) as session:
        session.add(
            Transaction(
                transaction_id=101,
                derived_user_id="a" * 32,
                is_fraud=0,
                transaction_amt=50.0,
                product_cd="W",
                transaction_dt=0,
                transaction_at=BASE_TIME,
                card1=100.0,
                card4="visa",
            )
        )
        session.add(
            Transaction(
                transaction_id=102,
                derived_user_id="b" * 32,
                is_fraud=1,
                transaction_amt=75.0,
                product_cd="C",
                transaction_dt=60,
                transaction_at=BASE_TIME.replace(minute=1),
                card1=200.0,
                card4="mastercard",
            )
        )
        session.add(
            TransactionIdentity(
                transaction_id=101,
                id_30="Windows 10",
                device_type="desktop",
            )
        )
        session.commit()
        yield session


def test_load_transaction_model_columns_returns_postgres_features(
    seeded_session: Session,
) -> None:
    dataframe = load_transaction_model_columns(
        seeded_session,
        transaction_ids=[101, 102],
    )

    assert list(dataframe.columns) == list(POSTGRES_TRAINING_COLUMNS)
    assert len(dataframe) == 2
    assert set(dataframe["transaction_id"]) == {101, 102}
    assert dataframe.loc[dataframe["transaction_id"] == 101, "card1"].iloc[0] == 100.0
    assert dataframe.loc[dataframe["transaction_id"] == 101, "id_30"].iloc[0] == "Windows 10"
    assert pd.isna(dataframe.loc[dataframe["transaction_id"] == 102, "id_30"].iloc[0])


def test_load_static_features_filters_by_transaction_ids(tmp_path: Path) -> None:
    static = pd.DataFrame(
        {
            "transaction_id": [101, 102, 103],
            "C1": [1.0, 2.0, 3.0],
            "D1": [4.0, 5.0, 6.0],
        }
    )
    path = tmp_path / "train_features.parquet"
    static.to_parquet(path, index=False)

    loaded = load_static_features(path, transaction_ids=[102, 101])

    assert list(loaded["transaction_id"]) == [101, 102]
    assert list(loaded.columns) == ["transaction_id", "C1", "D1"]


def test_build_training_frame_merges_all_sources(
    seeded_session: Session,
    tmp_path: Path,
) -> None:
    static = pd.DataFrame(
        {
            "transaction_id": [101, 102],
            "C1": [1.0, 2.0],
            "D1": [10.0, 20.0],
        }
    )
    parquet_path = tmp_path / "train_features.parquet"
    static.to_parquet(parquet_path, index=False)

    dataframe = build_training_frame(
        seeded_session,
        limit=None,
        train_features_path=parquet_path,
    )

    assert len(dataframe) == 2
    assert "card1" in dataframe.columns
    assert "C1" in dataframe.columns
    assert "velocity_1h" in dataframe.columns
    assert "is_fraud" in dataframe.columns
    assert "transaction_dt" in dataframe.columns
    assert set(dataframe["transaction_id"]) == {101, 102}


def test_build_training_frame_respects_limit(
    seeded_session: Session,
    tmp_path: Path,
) -> None:
    static = pd.DataFrame(
        {
            "transaction_id": [101, 102],
            "C1": [1.0, 2.0],
        }
    )
    parquet_path = tmp_path / "train_features.parquet"
    static.to_parquet(parquet_path, index=False)

    dataframe = build_training_frame(
        seeded_session,
        limit=1,
        train_features_path=parquet_path,
    )

    assert len(dataframe) == 1
    assert dataframe["transaction_id"].iloc[0] == 101


def test_compute_transaction_features_dataframe_preserves_transaction_id_order(
    seeded_session: Session,
) -> None:
    from fraud_scoring_engine.features import compute_transaction_features_dataframe

    dataframe = compute_transaction_features_dataframe(
        seeded_session,
        transaction_ids=[102, 101],
    )

    assert list(dataframe["transaction_id"]) == [102, 101]


def test_build_training_frame_raises_when_parquet_is_empty(
    seeded_session: Session,
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "train_features.parquet"
    pd.DataFrame(columns=["transaction_id", "C1"]).to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="No training rows found"):
        build_training_frame(
            seeded_session,
            train_features_path=parquet_path,
        )


def test_build_training_frame_raises_when_postgres_has_no_matching_ids(
    seeded_session: Session,
    tmp_path: Path,
) -> None:
    static = pd.DataFrame({"transaction_id": [999], "C1": [1.0]})
    parquet_path = tmp_path / "train_features.parquet"
    static.to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="No rows matched between .* and PostgreSQL"):
        build_training_frame(
            seeded_session,
            train_features_path=parquet_path,
        )


def test_build_training_frame_keeps_rows_when_behavioral_features_missing(
    seeded_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = pd.DataFrame(
        {
            "transaction_id": [101, 102],
            "C1": [1.0, 2.0],
        }
    )
    parquet_path = tmp_path / "train_features.parquet"
    static.to_parquet(parquet_path, index=False)

    def empty_behavioral(
        session: Session,
        *,
        limit: int | None = 10_000,
        transaction_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        del session, limit, transaction_ids
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "is_fraud",
                "velocity_1h",
                "cumulative_spend_24h",
                "avg_amount_ratio_30d",
                "avg_amount_ratio_90d",
            ]
        )

    monkeypatch.setattr(
        "fraud_scoring_engine.training.dataset.compute_transaction_features_dataframe",
        empty_behavioral,
    )

    dataframe = build_training_frame(
        seeded_session,
        train_features_path=parquet_path,
    )

    assert len(dataframe) == 2
    assert list(dataframe["is_fraud"]) == [0, 1]
    assert list(dataframe["velocity_1h"]) == [0, 0]
    assert pd.isna(dataframe["avg_amount_ratio_30d"]).all()
