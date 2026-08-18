"""Assemble unified training frames from Parquet and PostgreSQL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.models import Transaction, TransactionIdentity
from fraud_scoring_engine.features import (
    BEHAVIORAL_FEATURE_COLUMNS,
    DEFAULT_BEHAVIORAL_FEATURES,
    compute_transaction_features_dataframe,
)
from fraud_scoring_engine.ingest.paths import ieee_data_paths

# Operational + identity columns stored in Postgres and used as model features.
POSTGRES_MODEL_COLUMNS: tuple[str, ...] = (
    "transaction_amt",
    "product_cd",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "p_emaildomain",
    "r_emaildomain",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "id_30",
    "id_31",
    "device_type",
    "device_info",
)

# ``transaction_dt`` is loaded for time-based splits, not as a model feature.
POSTGRES_TRAINING_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "is_fraud",
    "transaction_dt",
    *POSTGRES_MODEL_COLUMNS,
)

_BEHAVIORAL_DEFAULTS = asdict(DEFAULT_BEHAVIORAL_FEATURES)


def load_transaction_model_columns(
    session: Session,
    *,
    transaction_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load Postgres modeling columns for transactions (with identity join).

    Args:
        session: Active SQLAlchemy ORM session.
        transaction_ids: Optional subset of ``transaction_id`` values to load.

    Returns:
        DataFrame keyed by ``transaction_id`` with ``is_fraud``,
        ``transaction_dt`` (for time-based splits), and
        :data:`POSTGRES_MODEL_COLUMNS`.
    """
    stmt = (
        select(
            Transaction.transaction_id,
            Transaction.is_fraud,
            Transaction.transaction_dt,
            Transaction.transaction_amt,
            Transaction.product_cd,
            Transaction.card1,
            Transaction.card2,
            Transaction.card3,
            Transaction.card4,
            Transaction.card5,
            Transaction.card6,
            Transaction.p_emaildomain,
            Transaction.r_emaildomain,
            Transaction.addr1,
            Transaction.addr2,
            Transaction.dist1,
            Transaction.dist2,
            TransactionIdentity.id_30,
            TransactionIdentity.id_31,
            TransactionIdentity.device_type,
            TransactionIdentity.device_info,
        )
        .outerjoin(
            TransactionIdentity,
            Transaction.transaction_id == TransactionIdentity.transaction_id,
        )
        .order_by(Transaction.transaction_at, Transaction.transaction_id)
    )
    if transaction_ids is not None:
        ids = list({int(transaction_id) for transaction_id in transaction_ids})
        if not ids:
            return pd.DataFrame(columns=list(POSTGRES_TRAINING_COLUMNS))
        stmt = stmt.where(Transaction.transaction_id.in_(ids))

    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=list(POSTGRES_TRAINING_COLUMNS))
    return pd.DataFrame(rows)


def load_static_features(
    path: Path,
    *,
    transaction_ids: Iterable[int] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load static IEEE feature columns exported to Parquet.

    Args:
        path: Path to ``train_features.parquet``.
        transaction_ids: Optional subset of rows to retain after loading.
        limit: Optional cap on rows, applied in Parquet file order before filtering.

    Returns:
        Static feature DataFrame keyed by ``transaction_id``.
    """
    static = pd.read_parquet(path)
    if "TransactionID" in static.columns:
        static = static.rename(columns={"TransactionID": "transaction_id"})

    if limit is not None:
        static = static.head(limit)

    if transaction_ids is not None:
        ids = {int(transaction_id) for transaction_id in transaction_ids}
        static = static.loc[static["transaction_id"].isin(ids)].copy()

    return static


def _merge_behavioral_features(
    frame: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join behavioral feature columns and fill missing values."""
    if behavioral.empty:
        behavioral_features = pd.DataFrame(
            columns=["transaction_id", *BEHAVIORAL_FEATURE_COLUMNS],
        )
    else:
        behavioral_features = behavioral[["transaction_id", *BEHAVIORAL_FEATURE_COLUMNS]]

    merged = frame.merge(behavioral_features, on="transaction_id", how="left")
    for column, default in _BEHAVIORAL_DEFAULTS.items():
        if default is None:
            continue
        merged[column] = merged[column].fillna(default)
    return merged


def build_training_frame(
    session: Session,
    *,
    limit: int | None = 10_000,
    train_features_path: Path | None = None,
) -> pd.DataFrame:
    """Build a unified training DataFrame from Parquet and PostgreSQL.

    Static IEEE columns come from ``train_features.parquet``. Operational and
    identity columns are read from PostgreSQL. Rolling behavioral features are
    computed on the fly and left-joined so every matched transaction is kept.

    Row scope follows the Parquet slice: when ``limit`` is set, the first
    ``limit`` rows from the Parquet file define ``transaction_id`` values.
    Behavioral features are computed over all ingested transactions, then
    filtered to that ID set so rolling windows stay correct.

    Args:
        session: Active SQLAlchemy ORM session.
        limit: Maximum rows to include, based on Parquet file order. ``None``
            uses every row in the Parquet file.
        train_features_path: Optional override for ``train_features.parquet``.

    Returns:
        Merged training frame with ``transaction_id``, ``is_fraud``, Postgres
        columns, behavioral features, and static IEEE columns.

    Raises:
        FileNotFoundError: If the Parquet feature file does not exist.
        ValueError: If the Parquet slice is empty or Parquet/Postgres IDs do not match.
    """
    features_path = (
        train_features_path
        if train_features_path is not None
        else ieee_data_paths().train_features_parquet
    )
    if not features_path.exists():
        msg = f"Train features Parquet file not found: {features_path}"
        raise FileNotFoundError(msg)

    static = load_static_features(features_path, limit=limit)
    if static.empty:
        msg = (
            f"No training rows found in {features_path} (limit={limit!r}). "
            "Run ingest to export train_features.parquet."
        )
        raise ValueError(msg)

    transaction_ids = static["transaction_id"].astype(int).tolist()
    postgres = load_transaction_model_columns(session, transaction_ids=transaction_ids)
    behavioral = compute_transaction_features_dataframe(
        session,
        limit=None,
        transaction_ids=transaction_ids,
    )

    merged = static.merge(postgres, on="transaction_id", how="inner")
    if merged.empty:
        msg = (
            f"No rows matched between {features_path} and PostgreSQL "
            f"for {len(transaction_ids)} Parquet transaction_id(s). "
            "Ensure ingest loaded the same transaction slice into the database."
        )
        raise ValueError(msg)

    return _merge_behavioral_features(merged, behavioral)
