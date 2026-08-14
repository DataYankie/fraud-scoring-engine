"""SQLAlchemy queries for per-transaction fraud scoring features."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.models import Transaction


def _prior_transaction_filters(
    *,
    derived_user_id: str,
    transaction_at: datetime,
    window: timedelta,
    transaction_id: int | None = None,
) -> list:
    """Build SQLAlchemy filters for prior transactions within a rolling window.

    The current transaction is excluded. When ``transaction_id`` is provided,
    same-timestamp rows with a lower ID are treated as prior transactions.

    Args:
        derived_user_id: Synthetic user identifier to match.
        transaction_at: Anchor timestamp for the window.
        window: Lookback duration before ``transaction_at``.
        transaction_id: Optional current transaction ID for tie-breaking.

    Returns:
        List of SQLAlchemy filter expressions.
    """
    window_start = transaction_at - window
    filters = [
        Transaction.derived_user_id == derived_user_id,
        Transaction.transaction_at >= window_start,
    ]
    if transaction_id is not None:
        filters.append(
            or_(
                Transaction.transaction_at < transaction_at,
                and_(
                    Transaction.transaction_at == transaction_at,
                    Transaction.transaction_id < transaction_id,
                ),
            )
        )
    else:
        filters.append(Transaction.transaction_at < transaction_at)
    return filters


def compute_velocity(
    session: Session,
    *,
    derived_user_id: str | None,
    transaction_at: datetime,
    window_hours: int,
    transaction_id: int | None = None,
) -> int:
    """Count prior transactions by a user within a rolling hour window.

    Args:
        session: Active SQLAlchemy ORM session.
        derived_user_id: Synthetic user identifier; returns ``0`` when ``None``.
        transaction_at: Anchor timestamp for the window.
        window_hours: Lookback duration in hours.
        transaction_id: Optional current transaction ID for tie-breaking.

    Returns:
        Number of qualifying prior transactions.
    """
    if derived_user_id is None:
        return 0

    stmt = (
        select(func.count())
        .select_from(Transaction)
        .where(
            *_prior_transaction_filters(
                derived_user_id=derived_user_id,
                transaction_at=transaction_at,
                window=timedelta(hours=window_hours),
                transaction_id=transaction_id,
            )
        )
    )
    return int(session.scalar(stmt) or 0)


def compute_cumulative_spend(
    session: Session,
    *,
    derived_user_id: str | None,
    transaction_at: datetime,
    window_hours: int,
    transaction_id: int | None = None,
) -> float:
    """Sum prior transaction amounts by a user within a rolling hour window.

    Args:
        session: Active SQLAlchemy ORM session.
        derived_user_id: Synthetic user identifier; returns ``0.0`` when ``None``.
        transaction_at: Anchor timestamp for the window.
        window_hours: Lookback duration in hours.
        transaction_id: Optional current transaction ID for tie-breaking.

    Returns:
        Total spend in dollars for qualifying prior transactions.
    """
    if derived_user_id is None:
        return 0.0

    stmt = select(
        func.coalesce(func.sum(Transaction.transaction_amt), 0)
    ).where(
        *_prior_transaction_filters(
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            window=timedelta(hours=window_hours),
            transaction_id=transaction_id,
        )
    )
    total = session.scalar(stmt) or 0
    if isinstance(total, Decimal):
        return float(total)
    return float(total)


def compute_avg_amount_ratio(
    session: Session,
    *,
    derived_user_id: str | None,
    transaction_at: datetime,
    transaction_amt: float,
    window_days: int,
    transaction_id: int | None = None,
) -> float | None:
    """Compute the ratio of current amount to prior average amount.

    Args:
        session: Active SQLAlchemy ORM session.
        derived_user_id: Synthetic user identifier; returns ``None`` when ``None``.
        transaction_at: Anchor timestamp for the window.
        transaction_amt: Current transaction amount (numerator).
        window_days: Lookback duration in days.
        transaction_id: Optional current transaction ID for tie-breaking.

    Returns:
        ``transaction_amt / prior_average``, or ``None`` when no prior
        transactions exist or the prior average is zero.
    """
    if derived_user_id is None:
        return None

    stmt = select(func.avg(Transaction.transaction_amt)).where(
        *_prior_transaction_filters(
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            window=timedelta(days=window_days),
            transaction_id=transaction_id,
        )
    )
    avg_amount = session.scalar(stmt)
    if avg_amount is None:
        return None

    avg_value = float(avg_amount) if isinstance(avg_amount, Decimal) else float(avg_amount)
    if avg_value == 0:
        return None
    return transaction_amt / avg_value


@dataclass(frozen=True)
class TransactionFeatures:
    """Aggregated fraud scoring features for a single transaction.

    Attributes:
        velocity_1h: Prior transaction count in the last hour.
        cumulative_spend_24h: Prior total spend in the last 24 hours.
        avg_amount_ratio_30d: Current amount divided by the prior 30-day average.
        avg_amount_ratio_90d: Current amount divided by the prior 90-day average.
    """

    velocity_1h: int
    cumulative_spend_24h: float
    avg_amount_ratio_30d: float | None
    avg_amount_ratio_90d: float | None


BEHAVIORAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "velocity_1h",
    "cumulative_spend_24h",
    "avg_amount_ratio_30d",
    "avg_amount_ratio_90d",
)


@dataclass(frozen=True)
class _TxnRow:
    """Minimal transaction fields used for bulk feature computation."""

    transaction_id: int
    derived_user_id: str | None
    transaction_at: datetime
    transaction_amt: float
    is_fraud: int | None


_NULL_FEATURES = TransactionFeatures(
    velocity_1h=0,
    cumulative_spend_24h=0.0,
    avg_amount_ratio_30d=None,
    avg_amount_ratio_90d=None,
)

DEFAULT_BEHAVIORAL_FEATURES = _NULL_FEATURES


def _compute_bulk_user_group(rows: list[_TxnRow]) -> dict[int, TransactionFeatures]:
    """Compute features for a single user's chronologically sorted transactions."""
    n = len(rows)
    result: dict[int, TransactionFeatures] = {}
    left_1h = 0
    left_24h = 0
    left_30d = 0
    left_90d = 0
    sum_24h = 0.0
    sum_30d = 0.0
    sum_90d = 0.0

    for i in range(n):
        at = rows[i].transaction_at
        amt = float(rows[i].transaction_amt)

        while left_1h < i and rows[left_1h].transaction_at < at - timedelta(hours=1):
            left_1h += 1

        while left_24h < i and rows[left_24h].transaction_at < at - timedelta(hours=24):
            sum_24h -= float(rows[left_24h].transaction_amt)
            left_24h += 1

        while left_30d < i and rows[left_30d].transaction_at < at - timedelta(days=30):
            sum_30d -= float(rows[left_30d].transaction_amt)
            left_30d += 1

        while left_90d < i and rows[left_90d].transaction_at < at - timedelta(days=90):
            sum_90d -= float(rows[left_90d].transaction_amt)
            left_90d += 1

        count_30d = i - left_30d
        avg_30d = (sum_30d / count_30d) if count_30d > 0 else None
        ratio_30d = (amt / avg_30d) if avg_30d else None

        count_90d = i - left_90d
        avg_90d = (sum_90d / count_90d) if count_90d > 0 else None
        ratio_90d = (amt / avg_90d) if avg_90d else None

        result[rows[i].transaction_id] = TransactionFeatures(
            velocity_1h=i - left_1h,
            cumulative_spend_24h=sum_24h,
            avg_amount_ratio_30d=ratio_30d,
            avg_amount_ratio_90d=ratio_90d,
        )

        sum_24h += amt
        sum_30d += amt
        sum_90d += amt

    return result


def _fetch_all_transaction_rows(session: Session) -> list[_TxnRow]:
    """Load all transactions in chronological order with a single query."""
    stmt = select(Transaction).order_by(
        Transaction.transaction_at,
        Transaction.transaction_id,
    )
    return [
        _TxnRow(
            transaction_id=txn.transaction_id,
            derived_user_id=txn.derived_user_id,
            transaction_at=txn.transaction_at,
            transaction_amt=float(txn.transaction_amt),
            is_fraud=txn.is_fraud,
        )
        for txn in session.scalars(stmt)
    ]


def _compute_features_by_transaction_id(rows: list[_TxnRow]) -> dict[int, TransactionFeatures]:
    """Compute rolling features for all rows grouped by ``derived_user_id``."""
    groups: dict[str, list[_TxnRow]] = defaultdict(list)
    feature_by_id: dict[int, TransactionFeatures] = {}

    for row in rows:
        if row.derived_user_id is None:
            feature_by_id[row.transaction_id] = _NULL_FEATURES
        else:
            groups[row.derived_user_id].append(row)

    for user_rows in groups.values():
        user_rows.sort(key=lambda row: (row.transaction_at, row.transaction_id))
        feature_by_id.update(_compute_bulk_user_group(user_rows))

    return feature_by_id


def _features_to_record(row: _TxnRow, features: TransactionFeatures) -> dict[str, object]:
    """Flatten a transaction row and its features into a training record."""
    record = asdict(features)
    record["transaction_id"] = row.transaction_id
    record["is_fraud"] = row.is_fraud
    return record


def compute_transaction_features_dataframe(
    session: Session,
    *,
    limit: int | None = 10_000,
    transaction_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling behavioral features for many transactions efficiently.

    Loads all transactions with one query, computes features in memory using
    per-user sliding windows (linear time), then returns rows filtered by
    ``transaction_ids`` or truncated to ``limit`` in chronological order.

    Args:
        session: Active SQLAlchemy ORM session.
        limit: Maximum number of rows to return, ordered by ``transaction_at``
            then ``transaction_id``. Ignored when ``transaction_ids`` is set.
            ``None`` returns all rows unless ``transaction_ids`` is set.
        transaction_ids: Optional explicit ``transaction_id`` values to export,
            preserving this order in the result.

    Returns:
        DataFrame with ``transaction_id``, ``is_fraud``, and feature columns.
    """
    rows = _fetch_all_transaction_rows(session)
    columns = [
        "transaction_id",
        "is_fraud",
        "velocity_1h",
        "cumulative_spend_24h",
        "avg_amount_ratio_30d",
        "avg_amount_ratio_90d",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    feature_by_id = _compute_features_by_transaction_id(rows)
    if transaction_ids is not None:
        row_by_id = {row.transaction_id: row for row in rows}
        export_rows = [
            row_by_id[int(transaction_id)]
            for transaction_id in transaction_ids
            if int(transaction_id) in row_by_id
        ]
    elif limit is None:
        export_rows = rows
    else:
        export_rows = rows[:limit]

    records = [
        _features_to_record(row, feature_by_id[row.transaction_id])
        for row in export_rows
    ]
    return pd.DataFrame.from_records(records)


def compute_transaction_features(
    session: Session,
    *,
    derived_user_id: str | None,
    transaction_at: datetime,
    transaction_amt: float,
    transaction_id: int | None = None,
) -> TransactionFeatures:
    """Compute all configured fraud scoring features for a transaction.

    Uses fixed windows: 1-hour velocity, 24-hour spend, and 30/90-day
    amount ratios.

    Args:
        session: Active SQLAlchemy ORM session.
        derived_user_id: Synthetic user identifier.
        transaction_at: Timestamp of the transaction being scored.
        transaction_amt: Amount of the transaction being scored.
        transaction_id: Optional current transaction ID for tie-breaking.

    Returns:
        Populated :class:`TransactionFeatures` instance.
    """
    return TransactionFeatures(
        velocity_1h=compute_velocity(
            session,
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            window_hours=1,
            transaction_id=transaction_id,
        ),
        cumulative_spend_24h=compute_cumulative_spend(
            session,
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            window_hours=24,
            transaction_id=transaction_id,
        ),
        avg_amount_ratio_30d=compute_avg_amount_ratio(
            session,
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            transaction_amt=transaction_amt,
            window_days=30,
            transaction_id=transaction_id,
        ),
        avg_amount_ratio_90d=compute_avg_amount_ratio(
            session,
            derived_user_id=derived_user_id,
            transaction_at=transaction_at,
            transaction_amt=transaction_amt,
            window_days=90,
            transaction_id=transaction_id,
        ),
    )


def compute_transaction_features_from_model(
    session: Session,
    transaction: Transaction,
) -> TransactionFeatures:
    """Compute all configured fraud scoring features from an ORM instance.

    Args:
        session: Active SQLAlchemy ORM session.
        transaction: Persisted transaction row to score.

    Returns:
        Populated :class:`TransactionFeatures` instance.
    """
    return compute_transaction_features(
        session,
        derived_user_id=transaction.derived_user_id,
        transaction_at=transaction.transaction_at,
        transaction_amt=float(transaction.transaction_amt),
        transaction_id=transaction.transaction_id,
    )
