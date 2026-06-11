"""Transform IEEE-CIS rows into SQLAlchemy ORM models."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from numbers import Real

import pandas as pd

from fraud_scoring_engine.db.models import Transaction, TransactionIdentity

IEEE_EPOCH = datetime(2017, 11, 30)

USER_ID_COMPONENTS = (
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
)

IDENTITY_COLUMNS = ("id_30", "id_31", "DeviceType", "DeviceInfo")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA


def _component(value: object) -> str:
    """Convert a CSV cell to a stable hash component string."""
    if _is_missing(value):
        return ""
    return str(value)


def generate_user_id(row: pd.Series) -> str:
    """Derive a synthetic user ID from card and address features.

    Args:
        row: Merged IEEE row (Series) containing card and addr columns.

    Returns:
        32-character MD5 hex digest.
    """
    components = [_component(row.get(col)) for col in USER_ID_COMPONENTS]
    uid_string = "_".join(components)
    return hashlib.md5(uid_string.encode("utf-8")).hexdigest()


def derive_transaction_at(dt_seconds: int) -> datetime:
    """Convert IEEE ``TransactionDT`` seconds to a wall-clock datetime.

    Args:
        dt_seconds: Seconds elapsed since 2017-11-30.

    Returns:
        Derived ``transaction_at`` value.
    """
    return IEEE_EPOCH + timedelta(seconds=int(dt_seconds))


def _nullable_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, Real):
        return float(value)
    return float(str(value))


def _nullable_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return int(float(value))
    return int(str(value))


def _nullable_str(value: object, *, max_len: int | None = None) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    if max_len is not None:
        return text[:max_len]
    return text


def _has_identity_data(row: pd.Series) -> bool:
    return any(not _is_missing(row.get(col)) for col in IDENTITY_COLUMNS)


def merged_row_to_models(row: pd.Series) -> tuple[Transaction, TransactionIdentity | None]:
    """Map one merged IEEE row to ORM models.

    Args:
        row: Merged transaction + identity row.

    Returns:
        ``(Transaction, TransactionIdentity | None)`` tuple.
    """
    transaction_id = int(row.at["TransactionID"])
    transaction_dt = int(row.at["TransactionDT"])
    transaction = Transaction(
        transaction_id=transaction_id,
        derived_user_id=generate_user_id(row),
        is_fraud=_nullable_int(row.get("isFraud")),
        transaction_amt=float(row.at["TransactionAmt"]),
        product_cd=_nullable_str(row.get("ProductCD"), max_len=10),
        transaction_dt=transaction_dt,
        transaction_at=derive_transaction_at(transaction_dt),
        card1=_nullable_float(row.get("card1")),
        card2=_nullable_float(row.get("card2")),
        card3=_nullable_float(row.get("card3")),
        card4=_nullable_str(row.get("card4"), max_len=50),
        card5=_nullable_float(row.get("card5")),
        card6=_nullable_str(row.get("card6"), max_len=50),
        p_emaildomain=_nullable_str(row.get("P_emaildomain"), max_len=100),
        r_emaildomain=_nullable_str(row.get("R_emaildomain"), max_len=100),
        addr1=_nullable_float(row.get("addr1")),
        addr2=_nullable_float(row.get("addr2")),
        dist1=_nullable_float(row.get("dist1")),
        dist2=_nullable_float(row.get("dist2")),
    )

    identity: TransactionIdentity | None = None
    if _has_identity_data(row):
        identity = TransactionIdentity(
            transaction_id=transaction_id,
            id_30=_nullable_str(row.get("id_30"), max_len=100),
            id_31=_nullable_str(row.get("id_31"), max_len=100),
            device_type=_nullable_str(row.get("DeviceType"), max_len=50),
            device_info=_nullable_str(row.get("DeviceInfo"), max_len=100),
        )

    return transaction, identity
