"""Transform IEEE-CIS DataFrames for PostgreSQL bulk ingest."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from numbers import Real

import pandas as pd

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

TRANSACTION_DB_COLUMNS = (
    "transaction_id",
    "derived_user_id",
    "is_fraud",
    "transaction_amt",
    "product_cd",
    "transaction_dt",
    "transaction_at",
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
)

IDENTITY_DB_COLUMNS = (
    "transaction_id",
    "id_30",
    "id_31",
    "device_type",
    "device_info",
)


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


def generate_user_id_from_components(components: dict[str, object]) -> str:
    """Derive a synthetic user ID from card and address feature values.

    Args:
        components: Mapping of column name to cell value for USER_ID_COMPONENTS.

    Returns:
        32-character MD5 hex digest.
    """
    parts = [_component(components.get(col)) for col in USER_ID_COMPONENTS]
    uid_string = "_".join(parts)
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


def _nullable_float_series(series: pd.Series) -> pd.Series:
    return series.map(_nullable_float)


def _nullable_int_series(series: pd.Series) -> pd.Series:
    return series.map(_nullable_int)


def _nullable_str_series(series: pd.Series, *, max_len: int | None = None) -> pd.Series:
    if max_len is None:
        return series.map(_nullable_str)
    return series.map(lambda value: _nullable_str(value, max_len=max_len))


def _derive_user_id_series(merged: pd.DataFrame) -> pd.Series:
    components = pd.concat(
        [merged[col].map(_component) for col in USER_ID_COMPONENTS],
        axis=1,
    )
    uid_strings = components.agg("_".join, axis=1)
    return uid_strings.map(lambda value: hashlib.md5(value.encode("utf-8")).hexdigest())


def _has_identity_data_mask(merged: pd.DataFrame) -> pd.Series:
    masks = [merged[col].map(lambda value: not _is_missing(value)) for col in IDENTITY_COLUMNS]
    return pd.concat(masks, axis=1).any(axis=1)


def prepare_transactions_df(merged: pd.DataFrame) -> pd.DataFrame:
    """Return a DB-ready transactions DataFrame with snake_case columns."""
    return pd.DataFrame(
        {
            "transaction_id": merged["TransactionID"].astype(int),
            "derived_user_id": _derive_user_id_series(merged),
            "is_fraud": _nullable_int_series(merged["isFraud"]),
            "transaction_amt": merged["TransactionAmt"].astype(float),
            "product_cd": _nullable_str_series(merged["ProductCD"], max_len=10),
            "transaction_dt": merged["TransactionDT"].astype(int),
            "transaction_at": IEEE_EPOCH + pd.to_timedelta(merged["TransactionDT"], unit="s"),
            "card1": _nullable_float_series(merged["card1"]),
            "card2": _nullable_float_series(merged["card2"]),
            "card3": _nullable_float_series(merged["card3"]),
            "card4": _nullable_str_series(merged["card4"], max_len=50),
            "card5": _nullable_float_series(merged["card5"]),
            "card6": _nullable_str_series(merged["card6"], max_len=50),
            "p_emaildomain": _nullable_str_series(merged["P_emaildomain"], max_len=100),
            "r_emaildomain": _nullable_str_series(merged["R_emaildomain"], max_len=100),
            "addr1": _nullable_float_series(merged["addr1"]),
            "addr2": _nullable_float_series(merged["addr2"]),
            "dist1": _nullable_float_series(merged["dist1"]),
            "dist2": _nullable_float_series(merged["dist2"]),
        }
    )


def prepare_identities_df(merged: pd.DataFrame) -> pd.DataFrame:
    """Return DB-ready identity rows for transactions with identity data."""
    mask = _has_identity_data_mask(merged)
    if not mask.any():
        return pd.DataFrame(columns=list(IDENTITY_DB_COLUMNS))

    subset = merged.loc[mask]
    return pd.DataFrame(
        {
            "transaction_id": subset["TransactionID"].astype(int),
            "id_30": _nullable_str_series(subset["id_30"], max_len=100),
            "id_31": _nullable_str_series(subset["id_31"], max_len=100),
            "device_type": _nullable_str_series(subset["DeviceType"], max_len=50),
            "device_info": _nullable_str_series(subset["DeviceInfo"], max_len=100),
        }
    )


def count_identity_rows(merged: pd.DataFrame) -> int:
    """Count merged rows that include at least one identity field."""
    return int(_has_identity_data_mask(merged).sum())
