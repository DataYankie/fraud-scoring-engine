"""Column sets for PostgreSQL ingestion vs Parquet feature export."""

from __future__ import annotations

import pandas as pd

# CSV columns loaded into PostgreSQL ``transactions`` (+ ``TransactionID`` join key).
TRANSACTION_USECOLS = [
    "TransactionID",
    "isFraud",
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
]

# CSV columns loaded into PostgreSQL ``transaction_identities``.
IDENTITY_USECOLS = [
    "TransactionID",
    "id_30",
    "id_31",
    "DeviceType",
    "DeviceInfo",
]

TRANSACTION_DTYPES: dict[str, str] = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "isFraud": "int8",
}

# Columns stored in Postgres that are excluded from the Parquet feature matrix.
# ``TransactionID`` is kept in Parquet as the join key to ``transactions.transaction_id``.
PARQUET_EXCLUDE_COLUMNS = frozenset(
    (set(TRANSACTION_USECOLS) - {"TransactionID"})
    | {col for col in IDENTITY_USECOLS if col != "TransactionID"}
)


def database_columns(merged: pd.DataFrame) -> list[str]:
    """Return ordered CSV columns required for ORM ingestion."""
    columns: list[str] = []
    for name in TRANSACTION_USECOLS:
        if name in merged.columns:
            columns.append(name)
    for name in IDENTITY_USECOLS:
        if name != "TransactionID" and name in merged.columns:
            columns.append(name)
    return columns


def split_parquet_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Return columns not stored in Postgres (plus ``TransactionID`` join key).

    Args:
        merged: Full train transaction + identity DataFrame.

    Returns:
        Feature matrix for ML training with one row per transaction.
    """
    drop_cols = [col for col in PARQUET_EXCLUDE_COLUMNS if col in merged.columns]
    features = merged.drop(columns=drop_cols).copy()
    if "TransactionID" not in features.columns:
        msg = "Parquet feature frame must include TransactionID as join key"
        raise ValueError(msg)
    return features
