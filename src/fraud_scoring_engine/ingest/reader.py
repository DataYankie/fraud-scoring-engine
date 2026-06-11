"""Load IEEE-CIS CSV subsets for database ingestion."""

from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Literal, cast

import pandas as pd

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

IDENTITY_USECOLS = [
    "TransactionID",
    "id_30",
    "id_31",
    "DeviceType",
    "DeviceInfo",
]

TRANSACTION_DTYPES: dict[
    Literal["TransactionID", "TransactionDT", "isFraud"],
    Literal["int32", "int8"],
] = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "isFraud": "int8",
}


def load_train_transactions(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load train transaction rows with DB-relevant columns only.

    Args:
        path: Path to ``train_transaction.csv``.
        limit: If set, read only the first ``limit`` rows.

    Returns:
        Transaction DataFrame keyed by ``TransactionID``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    return pd.read_csv(
        path,
        usecols=TRANSACTION_USECOLS,
        dtype=cast(Mapping[Hashable, str | type], TRANSACTION_DTYPES),
        nrows=limit,
        low_memory=False,
    )


def load_train_identity(path: Path, transaction_ids: Iterable[int]) -> pd.DataFrame:
    """Load identity rows for the given transaction IDs.

    Args:
        path: Path to ``train_identity.csv``.
        transaction_ids: Transaction IDs to retain after loading.

    Returns:
        Filtered identity DataFrame; may be empty if no matches.
    """
    if not path.exists():
        raise FileNotFoundError(f"Identity file not found: {path}")

    ids = list({int(transaction_id) for transaction_id in transaction_ids})
    identity = pd.read_csv(
        path,
        usecols=IDENTITY_USECOLS,
        dtype={"TransactionID": "int32"},
        low_memory=False,
    )
    filtered = identity.loc[identity["TransactionID"].isin(ids)]
    return cast(pd.DataFrame, filtered.copy())


def load_merged_train_data(
    transaction_path: Path,
    identity_path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and left-join train transaction and identity data.

    Args:
        transaction_path: Path to ``train_transaction.csv``.
        identity_path: Path to ``train_identity.csv``.
        limit: If set, read only the first ``limit`` transaction rows.

    Returns:
        Merged DataFrame with one row per transaction.
    """
    transactions = load_train_transactions(transaction_path, limit=limit)
    identity = load_train_identity(
        identity_path,
        transactions["TransactionID"].astype(int).tolist(),
    )
    return transactions.merge(identity, on="TransactionID", how="left")
