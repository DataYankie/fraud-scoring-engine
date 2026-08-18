"""Load and merge IEEE-CIS CSV inputs for downstream ingest steps.

This module provides the raw-data readers used by the ingest pipeline,
including helpers for DB-only column slices, full feature loads, identity
filtering, and the merged training frame consumed by orchestration code.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from fraud_scoring_engine.ingest.columns import (
    IDENTITY_USECOLS,
    TRANSACTION_DTYPES,
    TRANSACTION_USECOLS,
    database_columns,
    split_parquet_features,
)


def load_train_transactions(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load train transaction rows with DB-relevant columns only.

    Args:
        path: Path to ``train_transaction.csv``.
        limit: If set, read only the first ``limit`` rows.

    Returns:
        Transaction DataFrame keyed by ``TransactionID``.
    """
    full = load_train_transactions_full(path, limit=limit)
    return full[TRANSACTION_USECOLS].copy()


def load_train_transactions_full(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load all train transaction columns from CSV.

    Args:
        path: Path to ``train_transaction.csv``.
        limit: If set, read only the first ``limit`` rows.

    Returns:
        Full transaction DataFrame for the requested slice.
    """
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    return pd.read_csv(
        path,
        dtype=cast(Mapping[Hashable, str | type], TRANSACTION_DTYPES),
        nrows=limit,
        low_memory=False,
    )


def load_train_identity(path: Path, transaction_ids: Iterable[int]) -> pd.DataFrame:
    """Load DB-relevant identity rows for the given transaction IDs."""
    full = load_train_identity_full(path, transaction_ids)
    return full[IDENTITY_USECOLS].copy()


def load_train_identity_full(path: Path, transaction_ids: Iterable[int]) -> pd.DataFrame:
    """Load all identity columns for the given transaction IDs.

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
    """Load and left-join train transaction and identity data for Postgres.

    Args:
        transaction_path: Path to ``train_transaction.csv``.
        identity_path: Path to ``train_identity.csv``.
        limit: If set, read only the first ``limit`` transaction rows.

    Returns:
        Merged DataFrame with DB-relevant columns only.
    """
    merged_full = load_merged_train_data_full(
        transaction_path,
        identity_path,
        limit=limit,
    )
    return merged_full[database_columns(merged_full)].copy()


def load_merged_train_data_full(
    transaction_path: Path,
    identity_path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and left-join full train transaction and identity CSV columns.

    Args:
        transaction_path: Path to ``train_transaction.csv``.
        identity_path: Path to ``train_identity.csv``.
        limit: If set, read only the first ``limit`` transaction rows.

    Returns:
        Merged DataFrame with all CSV columns for the slice.
    """
    transactions = load_train_transactions_full(transaction_path, limit=limit)
    identity = load_train_identity_full(
        identity_path,
        transactions["TransactionID"].astype(int).tolist(),
    )
    return transactions.merge(identity, on="TransactionID", how="left")


def load_train_feature_matrix(
    transaction_path: Path,
    identity_path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load non-Postgres columns into a single feature matrix.

    Args:
        transaction_path: Path to ``train_transaction.csv``.
        identity_path: Path to ``train_identity.csv``.
        limit: If set, read only the first ``limit`` transaction rows.

    Returns:
        Parquet-ready feature DataFrame keyed by ``TransactionID``.
    """
    merged_full = load_merged_train_data_full(
        transaction_path,
        identity_path,
        limit=limit,
    )
    return split_parquet_features(merged_full)
