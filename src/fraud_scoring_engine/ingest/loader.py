"""Orchestrate IEEE transaction ingestion into PostgreSQL and Parquet.

This module coordinates the end-to-end ingest flow used by the script layer:
it loads merged raw data, skips already ingested transactions, writes database
records, and exports training features to Parquet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.engine import create_db_engine
from fraud_scoring_engine.db.models import Transaction, TransactionIdentity
from fraud_scoring_engine.ingest.columns import database_columns, split_parquet_features
from fraud_scoring_engine.ingest.exporter import write_train_features_parquet
from fraud_scoring_engine.ingest.paths import IeeeDataPaths, ieee_data_paths
from fraud_scoring_engine.ingest.reader import load_merged_train_data_full
from fraud_scoring_engine.ingest.transforms import (
    count_identity_rows,
    prepare_identities_df,
    prepare_transactions_df,
)

EXISTING_ID_LOOKUP_BATCH = 10_000


@dataclass(frozen=True)
class IngestResult:
    """Summary counts from an ingest run."""

    rows_read: int
    rows_inserted: int
    rows_skipped: int
    identities_inserted: int
    parquet_rows: int
    parquet_path: Path | None


def _existing_transaction_ids(session: Session, transaction_ids: list[int]) -> set[int]:
    if not transaction_ids:
        return set()
    stmt = select(Transaction.transaction_id).where(
        Transaction.transaction_id.in_(transaction_ids)
    )
    return set(session.scalars(stmt).all())


def _lookup_existing_transaction_ids(
    session: Session,
    transaction_ids: list[int],
    *,
    lookup_batch_size: int = EXISTING_ID_LOOKUP_BATCH,
) -> set[int]:
    existing: set[int] = set()
    for start in range(0, len(transaction_ids), lookup_batch_size):
        batch_ids = transaction_ids[start : start + lookup_batch_size]
        existing |= _existing_transaction_ids(session, batch_ids)
    return existing


def _bulk_insert_records(
    session: Session,
    table,
    records: list[Mapping[str, Any]],
    *,
    batch_size: int,
) -> int:
    inserted = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            continue
        session.execute(insert(table), batch)
        session.commit()
        inserted += len(batch)
    return inserted


def _ingest_to_postgres(
    merged: pd.DataFrame,
    *,
    batch_size: int,
) -> tuple[int, int, int]:
    rows_read = len(merged)
    transactions_df = prepare_transactions_df(merged)
    identities_df = prepare_identities_df(merged)

    engine = create_db_engine()
    with Session(engine) as session:
        incoming_ids = transactions_df["transaction_id"].astype(int).tolist()
        existing = _lookup_existing_transaction_ids(session, incoming_ids)
        rows_skipped = len(existing)

        to_insert = transactions_df[~transactions_df["transaction_id"].isin(existing)]
        inserted_ids = set(to_insert["transaction_id"].astype(int).tolist())
        identities_to_insert = identities_df[
            identities_df["transaction_id"].isin(inserted_ids)
        ]

        transaction_records = cast(
            list[Mapping[str, Any]],
            to_insert.to_dict(orient="records"),
        )
        identity_records = cast(
            list[Mapping[str, Any]],
            identities_to_insert.to_dict(orient="records"),
        )

        rows_inserted = _bulk_insert_records(
            session,
            Transaction.__table__,
            transaction_records,
            batch_size=batch_size,
        )
        identities_inserted = _bulk_insert_records(
            session,
            TransactionIdentity.__table__,
            identity_records,
            batch_size=batch_size,
        )

    return rows_inserted, rows_skipped, identities_inserted


def ingest_train_transactions(
    *,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    parquet_path: Path | None = None,
    limit: int | None = None,
    batch_size: int = 5000,
    dry_run: bool = False,
    skip_db: bool = False,
    skip_parquet: bool = False,
) -> IngestResult:
    """Load train IEEE data from CSV into PostgreSQL and/or Parquet.

    Postgres receives the operational column subset. All remaining CSV columns
    (plus ``TransactionID`` as join key) are written to a single Parquet file.

    Args:
        data_dir: Optional override for ``data/raw``.
        processed_dir: Optional override for ``data/processed``.
        parquet_path: Optional override for the output Parquet file.
        limit: If set, process only the first ``limit`` transaction rows.
        batch_size: Number of rows per database commit batch.
        dry_run: If True, report counts only; do not write Postgres or Parquet.
        skip_db: If True, export Parquet only.
        skip_parquet: If True, ingest Postgres only.

    Returns:
        Summary counts for the ingest run.

    Raises:
        FileNotFoundError: If required CSV files are missing.
        ValueError: If both ``skip_db`` and ``skip_parquet`` are True.
    """
    if skip_db and skip_parquet:
        msg = "At least one of Postgres ingest or Parquet export must be enabled"
        raise ValueError(msg)

    paths: IeeeDataPaths = ieee_data_paths(
        data_dir,
        processed_dir=processed_dir,
        parquet_path=parquet_path,
    )

    merged_full = load_merged_train_data_full(
        paths.train_transaction,
        paths.train_identity,
        limit=limit,
    )
    merged = merged_full[database_columns(merged_full)].copy()
    rows_read = len(merged)

    parquet_rows = 0
    output_parquet: Path | None = None
    rows_inserted = 0
    rows_skipped = 0
    identities_inserted = 0

    if not skip_parquet:
        features = split_parquet_features(merged_full)
        parquet_rows = len(features)
        output_parquet = paths.train_features_parquet
        if not dry_run:
            write_train_features_parquet(features, output_parquet)

    if dry_run:
        identity_count = count_identity_rows(merged)
        print(
            f"Dry run: {rows_read} transactions, {identity_count} with identity data, "
            f"{parquet_rows} parquet feature rows"
        )
        if output_parquet is not None:
            print(f"Parquet target: {output_parquet}")
        return IngestResult(
            rows_read=rows_read,
            rows_inserted=0,
            rows_skipped=0,
            identities_inserted=identity_count,
            parquet_rows=parquet_rows,
            parquet_path=output_parquet,
        )

    if not skip_db:
        rows_inserted, rows_skipped, identities_inserted = _ingest_to_postgres(
            merged,
            batch_size=batch_size,
        )

    parts = [
        f"read={rows_read}",
        f"inserted={rows_inserted}",
        f"skipped={rows_skipped}",
        f"identities={identities_inserted}",
    ]
    if not skip_parquet:
        parts.append(f"parquet_rows={parquet_rows}")
        parts.append(f"parquet_path={output_parquet}")
    print(f"Ingest complete: {', '.join(parts)}")

    return IngestResult(
        rows_read=rows_read,
        rows_inserted=rows_inserted,
        rows_skipped=rows_skipped,
        identities_inserted=identities_inserted,
        parquet_rows=parquet_rows,
        parquet_path=output_parquet,
    )
