"""Orchestrate IEEE CSV ingestion into PostgreSQL and Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.engine import create_db_engine
from fraud_scoring_engine.db.models import Transaction
from fraud_scoring_engine.ingest.columns import database_columns, split_parquet_features
from fraud_scoring_engine.ingest.exporter import write_train_features_parquet
from fraud_scoring_engine.ingest.paths import IeeeDataPaths, ieee_data_paths
from fraud_scoring_engine.ingest.reader import load_merged_train_data_full
from fraud_scoring_engine.ingest.transforms import merged_row_to_models


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


def _ingest_to_postgres(
    merged,
    *,
    batch_size: int,
) -> tuple[int, int, int]:
    rows_read = len(merged)
    rows_inserted = 0
    rows_skipped = 0
    identities_inserted = 0

    engine = create_db_engine()
    with Session(engine) as session:
        for start in range(0, rows_read, batch_size):
            batch = merged.iloc[start : start + batch_size]
            batch_ids = batch["TransactionID"].astype(int).tolist()
            existing = _existing_transaction_ids(session, batch_ids)

            to_insert: list[object] = []
            batch_inserted = 0
            batch_identities = 0

            for _, row in batch.iterrows():
                transaction_id = int(row["TransactionID"])
                if transaction_id in existing:
                    rows_skipped += 1
                    continue

                transaction, identity = merged_row_to_models(row)
                to_insert.append(transaction)
                batch_inserted += 1
                if identity is not None:
                    to_insert.append(identity)
                    batch_identities += 1

            if to_insert:
                session.add_all(to_insert)
                session.commit()

            rows_inserted += batch_inserted
            identities_inserted += batch_identities

    return rows_inserted, rows_skipped, identities_inserted


def ingest_train_transactions(
    *,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    parquet_path: Path | None = None,
    limit: int | None = None,
    batch_size: int = 500,
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
        identity_count = sum(
            1
            for _, row in merged.iterrows()
            if merged_row_to_models(row)[1] is not None
        )
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
