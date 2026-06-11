"""Orchestrate IEEE CSV ingestion into PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from fraud_scoring_engine.db.engine import create_db_engine
from fraud_scoring_engine.db.models import Transaction
from fraud_scoring_engine.ingest.paths import IeeeDataPaths, ieee_data_paths
from fraud_scoring_engine.ingest.reader import load_merged_train_data
from fraud_scoring_engine.ingest.transforms import merged_row_to_models


@dataclass(frozen=True)
class IngestResult:
    """Summary counts from an ingest run."""

    rows_read: int
    rows_inserted: int
    rows_skipped: int
    identities_inserted: int


def _existing_transaction_ids(session: Session, transaction_ids: list[int]) -> set[int]:
    if not transaction_ids:
        return set()
    stmt = select(Transaction.transaction_id).where(
        Transaction.transaction_id.in_(transaction_ids)
    )
    return set(session.scalars(stmt).all())


def ingest_train_transactions(
    *,
    data_dir: Path | None = None,
    limit: int | None = None,
    batch_size: int = 500,
    dry_run: bool = False,
) -> IngestResult:
    """Load train IEEE data from CSV and insert into PostgreSQL.

    Args:
        data_dir: Optional override for ``data/raw``.
        limit: If set, ingest only the first ``limit`` transaction rows.
        batch_size: Number of rows per database commit batch.
        dry_run: If True, transform only; do not write to the database.

    Returns:
        Summary counts for the ingest run.

    Raises:
        FileNotFoundError: If required CSV files are missing.
    """
    paths: IeeeDataPaths = ieee_data_paths(data_dir)
    merged = load_merged_train_data(
        paths.train_transaction,
        paths.train_identity,
        limit=limit,
    )

    rows_read = len(merged)
    rows_inserted = 0
    rows_skipped = 0
    identities_inserted = 0

    if dry_run:
        identity_count = sum(
            1
            for _, row in merged.iterrows()
            if merged_row_to_models(row)[1] is not None
        )
        print(f"Dry run: {rows_read} transactions, {identity_count} with identity data")
        return IngestResult(
            rows_read=rows_read,
            rows_inserted=0,
            rows_skipped=0,
            identities_inserted=identity_count,
        )

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

    print(
        f"Ingest complete: read={rows_read}, inserted={rows_inserted}, "
        f"skipped={rows_skipped}, identities={identities_inserted}"
    )
    return IngestResult(
        rows_read=rows_read,
        rows_inserted=rows_inserted,
        rows_skipped=rows_skipped,
        identities_inserted=identities_inserted,
    )
