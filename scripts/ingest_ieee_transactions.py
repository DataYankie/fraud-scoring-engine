"""Ingest IEEE-CIS train transactions into PostgreSQL and Parquet.

Usage:
    uv run python scripts/ingest_ieee_transactions.py --limit 10000
    uv run python scripts/ingest_ieee_transactions.py
    uv run python scripts/ingest_ieee_transactions.py --limit 100 --dry-run
    uv run python scripts/ingest_ieee_transactions.py --skip-db --limit 10000
    uv run python scripts/ingest_ieee_transactions.py --skip-parquet

Writes operational columns to PostgreSQL and all other CSV columns to
``data/processed/train_features.parquet`` (same row slice as ``--limit``).

Prerequisites:
    - Raw CSVs in data/raw/ (see scripts/download_ieee_fraud_data.py)
    - PostgreSQL running (docker-compose up -d) when not using ``--skip-db``
    - POSTGRES_PASSWORD or DATABASE_URL set (see env_local.ps1)
    - Schema applied: alembic upgrade head
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_scoring_engine.ingest.loader import ingest_train_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest IEEE-CIS train data into PostgreSQL and export ML features to Parquet."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N transaction rows (for dev iteration).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data/raw (default: {repo}/data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to data/processed (default: {repo}/data/processed).",
    )
    parser.add_argument(
        "--parquet-out",
        type=Path,
        default=None,
        help="Output Parquet path (default: {repo}/data/processed/train_features.parquet).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per database commit batch (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and transform only; do not write to Postgres or Parquet.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Export Parquet only; skip PostgreSQL ingest.",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Ingest PostgreSQL only; skip Parquet export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_train_transactions(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        parquet_path=args.parquet_out,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        skip_db=args.skip_db,
        skip_parquet=args.skip_parquet,
    )


if __name__ == "__main__":
    main()
