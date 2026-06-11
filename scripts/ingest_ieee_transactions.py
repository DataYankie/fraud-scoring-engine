"""Ingest IEEE-CIS train transactions into PostgreSQL.

Usage:
    uv run python scripts/ingest_ieee_transactions.py --limit 10000
    uv run python scripts/ingest_ieee_transactions.py
    uv run python scripts/ingest_ieee_transactions.py --limit 100 --dry-run

Prerequisites:
    - Raw CSVs in data/raw/ (see scripts/download_ieee_fraud_data.py)
    - PostgreSQL running (docker-compose up -d)
    - POSTGRES_PASSWORD or DATABASE_URL set (see env_local.ps1)
    - Schema applied: alembic upgrade head
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_scoring_engine.ingest.loader import ingest_train_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest IEEE-CIS train transactions into PostgreSQL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N transaction rows (for dev iteration).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data/raw (default: {repo}/data/raw).",
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
        help="Read and transform only; do not write to the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_train_transactions(
        data_dir=args.data_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
