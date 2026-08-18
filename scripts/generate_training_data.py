"""Generate behavioral training features from ingested transactions.

This script reads transaction records from PostgreSQL, computes rolling
behavioral features, and writes the resulting dataset to a Parquet file for
training and experimentation.

Usage:
    uv run python scripts/generate_training_data.py --limit 10000
    uv run python scripts/generate_training_data.py --limit 10000 --output data/processed/behavioral_features.parquet
    uv run python scripts/generate_training_data.py --dry-run

Prerequisites:
    - PostgreSQL running with ingested transactions
    - `POSTGRES_PASSWORD` or `DATABASE_URL` set
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from fraud_scoring_engine.db.engine import create_db_engine
from fraud_scoring_engine.features import compute_transaction_features_dataframe
from fraud_scoring_engine.ingest.paths import repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute rolling behavioral features for transactions and export to Parquet."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Maximum number of transactions to export (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output Parquet path "
            "(default: {repo}/data/processed/behavioral_features.parquet)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features but do not write Parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output
        if args.output is not None
        else repo_root() / "data" / "processed" / "behavioral_features.parquet"
    )

    engine = create_db_engine()
    with Session(engine) as session:
        dataframe = compute_transaction_features_dataframe(session, limit=args.limit)

    print(f"Computed behavioral features for {len(dataframe)} transactions.")

    if args.dry_run:
        print("Dry run: skipping Parquet write.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output, index=False)
    print(f"Wrote {len(dataframe)} rows to {output}")


if __name__ == "__main__":
    main()
