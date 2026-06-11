# Scripts

Operational commands for the fraud-scoring-engine data pipeline.

## Prerequisites
- uv installed
- Docker (PostgreSQL via docker-compose)
- `. env_local.ps1` or equivalent POSTGRES_* vars

## Workflow

1. Download IEEE data
2. Apply migrations
3. Ingest (dev → full)

### 1. Download data
uv run python scripts/download_ieee_fraud_data.py

### 2. Apply schema
alembic upgrade head

### 3. Ingest transactions
#### Dev pass (10k rows)
    uv run python scripts/ingest_ieee_transactions.py --limit 10000

#### Dry run
    uv run python scripts/ingest_ieee_transactions.py --limit 100 --dry-run

#### Full train load
    uv run python scripts/ingest_ieee_transactions.py