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
```bash
uv run python scripts/download_ieee_fraud_data.py
```

### 2. Apply schema
```bash
alembic upgrade head
```

### 3. Ingest transactions
Operational columns go to PostgreSQL. All other CSV columns are written to a single Parquet file at `data/processed/train_features.parquet` (same `--limit` slice).

#### Dev pass (10k rows)
```bash
uv run python scripts/ingest_ieee_transactions.py --limit 10000
```

#### Dry run
```bash
uv run python scripts/ingest_ieee_transactions.py --limit 100 --dry-run
```

#### Parquet only (no Postgres)
```bash
uv run python scripts/ingest_ieee_transactions.py --skip-db --limit 10000
```

#### Full train load
```bash
uv run python scripts/ingest_ieee_transactions.py
```

#### Training features in Python
```python
import pandas as pd

features = pd.read_parquet("data/processed/train_features.parquet")
# Join to Postgres on features["TransactionID"] == transactions.transaction_id
```
