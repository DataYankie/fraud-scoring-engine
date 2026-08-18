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

### 4. Generate behavioral training features
Computes rolling velocity, spend, and amount-ratio features from PostgreSQL (one query + in-memory pass).

```bash
uv run python scripts/generate_training_data.py --limit 10000
```

#### MLflow UI
Requires `POSTGRES_PASSWORD` (or `MLFLOW_TRACKING_URI`) in the environment.

```bash
uv run mlflow ui --backend-store-uri postgresql+psycopg://postgres:PASSWORD@localhost:5432/mlflow_db
```

Tracking metadata lives in `mlflow_db`; model artifacts are stored under `mlartifacts/`.
