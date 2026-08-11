# Fraud Scoring Engine
> 🚧 **Status:** Under active development — core data pipeline are in place; training experiments and scoring service coming next.
End-to-end system for scoring payment transactions for fraud risk, built around the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Current progress
- [x] Data download + Postgres ingest
- [x] Schema migrations (Alembic)
- [x] Feature scaffolding + tests/CI
- [x] EDA
- [ ] XGBoost notebook
- [ ] Training pipeline
- [ ] Scoring API
- [ ] Evaluation / monitoring

## Stack
Python, PostgreSQL, Alembic, scikit-learn/XGBoost, Docker Compose, GitHub Actions

## Data pipeline
See [scripts/README.md](scripts/README.md) for downloading IEEE-CIS data and ingesting into PostgreSQL.
