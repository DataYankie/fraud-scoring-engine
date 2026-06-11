"""IEEE-CIS data ingestion into PostgreSQL."""

from fraud_scoring_engine.ingest.loader import IngestResult, ingest_train_transactions

__all__ = ["IngestResult", "ingest_train_transactions"]
