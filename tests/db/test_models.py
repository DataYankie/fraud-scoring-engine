"""Tests for fraud_scoring_engine.db.models."""

from fraud_scoring_engine.db.models import Base, FraudAlert, Transaction, UserRiskProfile


def test_base_metadata_registers_all_tables() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert table_names == {"user_risk_profiles", "transactions", "fraud_alerts"}


def test_user_risk_profile_tablename() -> None:
    assert UserRiskProfile.__tablename__ == "user_risk_profiles"


def test_transaction_tablename() -> None:
    assert Transaction.__tablename__ == "transactions"


def test_fraud_alert_tablename() -> None:
    assert FraudAlert.__tablename__ == "fraud_alerts"


def test_transaction_foreign_keys() -> None:
    table = Base.metadata.tables["transactions"]
    foreign_keys = {fk.target_fullname for fk in table.foreign_keys}
    assert "user_risk_profiles.user_id" in foreign_keys


def test_fraud_alert_foreign_keys() -> None:
    table = Base.metadata.tables["fraud_alerts"]
    foreign_keys = {fk.target_fullname for fk in table.foreign_keys}
    assert "transactions.id" in foreign_keys


def test_transaction_timestamp_indexed() -> None:
    table = Base.metadata.tables["transactions"]
    indexed_columns = {
        column.name
        for index in table.indexes
        for column in index.columns
    }
    assert "timestamp" in indexed_columns
