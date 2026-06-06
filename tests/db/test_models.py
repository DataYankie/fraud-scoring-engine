"""Tests for fraud_scoring_engine.db.models."""

from fraud_scoring_engine.db.models import Base, FraudAlert, Transaction, TransactionIdentity


def test_base_metadata_registers_all_tables() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert table_names == {"transactions", "transaction_identities", "fraud_alerts"}


def test_transaction_tablename() -> None:
    assert Transaction.__tablename__ == "transactions"


def test_transaction_identity_tablename() -> None:
    assert TransactionIdentity.__tablename__ == "transaction_identities"


def test_fraud_alert_tablename() -> None:
    assert FraudAlert.__tablename__ == "fraud_alerts"


def test_transaction_foreign_keys() -> None:
    table = Base.metadata.tables["transactions"]
    assert len(table.foreign_keys) == 0


def test_fraud_alert_foreign_keys() -> None:
    table = Base.metadata.tables["fraud_alerts"]
    foreign_keys = {fk.target_fullname for fk in table.foreign_keys}
    assert foreign_keys == {"transactions.transaction_id"}


def test_transaction_identity_foreign_keys() -> None:
    table = Base.metadata.tables["transaction_identities"]
    foreign_keys = {fk.target_fullname for fk in table.foreign_keys}
    assert foreign_keys == {"transactions.transaction_id"}


def test_transaction_time_columns_indexed() -> None:
    table = Base.metadata.tables["transactions"]
    indexed_columns = {
        column.name
        for index in table.indexes
        for column in index.columns
    }
    assert indexed_columns == {"transaction_dt", "transaction_at", "derived_user_id"}


def test_transaction_has_ieee_ops_columns() -> None:
    column_names = {column.name for column in Transaction.__table__.columns}
    assert {"r_emaildomain", "addr2", "dist2", "card3", "card5"}.issubset(column_names)
