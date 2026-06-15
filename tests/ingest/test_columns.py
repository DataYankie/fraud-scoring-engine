"""Tests for fraud_scoring_engine.ingest.columns."""

import pandas as pd

from fraud_scoring_engine.ingest.columns import (
    PARQUET_EXCLUDE_COLUMNS,
    split_parquet_features,
)


def test_split_parquet_features_excludes_postgres_columns() -> None:
    merged = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "TransactionAmt": [10.0, 20.0],
            "TransactionDT": [86400, 86401],
            "ProductCD": ["W", "H"],
            "card1": [100.0, 200.0],
            "card4": ["visa", "mastercard"],
            "P_emaildomain": ["gmail.com", None],
            "addr1": [1.0, 2.0],
            "dist1": [3.0, 4.0],
            "V1": [0.1, 0.2],
            "C1": [1.0, 2.0],
            "id_01": [10.0, None],
            "id_30": ["Windows 10", None],
            "DeviceType": ["desktop", None],
        }
    )

    features = split_parquet_features(merged)

    assert "TransactionID" in features.columns
    assert "V1" in features.columns
    assert "C1" in features.columns
    assert "id_01" in features.columns
    for excluded in PARQUET_EXCLUDE_COLUMNS:
        assert excluded not in features.columns


def test_split_parquet_features_preserves_row_count() -> None:
    merged = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [1.0, 2.0, 3.0],
            "TransactionDT": [1, 2, 3],
            "V1": [0.1, 0.2, 0.3],
        }
    )
    features = split_parquet_features(merged)
    assert len(features) == 3
    assert list(features["TransactionID"]) == [1, 2, 3]
