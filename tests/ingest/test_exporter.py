"""Tests for fraud_scoring_engine.ingest.exporter."""

from pathlib import Path

import pandas as pd
import pytest

from fraud_scoring_engine.ingest.exporter import write_train_features_parquet


def test_write_train_features_parquet(tmp_path: Path) -> None:
    features = pd.DataFrame({"TransactionID": [1, 2], "V1": [0.1, 0.2]})
    output = tmp_path / "nested" / "train_features.parquet"

    write_train_features_parquet(features, output)

    assert output.exists()
    loaded = pd.read_parquet(output)
    pd.testing.assert_frame_equal(loaded, features)


def test_write_train_features_parquet_requires_transaction_id(tmp_path: Path) -> None:
    features = pd.DataFrame({"V1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="TransactionID"):
        write_train_features_parquet(features, tmp_path / "train_features.parquet")
