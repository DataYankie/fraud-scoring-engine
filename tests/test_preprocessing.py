"""Tests for fraud_scoring_engine.preprocessing."""

import pandas as pd
import pytest

from fraud_scoring_engine.preprocessing import (
    DROP_COLUMNS,
    FraudFeaturePreprocessor,
    feature_columns,
)


def test_feature_columns_excludes_metadata() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "is_fraud": [0, 1],
            "transaction_dt": [10, 20],
            "C1": [1.0, 2.0],
            "M1": ["T", "F"],
        }
    )

    assert feature_columns(frame) == ["C1", "M1"]
    assert DROP_COLUMNS == frozenset({"transaction_id", "is_fraud", "transaction_dt"})


def test_fit_transform_preserves_numeric_nulls() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "is_fraud": [0, 1, 0],
            "C1": [1.0, None, 3.0],
            "avg_amount_ratio_30d": [None, 1.5, 2.0],
        }
    )

    preprocessor = FraudFeaturePreprocessor()
    transformed = preprocessor.fit_transform(frame)

    assert list(transformed.columns) == ["C1", "avg_amount_ratio_30d"]
    assert transformed["C1"].isna().tolist() == [False, True, False]
    assert transformed["avg_amount_ratio_30d"].isna().tolist() == [True, False, False]


def test_categorical_missing_and_unseen_levels_map_to_missing_category() -> None:
    train = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "is_fraud": [0, 1],
            "M1": ["T", None],
            "C1": [1.0, 2.0],
        }
    )
    scoring = pd.DataFrame(
        {
            "transaction_id": [3, 4],
            "is_fraud": [0, 0],
            "M1": ["F", "brand_new"],
            "C1": [3.0, 4.0],
        }
    )

    preprocessor = FraudFeaturePreprocessor()
    train_out = preprocessor.fit_transform(train)
    score_out = preprocessor.transform(scoring)

    assert train_out["M1"].tolist() == ["T", "__MISSING__"]
    assert score_out["M1"].tolist() == ["__MISSING__", "__MISSING__"]


def test_transform_requires_fit() -> None:
    frame = pd.DataFrame({"C1": [1.0]})
    preprocessor = FraudFeaturePreprocessor()

    with pytest.raises(AttributeError, match="not been fitted"):
        preprocessor.transform(frame)


def test_transform_raises_when_required_columns_missing() -> None:
    train = pd.DataFrame(
        {
            "transaction_id": [1],
            "is_fraud": [0],
            "C1": [1.0],
            "M1": ["T"],
        }
    )
    scoring = pd.DataFrame({"transaction_id": [2], "is_fraud": [0], "C1": [2.0]})

    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(train)

    with pytest.raises(ValueError, match="missing fitted feature columns"):
        preprocessor.transform(scoring)


def test_explicit_column_lists_override_auto_detection() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "is_fraud": [0, 1],
            "C1": [1.0, 2.0],
            "M1": ["T", "F"],
        }
    )

    preprocessor = FraudFeaturePreprocessor(
        numeric_columns=["C1"],
        categorical_columns=["M1"],
    )
    transformed = preprocessor.fit_transform(frame)

    assert list(transformed.columns) == ["C1", "M1"]
    assert transformed["M1"].dtype.name == "category"


def test_get_feature_names_out_matches_transform_columns() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [1],
            "is_fraud": [0],
            "C1": [1.0],
            "M1": ["T"],
        }
    )

    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(frame)

    assert preprocessor.get_feature_names_out() == ["C1", "M1"]
