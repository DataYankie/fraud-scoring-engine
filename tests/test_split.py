"""Tests for fraud_scoring_engine.training.split."""

import pandas as pd
import pytest

from fraud_scoring_engine.training.split import DatasetSplit, time_split


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [5, 1, 2, 3, 4, 6, 7, 8, 9, 10],
            "transaction_dt": [50, 10, 20, 30, 40, 60, 70, 80, 90, 100],
            "is_fraud": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            "C1": range(10),
        }
    )


def test_time_split_orders_by_time_and_assigns_contiguous_blocks() -> None:
    split = time_split(_frame(), ratios=(0.5, 0.2, 0.3))

    assert isinstance(split, DatasetSplit)
    assert split.sizes == (5, 2, 3)
    assert list(split.train["transaction_id"]) == [1, 2, 3, 4, 5]
    assert list(split.val["transaction_id"]) == [6, 7]
    assert list(split.test["transaction_id"]) == [8, 9, 10]
    assert split.train["transaction_dt"].max() <= split.val["transaction_dt"].min()
    assert split.val["transaction_dt"].max() <= split.test["transaction_dt"].min()


def test_time_split_uses_tiebreaker_for_same_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [2, 1, 3],
            "transaction_dt": [10, 10, 20],
            "is_fraud": [0, 1, 0],
        }
    )

    split = time_split(frame, ratios=(1 / 3, 1 / 3, 1 / 3))

    assert list(split.train["transaction_id"]) == [1]
    assert list(split.val["transaction_id"]) == [2]
    assert list(split.test["transaction_id"]) == [3]


def test_time_split_does_not_mutate_input() -> None:
    frame = _frame()
    original_ids = list(frame["transaction_id"])

    time_split(frame)

    assert list(frame["transaction_id"]) == original_ids


def test_time_split_assigns_remainder_rows_to_test() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": range(11),
            "transaction_dt": range(11),
            "is_fraud": [0] * 11,
        }
    )

    split = time_split(frame, ratios=(0.70, 0.15, 0.15))

    assert sum(split.sizes) == 11
    assert split.sizes == (7, 1, 3)


def test_time_split_raises_when_time_column_missing() -> None:
    frame = pd.DataFrame({"transaction_id": [1, 2, 3], "is_fraud": [0, 1, 0]})

    with pytest.raises(ValueError, match="transaction_dt"):
        time_split(frame)


def test_time_split_raises_when_ratios_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        time_split(_frame(), ratios=(0.5, 0.2, 0.2))


def test_time_split_raises_when_a_split_would_be_empty() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "transaction_dt": [10, 20],
            "is_fraud": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="empty partition"):
        time_split(frame, ratios=(0.70, 0.15, 0.15))


def test_time_split_raises_on_empty_frame() -> None:
    frame = pd.DataFrame(columns=["transaction_id", "transaction_dt", "is_fraud"])

    with pytest.raises(ValueError, match="empty DataFrame"):
        time_split(frame)
