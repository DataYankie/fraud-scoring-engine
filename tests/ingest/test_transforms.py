"""Tests for fraud_scoring_engine.ingest.transforms."""

from datetime import datetime

import pandas as pd

from fraud_scoring_engine.ingest.transforms import (
    derive_transaction_at,
    generate_user_id,
    merged_row_to_models,
)


def test_generate_user_id_is_deterministic() -> None:
    row = pd.Series(
        {
            "card1": 13926.0,
            "card2": None,
            "card3": 150.0,
            "card4": "discover",
            "card5": 142.0,
            "card6": "credit",
            "addr1": 315.0,
            "addr2": 87.0,
        }
    )
    first = generate_user_id(row)
    second = generate_user_id(row)
    assert first == second
    assert len(first) == 32


def test_generate_user_id_treats_nan_as_empty() -> None:
    with_nan = pd.Series(
        {
            "card1": float("nan"),
            "card2": None,
            "card3": float("nan"),
            "card4": "visa",
            "card5": float("nan"),
            "card6": "debit",
            "addr1": 100.0,
            "addr2": float("nan"),
        }
    )
    without_nan = pd.Series(
        {
            "card1": None,
            "card2": None,
            "card3": None,
            "card4": "visa",
            "card5": None,
            "card6": "debit",
            "addr1": 100.0,
            "addr2": None,
        }
    )
    assert generate_user_id(with_nan) == generate_user_id(without_nan)


def test_derive_transaction_at() -> None:
    assert derive_transaction_at(86400) == datetime(2017, 12, 1)


def test_merged_row_to_models_with_identity() -> None:
    row = pd.Series(
        {
            "TransactionID": 2987000,
            "isFraud": 0,
            "TransactionAmt": 68.5,
            "TransactionDT": 86400,
            "ProductCD": "W",
            "card1": 13926.0,
            "card2": None,
            "card3": 150.0,
            "card4": "discover",
            "card5": 142.0,
            "card6": "credit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": None,
            "addr1": 315.0,
            "addr2": 87.0,
            "dist1": 19.0,
            "dist2": None,
            "id_30": "Windows 10",
            "id_31": "chrome 63.0",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
        }
    )
    transaction, identity = merged_row_to_models(row)
    assert transaction.transaction_id == 2987000
    assert transaction.derived_user_id is not None
    assert len(transaction.derived_user_id) == 32
    assert transaction.is_fraud == 0
    assert transaction.transaction_at == datetime(2017, 12, 1)
    assert identity is not None
    assert identity.transaction_id == 2987000
    assert identity.device_type == "desktop"


def test_merged_row_to_models_without_identity() -> None:
    row = pd.Series(
        {
            "TransactionID": 2987001,
            "isFraud": 1,
            "TransactionAmt": 29.0,
            "TransactionDT": 86401,
            "ProductCD": "W",
            "card1": 2755.0,
            "card2": 404.0,
            "card3": 150.0,
            "card4": "mastercard",
            "card5": 102.0,
            "card6": "credit",
            "P_emaildomain": None,
            "R_emaildomain": None,
            "addr1": 325.0,
            "addr2": 87.0,
            "dist1": None,
            "dist2": None,
            "id_30": None,
            "id_31": None,
            "DeviceType": None,
            "DeviceInfo": None,
        }
    )
    transaction, identity = merged_row_to_models(row)
    assert transaction.transaction_id == 2987001
    assert transaction.is_fraud == 1
    assert identity is None
