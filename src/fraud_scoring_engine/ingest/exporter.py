"""Export IEEE-CIS feature matrices to Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_train_features_parquet(features: pd.DataFrame, path: Path) -> None:
    """Write the train feature matrix to a Parquet file.

    Args:
        features: Feature DataFrame (must include ``TransactionID``).
        path: Output ``.parquet`` path; parent directories are created if needed.

    Raises:
        ValueError: If ``TransactionID`` is missing from ``features``.
    """
    if "TransactionID" not in features.columns:
        msg = "Feature matrix must include TransactionID"
        raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
