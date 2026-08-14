"""Column metadata for fraud model preprocessing."""

from __future__ import annotations

import pandas as pd

# Columns excluded from the model feature matrix.
DROP_COLUMNS: frozenset[str] = frozenset({"transaction_id", "is_fraud"})

# Placeholder category for null or unseen categorical levels at scoring time.
MISSING_CATEGORY = "__MISSING__"


def feature_columns(
    frame: pd.DataFrame,
    *,
    drop_columns: frozenset[str] = DROP_COLUMNS,
) -> list[str]:
    """Return model feature column names from a merged training frame.

    Args:
        frame: Merged Parquet + behavioral (+ optional Postgres) DataFrame.
        drop_columns: Metadata columns to exclude from modeling.

    Returns:
        Ordered feature names present in ``frame``.
    """
    return [col for col in frame.columns if col not in drop_columns]
