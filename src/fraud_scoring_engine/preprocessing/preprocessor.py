"""Sklearn-compatible preprocessing for fraud model training and scoring."""

from __future__ import annotations

from typing import Self, Any

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_scoring_engine.preprocessing.columns import (
    DROP_COLUMNS,
    MISSING_CATEGORY,
    feature_columns,
)


def _is_categorical_dtype(dtype: Any) -> bool:
    """Return whether ``dtype`` should be treated as categorical.

    Args:
        dtype: Pandas dtype object for a column.

    Returns:
        True for categorical, object, or string dtypes; False for bool/numeric.
    """
    if isinstance(dtype, CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return False
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def _normalize_categorical_value(value: object) -> str:
    """Map a categorical cell to a stable string category label.

    Args:
        value: Raw cell value from a categorical column.

    Returns:
        ``MISSING_CATEGORY`` for ``None``, ``NaN``, or ``pd.NA``; otherwise
        ``str(value)``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return MISSING_CATEGORY
    return str(value)


class FraudFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Prepare merged IEEE + behavioral frames for XGBoost with native categoricals.

    Numeric columns are passed through with ``NaN`` preserved so tree models can
    learn missing-value splits. Low-cardinality string columns are mapped to
    pandas ``category`` dtypes with a stable ``__MISSING__`` level learned on
    the training split.

    Args:
        categorical_columns: Explicit categorical names. When ``None``, object
            and category columns in the training frame are detected automatically.
        numeric_columns: Explicit numeric names. When ``None``, all non-categorical
            feature columns are treated as numeric passthrough.
        drop_columns: Metadata columns removed before modeling.
    """

    def __init__(
        self,
        *,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        drop_columns: frozenset[str] = DROP_COLUMNS,
    ) -> None:
        """Store column selection options for later ``fit`` / ``transform``."""
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.drop_columns = drop_columns

    def fit(self, X: pd.DataFrame, y: object | None = None) -> Self:
        """Learn categorical vocabularies and output column order from ``X``.

        Args:
            X: Training feature frame.
            y: Ignored; present for sklearn estimator compatibility.

        Returns:
            Fitted ``self`` with ``feature_columns_``, ``numeric_columns_``,
            ``categorical_columns_``, and ``categories_`` set.
        """
        del y
        frame = self._validate_frame(X)
        available = feature_columns(frame, drop_columns=self.drop_columns)

        categorical = self._resolve_categorical_columns(frame, available)
        numeric = self._resolve_numeric_columns(frame, available, categorical)

        self.feature_columns_: list[str] = numeric + categorical
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        self.categories_: dict[str, list[str]] = {}

        for column in categorical:
            normalized = frame[column].map(_normalize_categorical_value)
            categories = sorted(normalized.unique(), key=str)
            if MISSING_CATEGORY not in categories:
                categories.insert(0, MISSING_CATEGORY)
            self.categories_[column] = categories

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a model-ready feature matrix with stable column order.

        Args:
            X: Feature frame with all fitted feature columns present.

        Returns:
            DataFrame with numeric passthrough columns and pandas categorical
            columns in ``feature_columns_`` order.

        Raises:
            AttributeError: If ``fit`` has not been called.
            TypeError: If ``X`` is not a pandas DataFrame.
            ValueError: If ``X`` is missing any fitted feature columns.
        """
        if not hasattr(self, "feature_columns_"):
            msg = "FraudFeaturePreprocessor has not been fitted yet."
            raise AttributeError(msg)

        frame = self._validate_frame(X)
        missing = [col for col in self.feature_columns_ if col not in frame.columns]
        if missing:
            msg = f"Input frame is missing fitted feature columns: {missing[:5]}"
            raise ValueError(msg)

        numeric_parts: list[pd.Series] = []
        for column in self.numeric_columns_:
            numeric_parts.append(
                pd.to_numeric(frame[column], errors="coerce").rename(column)
            )

        categorical_parts: list[pd.Series] = []
        for column in self.categorical_columns_:
            categories = self.categories_[column]
            known = set(categories)
            normalized = frame[column].map(_normalize_categorical_value)
            normalized = normalized.where(normalized.isin(known), MISSING_CATEGORY)
            categorical_parts.append(
                normalized.astype(CategoricalDtype(categories=categories)).rename(column)
            )

        if not numeric_parts and not categorical_parts:
            return pd.DataFrame(index=frame.index)

        return pd.concat(numeric_parts + categorical_parts, axis=1)[self.feature_columns_]

    def get_feature_names_out(self, input_features: object | None = None) -> list[str]:
        """Return output feature names for sklearn pipeline compatibility.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            Fitted output column names in transform order.

        Raises:
            AttributeError: If ``fit`` has not been called.
        """
        del input_features
        if not hasattr(self, "feature_columns_"):
            msg = "FraudFeaturePreprocessor has not been fitted yet."
            raise AttributeError(msg)
        return list(self.feature_columns_)

    def _validate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure the input is a pandas DataFrame.

        Args:
            X: Candidate feature input.

        Returns:
            ``X`` unchanged when it is a DataFrame.

        Raises:
            TypeError: If ``X`` is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            msg = "FraudFeaturePreprocessor expects a pandas DataFrame."
            raise TypeError(msg)
        return X

    def _resolve_categorical_columns(
        self,
        frame: pd.DataFrame,
        available: list[str],
    ) -> list[str]:
        """Choose categorical feature columns for this fit.

        Args:
            frame: Training feature frame.
            available: Feature column names after dropping metadata columns.

        Returns:
            Explicit ``categorical_columns`` intersected with ``available``, or
            auto-detected categorical/object/string columns when unset.
        """
        if self.categorical_columns is not None:
            return [col for col in self.categorical_columns if col in available]

        return [
            col
            for col in available
            if col in frame.columns and _is_categorical_dtype(frame[col].dtype)
        ]

    def _resolve_numeric_columns(
        self,
        frame: pd.DataFrame,
        available: list[str],
        categorical: list[str],
    ) -> list[str]:
        """Choose numeric passthrough feature columns for this fit.

        Args:
            frame: Training feature frame (unused when columns are explicit).
            available: Feature column names after dropping metadata columns.
            categorical: Resolved categorical column names to exclude.

        Returns:
            Explicit ``numeric_columns`` intersected with ``available``, or the
            remaining non-categorical columns in ``available`` when unset.
        """
        if self.numeric_columns is not None:
            return [col for col in self.numeric_columns if col in available]

        categorical_set = set(categorical)
        return [col for col in available if col not in categorical_set]
