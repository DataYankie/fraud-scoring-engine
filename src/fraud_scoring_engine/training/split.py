"""Create time-ordered dataset splits for fraud model evaluation.

This module provides the sequential train, validation, and test split logic
used to preserve temporal ordering when preparing fraud modeling datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_TIME_COLUMN = "transaction_dt"
DEFAULT_TIEBREAKER_COLUMN = "transaction_id"
_RATIO_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test frames from a time-ordered split.

    Attributes:
        train: Earliest rows used for fitting.
        val: Middle rows used for early stopping and hyperparameter choice.
        test: Latest rows held out for final evaluation.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Return ``(train, val, test)`` row counts."""
        return (len(self.train), len(self.val), len(self.test))


def time_split(
    frame: pd.DataFrame,
    *,
    time_col: str = DEFAULT_TIME_COLUMN,
    tiebreaker_col: str | None = DEFAULT_TIEBREAKER_COLUMN,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> DatasetSplit:
    """Split ``frame`` into sequential train/val/test slices by time.

    Rows are sorted by ``time_col`` (then ``tiebreaker_col`` when provided)
    and cut into three contiguous blocks. Remaining rows after integer
    truncation go to the test split so no rows are dropped.

    Args:
        frame: Feature frame to split. Must contain ``time_col``.
        time_col: Timestamp or elapsed-seconds column used for ordering.
        tiebreaker_col: Optional second sort key for same-timestamp rows.
            Pass ``None`` to sort only by ``time_col``.
        ratios: ``(train, val, test)`` fractions. Must be positive and sum
            to 1.

    Returns:
        :class:`DatasetSplit` with copies of each slice.

    Raises:
        TypeError: If ``frame`` is not a pandas DataFrame.
        ValueError: If columns are missing, ratios are invalid, the frame
            is empty, or any resulting split would be empty.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = "time_split expects a pandas DataFrame."
        raise TypeError(msg)
    if frame.empty:
        msg = "Cannot split an empty DataFrame."
        raise ValueError(msg)

    _validate_ratios(ratios)
    _require_column(frame, time_col)
    sort_columns = [time_col]
    if tiebreaker_col is not None:
        _require_column(frame, tiebreaker_col)
        sort_columns.append(tiebreaker_col)

    ordered = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        msg = (
            f"Time split produced an empty partition for {n} rows with "
            f"ratios={ratios!r} (sizes={len(train)}, {len(val)}, {len(test)}). "
            "Use more rows or different ratios."
        )
        raise ValueError(msg)

    return DatasetSplit(train=train, val=val, test=test)


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    """Raise ``ValueError`` unless ratios are positive and sum to 1."""
    if len(ratios) != 3:
        msg = f"Expected 3 split ratios (train, val, test); got {len(ratios)}."
        raise ValueError(msg)
    if any(ratio <= 0 for ratio in ratios):
        msg = f"Split ratios must be positive; got {ratios!r}."
        raise ValueError(msg)
    total = sum(ratios)
    if abs(total - 1.0) > _RATIO_SUM_TOLERANCE:
        msg = f"Split ratios must sum to 1; got {total}."
        raise ValueError(msg)


def _require_column(frame: pd.DataFrame, column: str) -> None:
    """Raise ``ValueError`` if ``column`` is missing from ``frame``."""
    if column not in frame.columns:
        msg = f"Split column {column!r} is not in the DataFrame."
        raise ValueError(msg)
