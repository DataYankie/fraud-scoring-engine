"""Resolve raw and processed file paths for the IEEE-CIS pipeline.

This module centralizes repository-relative path construction for raw Kaggle
CSV inputs and generated Parquet outputs so scripts and library code use the
same default locations.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IeeeDataPaths:
    """Paths to train CSV files under ``data/raw/``."""

    data_dir: Path
    train_transaction: Path
    train_identity: Path
    processed_dir: Path
    train_features_parquet: Path


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def ieee_data_paths(
    data_dir: Path | None = None,
    *,
    processed_dir: Path | None = None,
    parquet_path: Path | None = None,
) -> IeeeDataPaths:
    """Build paths to IEEE train CSVs and the processed Parquet artifact.

    Args:
        data_dir: Optional override for the raw data directory.
            Defaults to ``{repo_root}/data/raw``.
        processed_dir: Optional override for ``data/processed``.
        parquet_path: Optional override for the train features Parquet file.

    Returns:
        Resolved paths for train transaction, identity, and feature files.
    """
    root = data_dir if data_dir is not None else repo_root() / "data" / "raw"
    processed = processed_dir if processed_dir is not None else repo_root() / "data" / "processed"
    features = parquet_path if parquet_path is not None else processed / "train_features.parquet"
    return IeeeDataPaths(
        data_dir=root,
        train_transaction=root / "train_transaction.csv",
        train_identity=root / "train_identity.csv",
        processed_dir=processed,
        train_features_parquet=features,
    )
