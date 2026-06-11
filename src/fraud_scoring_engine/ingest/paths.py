"""Path resolution for IEEE-CIS raw data files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IeeeDataPaths:
    """Paths to train CSV files under ``data/raw/``."""

    data_dir: Path
    train_transaction: Path
    train_identity: Path


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def ieee_data_paths(data_dir: Path | None = None) -> IeeeDataPaths:
    """Build paths to IEEE train transaction and identity CSVs.

    Args:
        data_dir: Optional override for the raw data directory.
            Defaults to ``{repo_root}/data/raw``.

    Returns:
        Resolved paths for train transaction and identity files.
    """
    root = data_dir if data_dir is not None else repo_root() / "data" / "raw"
    return IeeeDataPaths(
        data_dir=root,
        train_transaction=root / "train_transaction.csv",
        train_identity=root / "train_identity.csv",
    )
