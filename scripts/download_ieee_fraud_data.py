"""Download IEEE-CIS Fraud Detection competition data into data/raw/.

Usage:
    uv run python scripts/download_ieee_fraud_data.py

Requires Kaggle credentials via KAGGLE_API_TOKEN or ~/.kaggle/access_token.
You must join and accept the competition rules at:
https://www.kaggle.com/competitions/ieee-fraud-detection
"""

from pathlib import Path

import kagglehub

COMPETITION = "ieee-fraud-detection"
EXPECTED_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(data_dir),
        force_download=True,
    )
    print(f"Downloaded to: {path}")

    missing = [name for name in EXPECTED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Expected files missing in {data_dir}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
