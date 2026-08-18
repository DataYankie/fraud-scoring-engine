"""Tests for fraud_scoring_engine.training.mlflow_utils."""

from unittest.mock import MagicMock, patch

import pytest

from fraud_scoring_engine.training.mlflow_utils import setup_mlflow


@patch("fraud_scoring_engine.training.mlflow_utils.mlflow.set_experiment")
@patch("fraud_scoring_engine.training.mlflow_utils.mlflow.set_tracking_uri")
@patch("fraud_scoring_engine.training.mlflow_utils.MlflowClient")
def test_setup_mlflow_creates_experiment_when_missing(
    mock_client_cls: MagicMock,
    mock_set_tracking_uri: MagicMock,
    mock_set_experiment: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "mlartifacts"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql+psycopg://user:pass@localhost:5432/mlflow_db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(artifact_root))

    mock_client = mock_client_cls.return_value
    mock_client.get_experiment_by_name.return_value = None

    tracking_uri, artifact_root_uri = setup_mlflow("xgboost-fraud")

    mock_set_tracking_uri.assert_called_once_with("postgresql+psycopg://user:pass@localhost:5432/mlflow_db")
    mock_client.create_experiment.assert_called_once_with("xgboost-fraud", artifact_location=artifact_root_uri)
    mock_set_experiment.assert_called_once_with("xgboost-fraud")
    assert tracking_uri == "postgresql+psycopg://user:pass@localhost:5432/mlflow_db"
    assert artifact_root_uri == artifact_root.resolve().as_uri()


@patch("fraud_scoring_engine.training.mlflow_utils.mlflow.set_experiment")
@patch("fraud_scoring_engine.training.mlflow_utils.mlflow.set_tracking_uri")
@patch("fraud_scoring_engine.training.mlflow_utils.MlflowClient")
def test_setup_mlflow_reuses_existing_experiment(
    mock_client_cls: MagicMock,
    mock_set_tracking_uri: MagicMock,
    mock_set_experiment: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "mlartifacts"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql+psycopg://user:pass@localhost:5432/mlflow_db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(artifact_root))

    mock_client = mock_client_cls.return_value
    mock_client.get_experiment_by_name.return_value = object()

    setup_mlflow("xgboost-fraud")

    mock_client.create_experiment.assert_not_called()
    mock_set_experiment.assert_called_once_with("xgboost-fraud")
