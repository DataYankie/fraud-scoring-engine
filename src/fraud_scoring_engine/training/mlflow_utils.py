"""Configure MLflow tracking and artifact storage for experiments.

This module encapsulates the project MLflow setup, including tracking URI
resolution, artifact root creation, and experiment initialization for training
runs.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from fraud_scoring_engine.config import get_mlflow_artifact_root, get_mlflow_tracking_uri


def setup_mlflow(experiment_name: str) -> tuple[str, str]:
    """Configure MLflow to use PostgreSQL tracking and a local artifact root.

    Creates the experiment when missing and sets it as the active experiment.

    Args:
        experiment_name: MLflow experiment name.

    Returns:
        ``(tracking_uri, artifact_root_uri)`` used for the session.
    """
    tracking_uri = get_mlflow_tracking_uri()
    artifact_root = get_mlflow_artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root_uri = artifact_root.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_root_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri, artifact_root_uri
