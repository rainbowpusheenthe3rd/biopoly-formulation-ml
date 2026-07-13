"""MLflow-backed tracker (the default).

Runs against a local SQLite store by default (``settings.mlflow_tracking_uri``),
which is enough to browse runs in the MLflow UI with zero server setup:
``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from biopoly.config import settings


class MLflowTracker:
    """Experiment tracker backed by MLflow (local SQLite store by default)."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.experiment_name)

    def __enter__(self) -> MLflowTracker:
        self._run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, *exc: object) -> None:
        mlflow.end_run()

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(flat, step=step)

    def log_artifacts(self, path: str | Path) -> None:
        p = Path(path)
        if p.is_dir():
            mlflow.log_artifacts(str(p))
        else:
            mlflow.log_artifact(str(p))
