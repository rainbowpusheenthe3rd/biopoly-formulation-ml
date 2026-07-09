"""Experiment-tracking abstraction.

A thin ``ExperimentTracker`` protocol so the training code never imports a specific
backend. ClearML and MLflow are both supported; this lets either be plugged in
(or a no-op used in CI) via ``BIOPOLY_TRACKING_BACKEND``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    def __enter__(self) -> ExperimentTracker: ...
    def __exit__(self, *exc: object) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_artifacts(self, path: str | Path) -> None: ...


class NoOpTracker:
    """Used in tests/CI where no tracking server is desired."""

    def __init__(self, run_name: str = "", **_: object) -> None:
        self.run_name = run_name

    def __enter__(self) -> NoOpTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def log_params(self, params: dict) -> None:
        return None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        return None

    def log_artifacts(self, path: str | Path) -> None:
        return None


def get_tracker(run_name: str, *, backend: str | None = None) -> ExperimentTracker:
    """Factory selecting a tracker by settings/backend name."""
    from biopoly.config import settings

    backend = (backend or settings.tracking_backend).lower()
    if backend == "mlflow":
        from biopoly.tracking.mlflow_tracker import MLflowTracker

        return MLflowTracker(run_name)
    if backend == "clearml":
        from biopoly.tracking.clearml_tracker import ClearMLTracker

        return ClearMLTracker(run_name)
    return NoOpTracker(run_name)
