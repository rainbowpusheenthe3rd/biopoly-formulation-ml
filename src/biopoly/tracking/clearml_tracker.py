"""ClearML-backed tracker (optional swap).

ClearML decorators wrap the codebase so every run records its metrics, machine
type, environment and commit. Requires the ``clearml``
extra and a configured ClearML account/agent:

    uv sync --extra clearml
    BIOPOLY_TRACKING_BACKEND=clearml uv run biopoly-train
"""

from __future__ import annotations

from pathlib import Path

from biopoly.config import settings


class ClearMLTracker:
    """Experiment tracker backed by ClearML (needs the optional ``clearml`` extra)."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def __enter__(self) -> ClearMLTracker:
        from clearml import Task  # imported lazily so the extra is optional

        self.task = Task.init(project_name=settings.experiment_name, task_name=self.run_name)
        self.logger = self.task.get_logger()
        return self

    def __exit__(self, *exc: object) -> None:
        self.task.close()

    def log_params(self, params: dict) -> None:
        self.task.connect(dict(params))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.logger.report_single_value(k, float(v))

    def log_artifacts(self, path: str | Path) -> None:
        self.task.upload_artifact(Path(path).name, artifact_object=str(path))
