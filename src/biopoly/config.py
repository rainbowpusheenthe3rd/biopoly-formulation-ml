"""Runtime settings (paths, seeds, dataset size, tracking backend).

Uses pydantic-settings so everything is overridable via ``BIOPOLY_*`` env vars,
e.g. ``BIOPOLY_TRACKING_BACKEND=clearml`` or ``BIOPOLY_N_SAMPLES=5000``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings (paths, seeds, dataset size, tracking), overridable via env."""

    model_config = SettingsConfigDict(env_prefix="BIOPOLY_", env_file=".env", extra="ignore")

    # Data
    n_samples: int = 2000
    seed: int = 42
    start_date: str = "2023-06-01"
    end_date: str = "2026-06-01"
    data_path: Path = ROOT / "data" / "formulations.parquet"

    # Artifacts / model store
    artifact_dir: Path = ROOT / "artifacts"
    model_name: str = "biopoly-forward"

    # Experiment tracking: "mlflow" | "clearml" | "noop"
    tracking_backend: str = "mlflow"
    # SQLite backend (the file store is deprecated in recent MLflow and also
    # cannot host the model registry). Browse with:
    #   uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow_tracking_uri: str = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    experiment_name: str = "biopoly-forward"

    # Drift monitoring
    drift_p_value: float = 0.01

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
