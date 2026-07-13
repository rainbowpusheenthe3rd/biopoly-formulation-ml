"""A small, dependency-free model registry (versions + champion pointer).

Maps conceptually onto MLflow Model Registry / SageMaker Model Registry (register ->
promote -> roll back) but is filesystem-backed so it runs anywhere with no server.
Each version stores the serialised model, its metrics and the git commit that made it.

Promotion is **calibration-aware**: a candidate must beat the champion on a combined
score that rewards accuracy (mean R²) and penalises interval mis-coverage. This
matters because the p10-p90 bands are conformally calibrated (CQR) — a model that is
marginally more accurate but whose intervals stop covering ~80% of the truth should
not silently become champion, and a purely calibration-improving model should be able
to win even at equal R².
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biopoly.config import settings
from biopoly.models.forward import ForwardModel

# p10-p90 nominal coverage, and how much a unit of coverage error costs vs a unit of R².
_COVERAGE_TARGET = 0.80
_COVERAGE_WEIGHT = 0.5


def _coverage_error(metrics: dict) -> float:
    """Mean absolute gap between interval coverage and its ~0.80 nominal, over targets."""
    errs = [
        abs(m["interval_coverage"] - _COVERAGE_TARGET)
        for m in metrics.values()
        if isinstance(m, dict) and "interval_coverage" in m
    ]
    return sum(errs) / len(errs) if errs else 0.0


def promotion_score(metrics: dict, mean_r2: float) -> float:
    """Accuracy rewarded, mis-coverage penalised: ``mean_r2 - w * coverage_error``."""
    return mean_r2 - _COVERAGE_WEIGHT * _coverage_error(metrics)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class ModelRegistry:
    """Filesystem-backed model registry: versioned models + a champion pointer."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or (settings.artifact_dir / "registry"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.champion_file = self.root / "champion.json"

    def _next_version(self) -> int:
        versions = [int(p.name[1:]) for p in self.root.glob("v*") if p.name[1:].isdigit()]
        return (max(versions) + 1) if versions else 1

    def list_versions(self) -> list[int]:
        return sorted(int(p.name[1:]) for p in self.root.glob("v*") if p.name[1:].isdigit())

    def register(self, model_dir: Path, metrics: dict, mean_r2: float) -> int:
        version = self._next_version()
        vdir = self.root / f"v{version}"
        if vdir.exists():
            shutil.rmtree(vdir)
        shutil.copytree(model_dir, vdir)
        meta = {
            "version": version,
            "created_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "mean_r2": mean_r2,
            "coverage_error": _coverage_error(metrics),
            "promotion_score": promotion_score(metrics, mean_r2),
            "metrics": metrics,
        }
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return version

    def metadata(self, version: int) -> dict:
        return json.loads((self.root / f"v{version}" / "metadata.json").read_text())

    def champion(self) -> int | None:
        if not self.champion_file.exists():
            return None
        return json.loads(self.champion_file.read_text())["version"]

    def promote(self, version: int) -> None:
        if not (self.root / f"v{version}").exists():
            raise ValueError(f"version v{version} does not exist")
        self.champion_file.write_text(json.dumps({"version": version}))

    def rollback(self) -> int | None:
        """Promote the most recent version *below* the current champion."""
        current = self.champion()
        below = [v for v in self.list_versions() if current is None or v < current]
        if not below:
            return None
        target = max(below)
        self.promote(target)
        return target

    def load_champion(self) -> ForwardModel:
        version = self.champion()
        if version is None:
            raise RuntimeError("no champion registered yet")
        return ForwardModel.load(self.root / f"v{version}")

    def register_if_better(
        self, model_dir: Path, metrics: dict, mean_r2: float
    ) -> tuple[int, bool]:
        """Register a candidate and promote it only if it wins.

        Promotion uses the calibration-aware :func:`promotion_score` (accuracy minus a
        coverage-error penalty), not mean R² alone.
        """
        version = self.register(model_dir, metrics, mean_r2)
        champ = self.champion()
        if champ is None:
            self.promote(version)
            return version, True
        champ_meta = self.metadata(champ)
        champ_score = champ_meta.get("promotion_score")
        if champ_score is None:  # older metadata without the field
            champ_score = promotion_score(champ_meta.get("metrics", {}), champ_meta["mean_r2"])
        promoted = promotion_score(metrics, mean_r2) > champ_score
        if promoted:
            self.promote(version)
        return version, promoted
