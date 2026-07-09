"""A small, dependency-free model registry (versions + champion pointer).

Maps conceptually onto MLflow Model Registry / SageMaker Model Registry (register ->
promote -> roll back) but is filesystem-backed so it runs anywhere with no server.
Each version stores the serialised model, its metrics and the git commit that made it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biopoly.config import settings
from biopoly.models.forward import ForwardModel


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class ModelRegistry:
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
        """Register a candidate; promote it only if it beats the current champion."""
        version = self.register(model_dir, metrics, mean_r2)
        champ = self.champion()
        promoted = False
        if champ is None or mean_r2 > self.metadata(champ)["mean_r2"]:
            self.promote(version)
            promoted = True
        return version, promoted
