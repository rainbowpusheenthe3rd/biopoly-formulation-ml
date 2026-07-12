"""Shared fixtures + the spiral-layered test harness.

Tests are organised as a **spiral of increasing complexity**: every test carries a
``@pytest.mark.layer(n)`` (1 = simplest foundations, 9 = full system). The suite
runs in layer order and reports a **complexity frontier** — the highest contiguous
layer that fully passes — so a break at layer *k* tells you the code is correct up
to, but not through, complexity *k*. That's the honest version of "the pass rate
reflects how far the implementation actually works", and it mirrors the spiral
syllabus the wider project teaches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from biopoly.config import Settings
from biopoly.data.generate import build_dataset
from biopoly.models.forward import ForwardModel

# ── spiral layers ────────────────────────────────────────────────────────────

LAYER_NAMES: dict[int, str] = {
    1: "foundations - schema & config",
    2: "domain ground truth - chemistry",
    3: "data generation",
    4: "features & splitting",
    5: "forward model learns signal",
    6: "calibrated uncertainty (CQR)",
    7: "time series & seasonality",
    8: "drift detection",
    9: "inverse design & API",
}

_RESULTS: dict[int, dict[str, int]] = {}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "layer(n): spiral complexity layer, 1 (simplest) .. 9 (full system)"
    )


def _item_layer(item: pytest.Item) -> int:
    marker = item.get_closest_marker("layer")
    return int(marker.args[0]) if marker else 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run tests in ascending layer order so the frontier is meaningful."""
    items.sort(key=_item_layer)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        rec = _RESULTS.setdefault(_item_layer(item), {"passed": 0, "failed": 0})
        if report.passed:
            rec["passed"] += 1
        elif report.failed:
            rec["failed"] += 1


def _frontier(results: dict[int, dict[str, int]]) -> int:
    """Highest layer N such that every layer 1..N fully passed (contiguous from 1)."""
    frontier = 0
    for layer in range(1, (max(results, default=0)) + 1):
        rec = results.get(layer)
        if rec and rec["failed"] == 0 and rec["passed"] > 0:
            frontier = layer
        else:
            break
    return frontier


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    layers = {k: v for k, v in _RESULTS.items() if k > 0}
    if not layers:
        return
    tr = terminalreporter
    tr.write_sep("=", "spiral complexity frontier")
    max_layer = max(layers)
    for layer in range(1, max_layer + 1):
        rec = layers.get(layer, {"passed": 0, "failed": 0})
        total = rec["passed"] + rec["failed"]
        status = "ok  " if (total and rec["failed"] == 0) else ("FAIL" if total else "----")
        name = LAYER_NAMES.get(layer, "")
        tr.write_line(f"  L{layer} [{status}] {rec['passed']}/{total}  {name}")
    frontier = _frontier(layers)
    tr.write_line(
        f"  frontier: L{frontier}/{max_layer}  "
        f"-> implementation is correct through the L{frontier} complexity layer"
    )
    # Machine-readable artifact for CI (badge / gate).
    try:
        out = Path(config.rootdir) / "artifacts" / "spiral_frontier.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "frontier": frontier,
                    "max_layer": max_layer,
                    "layers": {
                        str(k): layers.get(k, {"passed": 0, "failed": 0})
                        for k in range(1, max_layer + 1)
                    },
                },
                indent=2,
            )
        )
    except Exception:
        pass


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    cfg = Settings(n_samples=400, seed=7)
    return build_dataset(cfg)


@pytest.fixture(scope="session")
def fast_model(small_df) -> ForwardModel:
    return ForwardModel({"n_estimators": 120, "learning_rate": 0.08}).fit(small_df)
