"""L9 — model registry: calibration-aware register-if-better promotion gate."""

from __future__ import annotations

import pytest

from biopoly.models.registry import ModelRegistry

pytestmark = pytest.mark.layer(9)  # inverse design & API / serving & ops


def _model_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "forward_model.joblib").write_text("stub")  # register only copies files
    return d


def _metrics(coverage: float):
    targets = ("tensile_strength_mpa", "optical_clarity_pct")
    return {t: {"interval_coverage": coverage} for t in targets}


def test_higher_r2_but_broken_coverage_is_rejected(tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    v1, p1 = reg.register_if_better(_model_dir(tmp_path, "m1"), _metrics(0.80), mean_r2=0.90)
    assert p1 is True  # first registration always promotes
    # more accurate (0.92) but its intervals now cover only 50% -> must NOT take the crown
    _v2, p2 = reg.register_if_better(_model_dir(tmp_path, "m2"), _metrics(0.50), mean_r2=0.92)
    assert p2 is False
    assert reg.champion() == v1


def test_better_calibration_wins_at_equal_accuracy(tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register_if_better(_model_dir(tmp_path, "a"), _metrics(0.60), mean_r2=0.90)  # poor coverage
    v2, promoted = reg.register_if_better(_model_dir(tmp_path, "b"), _metrics(0.80), mean_r2=0.90)
    assert promoted is True  # same R2, better calibrated -> promote
    assert reg.champion() == v2
