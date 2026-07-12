"""L2 — domain ground truth: the physics-informed chemistry must behave sanely."""

from __future__ import annotations

import numpy as np
import pytest

from biopoly.data.chemistry import (
    POLYMERS,
    PROC_TEMP_OPT,
    Formulation,
    blend_optimum_temp,
    forward_true,
)

pytestmark = pytest.mark.layer(2)  # domain ground truth — chemistry


def _neat(polymer: str, *, temp: float | None = None, additives: dict | None = None) -> Formulation:
    return Formulation(
        {polymer: 1.0}, additives or {}, temp if temp is not None else PROC_TEMP_OPT[polymer], 20.0
    )


def test_outputs_always_within_physical_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 4))
        chosen = rng.choice(POLYMERS, size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        form = Formulation(
            {p: float(f) for p, f in zip(chosen, w, strict=True)},
            {},
            float(rng.uniform(90, 255)),
            float(rng.uniform(4, 110)),
        )
        out = forward_true(form)
        assert 0.5 <= out["tensile_strength_mpa"] <= 80.0
        assert 0.0 <= out["biodegradation_60d_pct"] <= 100.0
        assert 0.0 <= out["optical_clarity_pct"] <= 95.0


def test_plasticizer_softens_tensile():
    base = forward_true(_neat("PLA"))["tensile_strength_mpa"]
    plast = forward_true(_neat("PLA", additives={"plasticizer": 0.15}))["tensile_strength_mpa"]
    assert plast < base


def test_processing_at_optimum_maximises_tensile():
    opt = PROC_TEMP_OPT["PLA"]
    at_opt = forward_true(_neat("PLA", temp=opt))["tensile_strength_mpa"]
    too_hot = forward_true(_neat("PLA", temp=opt + 40))["tensile_strength_mpa"]
    too_cold = forward_true(_neat("PLA", temp=opt - 40))["tensile_strength_mpa"]
    assert at_opt > too_hot and at_opt > too_cold


def test_blend_optimum_temp_lies_between_components():
    t = blend_optimum_temp({"PLA": 0.5, "PCL": 0.5})
    assert PROC_TEMP_OPT["PCL"] < t < PROC_TEMP_OPT["PLA"]


def test_compatibilizer_improves_immiscible_blend_clarity():
    blend = {"PLA": 0.5, "TPS": 0.5}  # strongly immiscible pair
    without = forward_true(Formulation(blend, {}, 180.0, 20.0))["optical_clarity_pct"]
    withc = forward_true(
        Formulation(blend, {"compatibilizer": 0.05}, 180.0, 20.0)
    )["optical_clarity_pct"]
    assert withc >= without
