"""L1 — foundations: the schema and config contracts every other layer rests on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from biopoly import TARGETS
from biopoly.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    TENSILE_PROTOCOLS,
    FormulationInput,
)
from biopoly.models.metrics import TOLERANCE

pytestmark = pytest.mark.layer(1)  # foundations — schema & config


def test_feature_cols_partition_cleanly():
    assert set(FEATURE_COLS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert not (set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES))
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))  # no duplicates


def test_tolerance_covers_every_target():
    assert set(TOLERANCE) == set(TARGETS)
    assert all(v > 0 for v in TOLERANCE.values())


def test_formulation_input_accepts_valid():
    m = FormulationInput(frac={"PLA": 0.8, "PBS": 0.2}, process_temp_c=195, process_time_min=20)
    assert m.tensile_protocol in TENSILE_PROTOCOLS


def test_formulation_input_rejects_unknown_polymer():
    with pytest.raises(ValidationError):
        FormulationInput(frac={"XYZ": 1.0}, process_temp_c=195, process_time_min=20)


def test_formulation_input_rejects_empty_fractions():
    with pytest.raises(ValidationError):
        FormulationInput(frac={}, process_temp_c=195, process_time_min=20)
