"""L8 — drift detection: the mid-2025 supplier shift must be caught, and only it."""

from __future__ import annotations

import pytest

from biopoly.config import Settings
from biopoly.data.generate import build_dataset
from biopoly.monitoring.drift import detect_drift

pytestmark = pytest.mark.layer(8)  # drift detection


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(Settings(n_samples=1500, seed=13))


def test_supplier_shift_triggers_alert(dataset):
    ref = dataset[dataset.supplier_batch == "S1"]
    cur = dataset[dataset.supplier_batch == "S2"]
    report = detect_drift(
        ref, cur, ["melt_flow_index_g10min", "tensile_strength_mpa", "frac_PBS"]
    )
    assert report["alert"] is True


def test_no_alert_on_same_distribution(dataset):
    ref = dataset[dataset.supplier_batch == "S1"]
    half = len(ref) // 2
    a, b = ref.iloc[:half], ref.iloc[half:]
    report = detect_drift(a, b, ["process_temp_c", "process_time_min"])
    assert report["alert"] is False
