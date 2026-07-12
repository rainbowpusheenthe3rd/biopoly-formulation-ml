from __future__ import annotations

import numpy as np
import pytest

from biopoly import TARGETS
from biopoly.data.chemistry import Formulation, forward_true
from biopoly.data.schema import FEATURE_COLS

pytestmark = pytest.mark.layer(3)  # data generation


def test_schema_columns_present(small_df):
    for col in FEATURE_COLS + TARGETS:
        assert col in small_df.columns


def test_fraction_mass_balance(small_df):
    frac_cols = [c for c in small_df.columns if c.startswith("frac_")]
    add_cols = [c for c in small_df.columns if c.startswith("add_")]
    total = small_df[frac_cols].sum(axis=1) + small_df[add_cols].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_structured_missingness(small_df):
    # biodegradation and clarity are deliberately missing-not-at-random
    assert small_df["biodegradation_60d_pct"].isna().mean() > 0.1
    assert small_df["optical_clarity_pct"].isna().mean() > 0.1
    # other targets are always measured
    assert small_df["tensile_strength_mpa"].isna().sum() == 0


def test_targets_within_physical_bounds(small_df):
    assert small_df["biodegradation_60d_pct"].max() <= 100.0
    assert small_df["optical_clarity_pct"].max() <= 99.0
    assert (small_df[TARGETS].min(numeric_only=True) >= 0).all()


def test_supplier_shift_raises_pbs_mfi():
    # same PBS-rich formulation, before vs after the purity shift
    form = Formulation({"PBS": 1.0}, {}, 160.0, 20.0)
    before = forward_true(form, after_supplier_shift=False)
    after = forward_true(form, after_supplier_shift=True)
    assert after["melt_flow_index_g10min"] > before["melt_flow_index_g10min"]
    assert after["tensile_strength_mpa"] < before["tensile_strength_mpa"]
