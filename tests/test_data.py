from __future__ import annotations

import numpy as np
import pytest

from biopoly import TARGETS
from biopoly.config import Settings
from biopoly.data.chemistry import Formulation, forward_true
from biopoly.data.generate import build_dataset
from biopoly.data.schema import FEATURE_COLS, SIGNAL_FEATURES

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


def test_feedstock_quality_raises_tensile():
    # purer feedstock -> stronger polymer (monotone, learnable covariate)
    form = Formulation({"PLA": 1.0}, {}, 200.0, 20.0)
    low = forward_true(form, feedstock_quality=0.9)
    high = forward_true(form, feedstock_quality=1.1)
    assert high["tensile_strength_mpa"] > low["tensile_strength_mpa"]


def test_feedstock_quality_column_wired(small_df):
    # the seasonal signal is present as a real, varying covariate in the dataset
    q = small_df["feedstock_quality"]
    assert q.notna().all()
    assert ((q >= 0.5) & (q <= 1.5)).all()
    assert q.std() > 0.0


def test_crystallinity_lowers_clarity():
    # the realized-crystallinity latent drives haze: higher crystallinity -> lower clarity
    form = Formulation({"PLA": 1.0}, {}, 200.0, 20.0)
    low = forward_true(form, crystallinity=0.85)
    high = forward_true(form, crystallinity=1.15)
    assert high["optical_clarity_pct"] < low["optical_clarity_pct"]


def test_signal_features_added_when_requested():
    df = build_dataset(Settings(n_samples=150, seed=3), with_signal_features=True)
    for col in SIGNAL_FEATURES:
        assert col in df.columns
    assert df["dsc_max_height"].std() > 0.0  # a real, varying signal feature


def test_real_seed_loads_and_anchors_synthetic():
    from biopoly.data.real_seed import REAL_PROPERTIES, load_real_seed, synthetic_vs_real

    seed = load_real_seed()
    assert len(seed) >= 5
    assert {"polymer", *REAL_PROPERTIES} <= set(seed.columns)
    cmp = synthetic_vs_real()
    assert not cmp.empty
    # the synthetic ground truth sits close to the literature seed (loose, honest bound)
    assert cmp["abs_pct_gap"].median() < 25.0
