"""L4 — features & splitting: the model-input contract and train/test discipline."""

from __future__ import annotations

import pytest

from biopoly.data.schema import CATEGORICAL_FEATURES, FEATURE_COLS
from biopoly.features import make_x, split

pytestmark = pytest.mark.layer(4)  # features & splitting


def test_make_x_matches_feature_contract(small_df):
    x = make_x(small_df)
    assert list(x.columns) == FEATURE_COLS
    for col in CATEGORICAL_FEATURES:
        assert str(x[col].dtype) == "category"


def test_random_split_partitions_disjointly(small_df):
    tr, te = split(small_df, test_size=0.2, mode="random", seed=1)
    assert len(tr) + len(te) == len(small_df)
    assert abs(len(te) / len(small_df) - 0.2) < 0.02
    assert set(tr.index).isdisjoint(set(te.index))


def test_temporal_split_is_time_ordered(small_df):
    tr, te = split(small_df, test_size=0.2, mode="temporal")
    assert tr["date"].max() <= te["date"].min()
