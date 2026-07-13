"""L9 — active learning: query-by-committee acquisition and the AL loop."""

from __future__ import annotations

import numpy as np
import pytest

from biopoly.active_learning import (
    Committee,
    active_learning_curve,
    active_learning_shift_curve,
    propose_experiment,
)
from biopoly.config import Settings
from biopoly.data.generate import build_dataset

pytestmark = pytest.mark.layer(9)  # inverse design & API family (search + forward model)


def test_disagreement_shape_and_nonneg(small_df):
    committee = Committee.fit(small_df, k=3, seed=0)
    acq = committee.disagreement(small_df.head(20))
    assert acq.shape == (20,)
    assert np.all(acq >= 0.0)


def test_committee_mean_predicts_all_targets(small_df):
    committee = Committee.fit(small_df, k=3, seed=1)
    pred = committee.predict_mean(small_df.head(5))
    from biopoly import TARGETS

    assert set(pred) == set(TARGETS)
    assert all(len(pred[t]) == 5 for t in TARGETS)


def test_propose_experiment_returns_valid_recipe(small_df):
    committee = Committee.fit(small_df, k=3, seed=2)
    out = propose_experiment(committee, n_candidates=200, top_k=2, seed=3)
    assert len(out) == 2
    assert out[0]["information_gain"] >= out[1]["information_gain"]  # ranked, best first
    assert out[0]["formulation"]["polymers"]  # a non-empty formulation


def test_active_learning_curve_improves_with_data(small_df):
    curves = active_learning_curve(small_df, seed_size=50, batch=40, rounds=3, k=3, seed=0)
    assert len(curves["active"]) == len(curves["labels"]) == 4
    assert curves["labels"][-1] > curves["labels"][0]  # the labelled set grows
    assert curves["active"][-1] > curves["active"][0]  # more data -> better model


def test_shift_curve_adapts_to_new_regime():
    # seed pre-shift (S1), test post-shift (S2): more labels -> better on the new regime
    df = build_dataset(Settings(n_samples=700, seed=5))
    curves = active_learning_shift_curve(
        df,
        seed_size=30,
        batch=20,
        rounds=3,
        k=3,
        params={"n_estimators": 60, "learning_rate": 0.1},
        seed=0,
    )
    assert len(curves["active"]) == len(curves["labels"]) == 4
    assert curves["labels"][-1] > curves["labels"][0]
    assert curves["active"][-1] > curves["active"][0]
