from __future__ import annotations

import numpy as np
import pytest

from biopoly.inverse import baseline, bayesopt

pytestmark = pytest.mark.layer(9)  # inverse design & API


def test_baseline_returns_ranked_candidates(fast_model):
    target = {"tensile_strength_mpa": 40.0, "optical_clarity_pct": 70.0}
    res = baseline.design(fast_model, target, n_candidates=500, top_k=3, seed=0)
    assert len(res) == 3
    scores = [r["score"] for r in res]
    assert scores == sorted(scores)  # best (lowest) first
    assert all(np.isfinite(r["score"]) for r in res)


def test_bayesopt_hits_achievable_target(fast_model):
    # a PLA-like spec is achievable; warm-started TPE should get tensile close
    target = {"tensile_strength_mpa": 50.0, "optical_clarity_pct": 80.0}
    res = bayesopt.design(fast_model, target, n_trials=200, top_k=1, seed=1)
    best = res[0]
    # weak session model on 400 rows: require a reasonably close hit, not exact
    assert abs(best["predicted"]["tensile_strength_mpa"] - 50.0) < 12.0


def test_bayesopt_beats_or_matches_random(fast_model):
    target = {"tensile_strength_mpa": 45.0, "biodegradation_60d_pct": 40.0}
    b = baseline.design(fast_model, target, n_candidates=400, top_k=1, seed=2)[0]["score"]
    bo = bayesopt.design(fast_model, target, n_trials=200, top_k=1, seed=2)[0]["score"]
    assert bo <= b + 0.1  # optimisation should not be meaningfully worse than sampling
