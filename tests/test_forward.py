from __future__ import annotations

import numpy as np
import pytest

from biopoly import TARGETS
from biopoly.models.metrics import evaluate, summary_row

pytestmark = pytest.mark.layer(5)  # forward model learns signal


def test_predict_shape_and_quantile_order(fast_model, small_df):
    preds = fast_model.predict(small_df)
    for t in TARGETS:
        assert set(preds[t]) == {"p10", "value", "p90"}
        assert len(preds[t]["value"]) == len(small_df)
        # p10 <= value <= p90 everywhere (quantile crossing repaired)
        assert np.all(preds[t]["p10"] <= preds[t]["value"] + 1e-9)
        assert np.all(preds[t]["value"] <= preds[t]["p90"] + 1e-9)


def test_learns_signal(fast_model, small_df):
    # in-sample fit should be strong: the data has real structure to learn
    metrics = evaluate(small_df, fast_model.predict(small_df))
    assert summary_row(metrics) > 0.7


def test_feature_importance_sums_to_one(fast_model):
    imp = fast_model.feature_importances()
    for t in TARGETS:
        assert abs(sum(imp[t].values()) - 1.0) < 1e-6


def test_process_temp_drives_tensile(fast_model):
    # domain sanity: processing temperature is a top-3 driver of tensile strength
    imp = fast_model.feature_importances()["tensile_strength_mpa"]
    top3 = sorted(imp, key=imp.get, reverse=True)[:3]
    assert "process_temp_c" in top3
