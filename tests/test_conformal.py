from __future__ import annotations

import numpy as np
import pytest

from biopoly import TARGETS
from biopoly.config import Settings
from biopoly.data.generate import build_dataset
from biopoly.features import split
from biopoly.models.conformal import ConformalCalibrator
from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import evaluate

pytestmark = pytest.mark.layer(6)  # calibrated uncertainty (CQR)


def _fit_with_calibration(seed: int = 11):
    """Train on one split and calibrate on a disjoint one — as biopoly-train does."""
    df = build_dataset(Settings(n_samples=1200, seed=seed))
    train_df, test_df = split(df, test_size=0.25, mode="random", seed=seed)
    fit_df, cal_df = split(train_df, test_size=0.25, mode="random", seed=seed + 1)
    model = ForwardModel({"n_estimators": 200, "learning_rate": 0.06}).fit(fit_df)
    calibrator = ConformalCalibrator(alpha=0.20).fit(cal_df, model.predict(cal_df))
    return model, calibrator, test_df


def test_apply_preserves_point_and_interval_validity():
    model, calibrator, test_df = _fit_with_calibration()
    raw = model.predict(test_df)
    cal = calibrator.apply(raw)
    for t in TARGETS:
        # point estimate untouched, and stays inside the calibrated band
        assert np.allclose(cal[t]["value"], raw[t]["value"])
        assert np.all(cal[t]["p10"] <= cal[t]["value"] + 1e-9)
        assert np.all(cal[t]["value"] <= cal[t]["p90"] + 1e-9)


def test_conformal_coverage_near_nominal():
    """CQR should pull held-out coverage close to the 80% nominal level.

    We assert a generous lower bound (>=0.72): the finite-sample guarantee is
    marginal and the test set is small, but conformalising should never leave us
    badly *under* the target the way an uncalibrated quantile band can.
    """
    model, calibrator, test_df = _fit_with_calibration()
    model.conformal_ = calibrator
    metrics = evaluate(test_df, model.predict(test_df))
    for t in TARGETS:
        assert metrics[t]["interval_coverage"] >= 0.72, (t, metrics[t]["interval_coverage"])


def test_calibrator_attaches_to_model_predict():
    model, calibrator, test_df = _fit_with_calibration()
    before = model.predict(test_df)
    model.conformal_ = calibrator
    after = model.predict(test_df)
    # attaching the calibrator changes the served bands (unless Q happened to be 0)
    changed = any(
        not np.allclose(before[t]["p90"] - before[t]["p10"], after[t]["p90"] - after[t]["p10"])
        for t in TARGETS
    )
    assert changed
