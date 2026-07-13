"""Forward model: formulation -> predicted properties, with uncertainty.

Gradient-boosted trees (LightGBM), chosen deliberately:
they train in seconds on ~2k rows, handle mixed categorical/numeric inputs and
missing values natively, and expose feature importance so the scientist gets a
*readable* model, not a black box.

Uncertainty comes from quantile regression: for each target we fit p10 / p50 / p90
boosters, so every prediction ships with an interval a scientist can weigh.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from biopoly import TARGETS
from biopoly.data.schema import CATEGORICAL_FEATURES, FEATURE_COLS
from biopoly.features import make_x

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

_DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbosity=-1,
)


class ForwardModel:
    """Multi-output quantile forward model (one LightGBM booster per target×quantile)."""

    def __init__(self, params: dict | None = None, feature_cols: list[str] | None = None):
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.models_: dict[str, dict[str, LGBMRegressor]] = {}
        # Feature contract; defaults to schema.FEATURE_COLS. A superset (e.g. with the
        # DSC signal features) is used for the with-vs-without ablation.
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
        # Optional CQR calibrator; when set, predict() returns calibrated bands.
        # Attached by biopoly-train after fitting on a held-out calibration split.
        self.conformal_ = None

    def fit(self, df: pd.DataFrame) -> ForwardModel:
        x_all = make_x(df, self.feature_cols)
        for target in TARGETS:
            mask = df[target].notna().to_numpy()
            x, y = x_all[mask], df.loc[mask, target]
            self.models_[target] = {}
            for name, alpha in QUANTILES.items():
                model = LGBMRegressor(objective="quantile", alpha=alpha, **self.params)
                model.fit(x, y, categorical_feature=CATEGORICAL_FEATURES)
                self.models_[target][name] = model
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        """Return ``{target: {"value": .., "p10": .., "p90": ..}}`` arrays.

        Quantile crossing (p10 > p90 etc.) is repaired by sorting the three
        quantile predictions row-wise.
        """
        x = make_x(df, self.feature_cols)
        out: dict[str, dict[str, np.ndarray]] = {}
        for target in TARGETS:
            preds = {n: m.predict(x) for n, m in self.models_[target].items()}
            stacked = np.sort(np.vstack([preds["p10"], preds["p50"], preds["p90"]]), axis=0)
            out[target] = {"p10": stacked[0], "value": stacked[1], "p90": stacked[2]}
        # If a CQR calibrator is attached, serve calibrated bands transparently so
        # every consumer (API, inverse design, metrics) gets guaranteed coverage.
        calibrator = getattr(self, "conformal_", None)
        if calibrator is not None:
            out = calibrator.apply(out)
        return out

    def feature_importances(self) -> dict[str, dict[str, float]]:
        """Median-model gain importance per target (sums to 1 per target)."""
        out: dict[str, dict[str, float]] = {}
        for target in TARGETS:
            imp = self.models_[target]["p50"].booster_.feature_importance(importance_type="gain")
            total = imp.sum() or 1.0
            out[target] = {c: float(v / total) for c, v in zip(self.feature_cols, imp, strict=True)}
        return out

    # --- persistence ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "forward_model.joblib")
        (path / "feature_importances.json").write_text(
            json.dumps(self.feature_importances(), indent=2)
        )
        return path

    @staticmethod
    def load(path: str | Path) -> ForwardModel:
        return joblib.load(Path(path) / "forward_model.joblib")
