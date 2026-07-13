"""Conformalized Quantile Regression (CQR) for calibrated prediction intervals.

The forward model already emits p10/p90 quantile bands, but a quantile booster's
nominal 80% interval rarely *covers* the truth 80% of the time on held-out data —
it is typically a little too tight or too wide, with no guarantee either way.

CQR (Romano, Patterson & Candès, 2019) fixes this with a distribution-free,
finite-sample guarantee: on a held-out **calibration** set, score each point by how
far outside its band the truth fell,

    E_i = max(p10_i - y_i,  y_i - p90_i)

(negative when the truth sits comfortably inside the band), then take the
conformal quantile ``Q`` of those scores and widen every future band to
``[p10 - Q, p90 + Q]``. This yields marginal coverage of at least ``1 - alpha``
regardless of whether the underlying quantiles were any good. When the base bands
are *over*-covering, ``Q`` goes negative and CQR correctly *tightens* them.

Calibration must be done on data the forward model did not train on, so
``biopoly-train`` carves a dedicated calibration split (see ``models/train.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biopoly import TARGETS


class ConformalCalibrator:
    """Per-target CQR adjustments. ``alpha=0.2`` matches the p10-p90 (80%) band."""

    def __init__(self, alpha: float = 0.20):
        self.alpha = alpha
        self.adjustments_: dict[str, float] = {}

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.

        If that rank exceeds ``n`` (too few calibration points for the requested
        level) the guarantee needs an infinite adjustment; we fall back to the
        largest observed score, the tightest bound the data can actually support.
        """
        n = len(scores)
        if n == 0:
            return 0.0
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        if k >= n:
            return float(np.max(scores))
        return float(np.sort(scores)[k - 1])

    def fit(self, df: pd.DataFrame, preds: dict[str, dict[str, np.ndarray]]) -> ConformalCalibrator:
        """Fit adjustments from calibration predictions and their true targets."""
        for target in TARGETS:
            y = df[target].to_numpy(dtype=float)
            mask = ~np.isnan(y)
            y = y[mask]
            p10 = preds[target]["p10"][mask]
            p90 = preds[target]["p90"][mask]
            scores = np.maximum(p10 - y, y - p90)
            self.adjustments_[target] = self._conformal_quantile(scores, self.alpha)
        return self

    def apply(self, preds: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        """Return a new preds dict with each band widened (or tightened) by ``Q``.

        The point estimate is untouched, and the calibrated band is clipped so the
        point estimate always stays inside it (a large negative ``Q`` cannot invert
        the interval).
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for target in TARGETS:
            q = self.adjustments_.get(target, 0.0)
            value = preds[target]["value"]
            p10 = np.minimum(preds[target]["p10"] - q, value)
            p90 = np.maximum(preds[target]["p90"] + q, value)
            out[target] = {"p10": p10, "value": value, "p90": p90}
        return out

    # --- persistence (also embedded in ForwardModel via joblib) ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "conformal.json").write_text(
            json.dumps({"alpha": self.alpha, "adjustments": self.adjustments_}, indent=2)
        )
        return path
