"""Evaluation metrics.

Beyond MAE/RMSE/R2 we report two things that matter more to a
scientist: **interval coverage** (does the p10-p90 band actually contain the truth
~80% of the time?) and **within-tolerance accuracy** against a *per-variable*
acceptable margin (the tolerance on water absorption is nothing like the one on
tensile strength).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biopoly import TARGETS

# Per-variable acceptable margin (absolute units). Illustrative — in reality these
# come from the scientist / stakeholders.
TOLERANCE = {
    "tensile_strength_mpa": 3.0,
    "melt_flow_index_g10min": 3.0,
    "biodegradation_60d_pct": 8.0,
    "water_absorption_pct": 2.0,
    "optical_clarity_pct": 5.0,
}


def evaluate(
    df: pd.DataFrame, preds: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        y = y[mask]
        val = preds[target]["value"][mask]
        p10 = preds[target]["p10"][mask]
        p90 = preds[target]["p90"][mask]
        err = val - y
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        out[target] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": float(1 - ss_res / ss_tot),
            "interval_coverage": float(np.mean((y >= p10) & (y <= p90))),
            "mean_interval_width": float(np.mean(p90 - p10)),
            "within_tolerance": float(np.mean(np.abs(err) <= TOLERANCE[target])),
        }
    return out


def summary_row(metrics: dict[str, dict[str, float]]) -> float:
    """Single headline number: mean R2 across targets."""
    return float(np.mean([m["r2"] for m in metrics.values()]))


def format_table(metrics: dict[str, dict[str, float]]) -> str:
    head = (
        f"{'target':26s} {'n':>5s} {'MAE':>7s} {'RMSE':>7s} {'R2':>6s} {'cover':>6s} {'<=tol':>6s}"
    )
    lines = [head, "-" * len(head)]
    for t, m in metrics.items():
        lines.append(
            f"{t:26s} {m['n']:5d} {m['mae']:7.2f} {m['rmse']:7.2f} "
            f"{m['r2']:6.3f} {m['interval_coverage']:6.2f} {m['within_tolerance']:6.2f}"
        )
    lines.append(f"\nmean R2 = {summary_row(metrics):.3f}")
    return "\n".join(lines)
