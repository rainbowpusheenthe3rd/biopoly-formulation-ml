"""Distribution-drift monitoring.

Lightweight, dependency-free drift detection: a two-sample Kolmogorov-Smirnov test
per numeric column and a Population Stability Index (PSI) for categoricals. Used to
raise the "incoming formulations/outputs have drifted from the training distribution"
alert that motivates retraining — here it catches the mid-2025
supplier-purity shift.

An Evidently-based report is available via ``evidently_report`` if the optional
``drift`` extra is installed, but KS/PSI is the always-on default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from biopoly.config import settings


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = ref.dropna()
    cur = cur.dropna()
    if ref.dtype == object or str(ref.dtype) == "category":
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        r = ref.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-6
        c = cur.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-6
    else:
        edges = np.histogram_bin_edges(ref, bins=bins)
        r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1) + 1e-6
        c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    p_threshold: float | None = None,
) -> dict:
    """Per-column drift report + overall verdict.

    Numeric columns use KS (drift if p < threshold); categoricals use PSI
    (drift if PSI > 0.2, the usual "significant shift" rule of thumb).
    """
    p_threshold = p_threshold if p_threshold is not None else settings.drift_p_value
    per_column: dict[str, dict] = {}
    for col in columns:
        ref, cur = reference[col].dropna(), current[col].dropna()
        if ref.empty or cur.empty:
            continue
        if pd.api.types.is_numeric_dtype(ref):
            stat, p = stats.ks_2samp(ref, cur)
            per_column[col] = {
                "test": "ks",
                "statistic": float(stat),
                "p_value": float(p),
                "psi": _psi(ref, cur),
                "drifted": bool(p < p_threshold),
            }
        else:
            psi = _psi(ref, cur)
            per_column[col] = {"test": "psi", "psi": psi, "drifted": bool(psi > 0.2)}

    drifted = [c for c, r in per_column.items() if r["drifted"]]
    return {
        "n_columns": len(per_column),
        "n_drifted": len(drifted),
        "drifted_columns": drifted,
        "alert": len(drifted) > 0,
        "per_column": per_column,
    }


def format_report(report: dict) -> str:
    lines = [
        f"drift: {report['n_drifted']}/{report['n_columns']} columns drifted "
        f"-> alert={report['alert']}"
    ]
    for col, r in report["per_column"].items():
        flag = "DRIFT" if r["drifted"] else "  ok "
        if r["test"] == "ks":
            lines.append(
                f"  [{flag}] {col:26s} KS={r['statistic']:.3f} p={r['p_value']:.2e} "
                f"PSI={r['psi']:.3f}"
            )
        else:
            lines.append(f"  [{flag}] {col:26s} PSI={r['psi']:.3f}")
    return "\n".join(lines)
