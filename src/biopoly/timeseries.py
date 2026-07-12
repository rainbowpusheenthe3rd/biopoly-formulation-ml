"""Temporal feedstock-quality signal, decomposition and a seasonal baseline.

Bio-feedstock quality is not constant: agricultural raw materials follow an annual
cycle (harvest → storage → depletion) on top of a slow multi-year trend, plus
noise. That seasonal signal propagates into the finished polymer's properties — so
it is real, learnable structure the forward model should see, and the mid-2025
supplier-purity shift is then a **regime change layered on top** of this seasonal
baseline (see :mod:`biopoly.monitoring.drift`).

This module provides three things, each deliberately simple and legible:

1. :func:`seasonal_feedstock_quality` — the generative signal used by the dataset
   generator (trend + annual seasonality + noise), returned as a multiplier ~1.0.
2. :func:`stl_decompose` / :func:`classical_decompose` — split an observed series
   into trend / seasonal / residual (STL via ``statsmodels`` when available, else a
   dependency-light classical moving-average decomposition).
3. :func:`seasonal_naive_forecast` — the honest baseline any ML forecast must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Signal defaults — plausible, gentle, and easy to read off a chart.
_TREND_PER_YEAR = 0.010  # slow multi-year improvement in feedstock quality
_SEASONAL_AMPLITUDE = 0.06  # ±6% annual swing
_PEAK_MONTH = 9  # quality peaks post-harvest (autumn, northern hemisphere)
_NOISE_SD = 0.015
_Q_MIN, _Q_MAX = 0.5, 1.5


def seasonal_feedstock_quality(
    dates,
    *,
    rng: np.random.Generator | None = None,
    trend_per_year: float = _TREND_PER_YEAR,
    seasonal_amplitude: float = _SEASONAL_AMPLITUDE,
    peak_month: int = _PEAK_MONTH,
    noise_sd: float = _NOISE_SD,
) -> np.ndarray:
    """Return a feedstock-quality multiplier (~1.0) for each date.

    Args:
        dates: Anything ``pd.to_datetime`` accepts (array/Series of timestamps).
        rng: Optional generator for the noise term; omit for a noise-free signal.
        trend_per_year: Linear drift in quality per year from the first date.
        seasonal_amplitude: Peak-to-mean amplitude of the annual cycle.
        peak_month: Month (1-12) at which quality peaks.
        noise_sd: Std-dev of the multiplicative noise term.

    Returns:
        A float array of quality multipliers, clipped to ``[0.5, 1.5]``.
    """
    ts = pd.to_datetime(pd.Series(list(pd.to_datetime(dates)))).reset_index(drop=True)
    years = (ts - ts.min()).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    doy = ts.dt.dayofyear.to_numpy()
    peak_doy = pd.Timestamp(year=2001, month=peak_month, day=15).dayofyear
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)
    trend = trend_per_year * years
    noise = rng.normal(0.0, noise_sd, len(ts)) if rng is not None else 0.0
    return np.clip(1.0 + trend + seasonal + noise, _Q_MIN, _Q_MAX)


@dataclass
class Decomposition:
    """Additive decomposition of a series: ``observed = trend + seasonal + resid``."""

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray

    @property
    def seasonal_strength(self) -> float:
        """Fraction of de-trended variance explained by the seasonal component (0-1)."""
        detr = self.observed - self.trend
        detr = detr[~np.isnan(detr)]
        seas = self.seasonal[~np.isnan(self.observed - self.trend)]
        var = np.var(detr)
        return float(np.clip(1.0 - np.var(detr - seas) / var, 0.0, 1.0)) if var > 0 else 0.0


def _as_array(series) -> np.ndarray:
    return np.asarray(series, dtype=float).ravel()


def classical_decompose(series, period: int = 12) -> Decomposition:
    """Classical additive decomposition via a centred moving-average trend.

    Dependency-light and transparent: trend is a centred rolling mean over one
    period, the seasonal component is the per-phase mean of the de-trended signal
    (re-centred to sum to zero), and the residual is whatever remains.
    """
    obs = _as_array(series)
    s = pd.Series(obs)
    trend = s.rolling(window=period, center=True, min_periods=max(2, period // 2)).mean()
    detrended = s - trend
    phase = np.arange(len(obs)) % period
    seasonal_by_phase = pd.Series(detrended.to_numpy()).groupby(phase).transform("mean")
    seasonal = seasonal_by_phase - np.nanmean(seasonal_by_phase)  # centre to zero
    resid = s - trend - seasonal
    return Decomposition(obs, trend.to_numpy(), seasonal.to_numpy(), resid.to_numpy())


def stl_decompose(series, period: int = 12) -> Decomposition:
    """STL decomposition via ``statsmodels`` when installed, else classical.

    STL is preferred (robust, handles changing seasonality), but the pipeline must
    not hard-depend on it — so we degrade gracefully to :func:`classical_decompose`.
    """
    obs = _as_array(series)
    try:
        from statsmodels.tsa.seasonal import STL

        res = STL(obs, period=period, robust=True).fit()
        return Decomposition(obs, res.trend, res.seasonal, res.resid)
    except Exception:
        return classical_decompose(obs, period=period)


def seasonal_naive_forecast(series, period: int = 12, horizon: int = 12) -> np.ndarray:
    """Seasonal-naïve forecast: repeat the last observed season forward.

    The honest baseline — any ML forecaster has to beat "next September looks like
    last September" before it earns its complexity.
    """
    obs = _as_array(series)
    if len(obs) < period:
        raise ValueError(f"need at least one full period ({period}) of history, got {len(obs)}")
    last_season = obs[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last_season, reps)[:horizon]


def monthly_mean(df: pd.DataFrame, value_col: str, date_col: str = "date") -> pd.Series:
    """Month-start mean of ``value_col`` — a clean series to decompose or plot."""
    s = df[[date_col, value_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col])
    return s.set_index(date_col)[value_col].resample("MS").mean()
