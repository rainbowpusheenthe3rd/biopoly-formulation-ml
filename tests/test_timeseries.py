"""L7 — time series & seasonality: the feedstock signal, decomposition and baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biopoly.timeseries import (
    Decomposition,
    classical_decompose,
    seasonal_feedstock_quality,
    seasonal_naive_forecast,
    stl_decompose,
)

pytestmark = pytest.mark.layer(7)  # time series & seasonality


def test_feedstock_quality_range_and_centre():
    dates = pd.date_range("2023-01-01", "2026-01-01", freq="D")
    q = seasonal_feedstock_quality(dates)
    assert len(q) == len(dates)
    assert np.all((q >= 0.5) & (q <= 1.5))
    assert abs(np.mean(q) - 1.0) < 0.05  # centred near 1.0


def test_feedstock_quality_has_annual_seasonality():
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    q = pd.Series(seasonal_feedstock_quality(dates), index=dates)
    monthly = q.groupby(q.index.month).mean()
    assert monthly.max() - monthly.min() > 0.03  # a real annual swing


def test_classical_decompose_recovers_seasonal():
    t = np.arange(48)
    series = 10 + 0.1 * t + 2.0 * np.sin(2 * np.pi * t / 12)
    d = classical_decompose(series, period=12)
    assert isinstance(d, Decomposition)
    assert d.seasonal_strength > 0.7


def test_stl_decompose_returns_full_length():
    series = 10 + np.sin(2 * np.pi * np.arange(36) / 12)
    d = stl_decompose(series, period=12)
    assert len(d.trend) == len(series) == len(d.seasonal) == len(d.resid)


def test_seasonal_naive_forecast_repeats_last_season():
    series = np.arange(24, dtype=float)
    fc = seasonal_naive_forecast(series, period=12, horizon=12)
    assert np.allclose(fc, series[-12:])


def test_seasonal_naive_forecast_needs_a_full_period():
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.arange(5.0), period=12, horizon=3)
