"""Shared fixtures: a small synthetic dataset and a fast-trained forward model."""

from __future__ import annotations

import pandas as pd
import pytest

from biopoly.config import Settings
from biopoly.data.generate import build_dataset
from biopoly.models.forward import ForwardModel


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    cfg = Settings(n_samples=400, seed=7)
    return build_dataset(cfg)


@pytest.fixture(scope="session")
def fast_model(small_df) -> ForwardModel:
    return ForwardModel({"n_estimators": 120, "learning_rate": 0.08}).fit(small_df)
