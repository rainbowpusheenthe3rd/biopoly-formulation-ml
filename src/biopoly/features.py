"""Feature preparation and train/test splitting.

Feature contract is ``schema.FEATURE_COLS``. Categorical columns are cast to the
pandas ``category`` dtype so LightGBM handles them natively (no one-hot), and
missing *targets* are handled per-model at fit time (a row missing one property
still trains the other four).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biopoly.data.schema import CATEGORICAL_FEATURES, FEATURE_COLS


def make_x(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return the model-input frame with categoricals typed for LightGBM.

    ``cols`` defaults to ``schema.FEATURE_COLS``; pass a superset (e.g. with the
    DSC signal features appended) to train an ablation variant.
    """
    cols = cols if cols is not None else FEATURE_COLS
    x = df[cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in x.columns:
            x[col] = x[col].astype("category")
    return x


def split(
    df: pd.DataFrame, *, test_size: float = 0.2, mode: str = "random", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test.

    ``mode="random"`` for headline metrics; ``mode="temporal"`` holds out the most
    recent records (useful to expose the mid-2025 supplier drift).
    """
    if mode == "temporal":
        df = df.sort_values("date")
        cut = int(len(df) * (1 - test_size))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
