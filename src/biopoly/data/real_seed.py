"""A small seed of real neat-polymer property values from the literature.

The first rung of the real-data ladder (see ``DATA_STRATEGY.md``). These are
*indicative literature mid-ranges* for the **neat** polymers, and only the commonly
and robustly reported properties — tensile strength, water absorption, optical
clarity. Melt-flow index and 60-day biodegradation are strongly condition-dependent
and are deliberately left out of the seed until primary citations are attached.

Their first honest use is **anchoring**: check that the synthetic generator's
neat-polymer outputs sit in the same ballpark as reality, and surface where they do
not — a lever for calibrating the synthetic ground truth. They are *not* yet used to
train the model (neat-polymer literature points carry no processing metadata).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biopoly.data.chemistry import PROC_TEMP_OPT, Formulation, forward_true

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SEED_PATH = _DATA_DIR / "real_seed.csv"
_FORMULATIONS_PATH = _DATA_DIR / "real_formulations.csv"

# Properties present in the seed (neat-polymer, commonly reported).
REAL_PROPERTIES = ["tensile_strength_mpa", "water_absorption_pct", "optical_clarity_pct"]


def load_real_seed() -> pd.DataFrame:
    """Load the literature neat-polymer seed (one row per polymer)."""
    return pd.read_csv(_SEED_PATH)


def load_real_formulations() -> pd.DataFrame:
    """Load the small set of real literature *blend* datapoints (tensile only).

    A handful of PLA/PBAT and PLA/PBS melt-blends with reported tensile strength — the
    first real *formulations* (as opposed to neat polymers). Still partial (tensile
    only, no full processing metadata), and blend values vary widely with processing
    and compatibilisation, so treat them as indicative rather than definitive.
    """
    return pd.read_csv(_FORMULATIONS_PATH)


def synthetic_vs_real() -> pd.DataFrame:
    """Line the synthetic ground truth up against the real seed, per neat polymer.

    Runs each neat polymer through :func:`forward_true` at its optimum processing
    temperature and compares the synthetic value to the literature value, with the
    absolute percentage gap — an honest anchoring check, not a fitted metric.
    """
    seed = load_real_seed()
    rows: list[dict] = []
    for _, r in seed.iterrows():
        polymer = r["polymer"]
        form = Formulation({polymer: 1.0}, {}, PROC_TEMP_OPT[polymer], 20.0)
        syn = forward_true(form)
        for prop in REAL_PROPERTIES:
            real = r[prop]
            if pd.isna(real):
                continue
            value = syn[prop]
            rows.append(
                {
                    "polymer": polymer,
                    "property": prop,
                    "synthetic": round(value, 2),
                    "real": round(float(real), 2),
                    "abs_pct_gap": round(100.0 * abs(value - real) / max(abs(real), 1e-6), 1),
                }
            )
    return pd.DataFrame(rows)
