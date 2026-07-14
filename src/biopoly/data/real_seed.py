"""Real literature property values — the first rungs of the real-data ladder.

Two datasets (see ``DATA_STRATEGY.md``):

* **Neat-polymer seed** (``real_seed.csv``) — *indicative literature mid-ranges* for
  the neat polymers, only the robustly reported properties (tensile, water absorption,
  optical clarity). Its honest use is **anchoring**: check the synthetic generator's
  neat-polymer outputs sit in reality's ballpark and surface where they do not.

* **Real formulations** (``real_formulations.csv``) — a handful of PLA/PBAT and PLA/PBS
  melt-blends with reported tensile strength, now carrying enough **processing metadata**
  (composition, melt temperature, protocol) to be *schema-complete* training rows. Only
  tensile is reported, so the other four targets stay missing — the per-target NaN
  masking in :class:`~biopoly.models.forward.ForwardModel` uses them for tensile alone.

Five literature points cannot *train* a model that already sees ~2k synthetic rows.
Their honest value is as a small **real-world validation set** — measuring the
synthetic-trained model's sim-to-real tensile gap and whether its calibrated band
covers reality (:func:`evaluate_sim_to_real`) — plus an honest leave-one-out
augmentation check (:func:`augmentation_experiment`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from biopoly import TARGETS
from biopoly.data.chemistry import PROC_TEMP_OPT, Formulation, forward_true
from biopoly.data.schema import FEATURE_COLS

if TYPE_CHECKING:
    from biopoly.models.forward import ForwardModel

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SEED_PATH = _DATA_DIR / "real_seed.csv"
_FORMULATIONS_PATH = _DATA_DIR / "real_formulations.csv"

# Properties present in the seed (neat-polymer, commonly reported).
REAL_PROPERTIES = ["tensile_strength_mpa", "water_absorption_pct", "optical_clarity_pct"]

# The only property the real formulations robustly report; the rest stay missing.
REAL_FORMULATION_TARGET = "tensile_strength_mpa"


def load_real_seed() -> pd.DataFrame:
    """Load the literature neat-polymer seed (one row per polymer)."""
    return pd.read_csv(_SEED_PATH)


def load_real_formulations() -> pd.DataFrame:
    """Load the real literature *blend* datapoints (schema-complete, tensile target).

    PLA/PBAT and PLA/PBS melt-blends with reported tensile strength and enough
    processing metadata (composition, melt temperature, protocol) to line up with the
    feature schema. Blend tensile varies widely with processing and compatibilisation,
    and the melt residence time is a representative extrusion value rather than a
    per-study figure, so treat these as indicative rather than definitive.
    """
    return pd.read_csv(_FORMULATIONS_PATH)


def real_formulations_training_frame() -> pd.DataFrame:
    """Real formulations as schema-complete rows ready to concat with the synthetic set.

    Returns a frame with exactly ``FEATURE_COLS + TARGETS``: every feature column
    populated from the literature metadata, ``tensile_strength_mpa`` set to the reported
    value, and the other four targets left as NaN (not measured). Concatenated into the
    training frame, the forward model's per-target NaN masking uses each real row for
    tensile only.
    """
    raw = load_real_formulations()
    frame = raw.reindex(columns=FEATURE_COLS).copy()
    for target in TARGETS:
        frame[target] = np.nan
    frame[REAL_FORMULATION_TARGET] = raw[REAL_FORMULATION_TARGET].to_numpy(dtype=float)
    frame["blend"] = raw["blend"].to_numpy()
    return frame


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


def evaluate_sim_to_real(model: ForwardModel) -> dict[str, object]:
    """Score a synthetic-trained model against the real literature blends (tensile).

    The scarce-data question that matters: does a model trained on synthetic,
    physics-informed data transfer to real measurements? This predicts tensile on each
    real formulation and reports the per-blend error and whether the (conformally
    calibrated) p10-p90 band covers the literature value — a genuine out-of-distribution
    check, not a fitted metric.

    Returns a dict with a per-blend ``table`` (DataFrame) plus aggregate ``mae``,
    ``coverage`` (fraction of blends whose band contains the real value) and ``n``.
    """
    frame = real_formulations_training_frame()
    preds = model.predict(frame)[REAL_FORMULATION_TARGET]
    real = frame[REAL_FORMULATION_TARGET].to_numpy(dtype=float)
    value, p10, p90 = preds["value"], preds["p10"], preds["p90"]
    covered = (real >= p10) & (real <= p90)
    table = pd.DataFrame(
        {
            "blend": frame["blend"].to_numpy(),
            "real_tensile_mpa": real.round(1),
            "pred_tensile_mpa": value.round(1),
            "p10": p10.round(1),
            "p90": p90.round(1),
            "abs_err_mpa": np.abs(value - real).round(1),
            "band_covers": covered,
        }
    )
    return {
        "table": table,
        "mae": float(np.mean(np.abs(value - real))),
        "coverage": float(np.mean(covered)),
        "n": int(len(real)),
    }


def augmentation_experiment(
    synth_df: pd.DataFrame, *, params: dict | None = None
) -> dict[str, object]:
    """Leave-one-out check of whether the real blends help as training augmentation.

    For each real formulation, predict its tensile from (a) a synthetic-only model and
    (b) a model trained on synthetic + the *other* real blends, and compare the errors.
    Honest by construction: with five points against thousands of synthetic rows the
    augmentation is expected to barely move the needle — which is the point. It argues
    the real data earns its keep as validation and (later) fine-tuning, not as raw
    augmentation, and motivates the next rung of the data ladder.

    Returns a dict with a per-blend ``table`` and aggregate ``mae_synth_only`` /
    ``mae_augmented``.
    """
    from biopoly.models.forward import ForwardModel

    real = real_formulations_training_frame()
    # Concat only the model-relevant columns so the real rows (which lack the
    # synthetic-only columns like `date`) don't back-fill unrelated dtypes.
    model_cols = FEATURE_COLS + TARGETS
    synth = synth_df[model_cols]
    real_x = real[model_cols]
    base = ForwardModel(params).fit(synth_df)
    rows: list[dict[str, object]] = []
    for i in range(len(real)):
        held = real.iloc[[i]]
        others = real_x.drop(real_x.index[i])
        augmented = ForwardModel(params).fit(pd.concat([synth, others], ignore_index=True))
        y = float(held[REAL_FORMULATION_TARGET].iloc[0])
        pred_base = float(base.predict(held)[REAL_FORMULATION_TARGET]["value"][0])
        pred_aug = float(augmented.predict(held)[REAL_FORMULATION_TARGET]["value"][0])
        rows.append(
            {
                "blend": held["blend"].iloc[0],
                "real_tensile_mpa": round(y, 1),
                "synth_only_err": round(abs(pred_base - y), 1),
                "augmented_err": round(abs(pred_aug - y), 1),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "table": table,
        "mae_synth_only": float(table["synth_only_err"].mean()),
        "mae_augmented": float(table["augmented_err"].mean()),
    }
