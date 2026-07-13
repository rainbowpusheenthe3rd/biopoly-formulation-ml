"""Shared pieces for inverse design: feature assembly, scoring, validity.

Inverse design sits *on top of* the forward model: search the
formulation space for a recipe whose predicted properties land on a target spec.
Both the baseline sampler and the Bayesian optimiser use the scorer below, which
scores in *tolerance units* so a 3 MPa tensile miss and a 2% water miss are
comparable, and penalises uncertain predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biopoly import TARGETS
from biopoly.data.chemistry import ADDITIVES, POLYMERS, Formulation
from biopoly.data.schema import FEATURE_COLS
from biopoly.models.metrics import TOLERANCE


def to_features(
    form: Formulation, protocol: str = "ISO527", *, feedstock_quality: float = 1.0
) -> pd.DataFrame:
    row = form.as_row()
    row["primary_polymer"] = max(form.polymer_frac, key=form.polymer_frac.get)
    row["tensile_protocol"] = protocol
    # Design/predict at nominal feedstock quality unless a batch value is supplied.
    row["feedstock_quality"] = feedstock_quality
    x = pd.DataFrame([row])[FEATURE_COLS]
    x["primary_polymer"] = x["primary_polymer"].astype("category")
    x["tensile_protocol"] = x["tensile_protocol"].astype("category")
    return x


def score_prediction(
    pred: dict[str, dict[str, float]],
    target: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
    uncertainty_lambda: float = 0.1,
) -> float:
    """Lower is better. Distance-to-target in tolerance units + uncertainty penalty."""
    weights = weights or {}
    dist = 0.0
    width = 0.0
    for t in target:
        tol = TOLERANCE[t]
        w = weights.get(t, 1.0)
        dist += w * abs(pred[t]["value"] - target[t]) / tol
        width += w * (pred[t]["p90"] - pred[t]["p10"]) / tol
    n = max(len(target), 1)
    return dist / n + uncertainty_lambda * width / n


def predict_one(
    model, form: Formulation, protocol: str = "ISO527", *, feedstock_quality: float = 1.0
) -> dict[str, dict[str, float]]:
    raw = model.predict(to_features(form, protocol, feedstock_quality=feedstock_quality))
    return {t: {k: float(v[0]) for k, v in raw[t].items()} for t in TARGETS}


def vector_to_formulation(
    polymer_w: np.ndarray, additive_w: dict[str, float], temp: float, time_min: float
) -> Formulation:
    """Turn a raw search vector into a valid Formulation (normalised, mass-balanced)."""
    total_add = min(sum(additive_w.values()), 0.40)
    matrix = 1.0 - total_add
    w = np.clip(polymer_w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(POLYMERS)) / len(POLYMERS)
    polymer_frac = {p: float(matrix * wi) for p, wi in zip(POLYMERS, w, strict=True)}
    return Formulation(polymer_frac, dict(additive_w), float(temp), float(time_min))


def describe(form: Formulation) -> dict:
    """Compact, human-readable formulation (drops near-zero components)."""
    # Cast keys to plain str: sampled formulations can carry numpy string keys.
    polymers = {str(p): round(f, 3) for p, f in form.polymer_frac.items() if f > 0.005}
    additives = {str(a): round(f, 3) for a, f in form.additive_frac.items() if f > 0.005}
    return {
        "polymers": polymers,
        "additives": additives,
        "process_temp_c": round(form.process_temp_c, 1),
        "process_time_min": round(form.process_time_min, 1),
    }


ADDITIVE_CAPS = {
    "plasticizer": 0.20,
    "nucleating": 0.03,
    "compatibilizer": 0.08,
    "fibre": 0.30,
    "chain_extender": 0.02,
}
assert set(ADDITIVE_CAPS) == set(ADDITIVES)
