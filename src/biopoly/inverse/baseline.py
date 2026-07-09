"""Baseline inverse design: sample -> predict -> rank.

Generate many valid candidate formulations, run them through the forward model,
and keep the ones whose predicted properties land closest to the target (in
tolerance units, with an uncertainty penalty). Simple, fast, fully interpretable —
the honest first thing to ship before reaching for optimisation.
"""

from __future__ import annotations

import numpy as np

from biopoly.data.generate import _sample_formulation
from biopoly.inverse.common import describe, predict_one, score_prediction


def design(
    model,
    target: dict[str, float],
    *,
    n_candidates: int = 4000,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Return the top-k candidate formulations for ``target``, best first."""
    rng = np.random.default_rng(seed)
    scored: list[tuple[float, object, dict]] = []
    for _ in range(n_candidates):
        form = _sample_formulation(rng)
        pred = predict_one(model, form)
        s = score_prediction(pred, target, weights=weights)
        scored.append((s, form, pred))

    scored.sort(key=lambda x: x[0])
    results = []
    for s, form, pred in scored[:top_k]:
        results.append(
            {
                "score": round(s, 4),
                "formulation": describe(form),
                "predicted": {k: round(v["value"], 2) for k, v in pred.items()},
            }
        )
    return results
