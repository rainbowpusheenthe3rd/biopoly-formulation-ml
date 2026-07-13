"""Bayesian-optimisation inverse design (Optuna/TPE).

The upgrade over the baseline: instead of blind sampling, use a
TPE surrogate to steer the search toward formulations predicted to hit spec,
balancing exploitation against the model's own uncertainty. This is the core
inverse-design loop: "tell us what the material needs to do -> we generate the
formulation" — target spec in, ranked recipes out.
"""

from __future__ import annotations

import numpy as np
import optuna

from biopoly.data.chemistry import ADDITIVES, POLYMERS
from biopoly.data.generate import _sample_formulation
from biopoly.inverse.common import (
    ADDITIVE_CAPS,
    describe,
    predict_one,
    score_prediction,
    vector_to_formulation,
)


def _warm_start_params(model, target, weights, n_seed, n_pool, seed):
    """Best random candidates, as Optuna param dicts to enqueue (warm start)."""
    rng = np.random.default_rng(seed)
    scored = []
    for _ in range(n_pool):
        form = _sample_formulation(rng)
        s = score_prediction(predict_one(model, form), target, weights=weights)
        scored.append((s, form))
    scored.sort(key=lambda x: x[0])
    params = []
    for _, form in scored[:n_seed]:
        p = {f"w_{poly}": form.polymer_frac.get(poly, 0.0) for poly in POLYMERS}
        for a in ADDITIVES:
            p[f"add_{a}"] = min(form.additive_frac.get(a, 0.0), ADDITIVE_CAPS[a])
        p["process_temp_c"] = float(np.clip(form.process_temp_c, 100.0, 240.0))
        p["process_time_min"] = float(np.clip(form.process_time_min, 4.0, 90.0))
        params.append(p)
    return params


def design(
    model,
    target: dict[str, float],
    *,
    n_trials: int = 400,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    warm_start: int = 12,
    seed: int = 0,
) -> list[dict]:
    """Return the top-k formulations found by TPE search, best first.

    The study is warm-started with the best random candidates so TPE refines from
    strong starting points rather than searching the 13-D space cold.
    """

    def objective(trial: optuna.Trial) -> float:
        polymer_w = [trial.suggest_float(f"w_{p}", 0.0, 1.0) for p in POLYMERS]
        if sum(polymer_w) == 0:
            polymer_w[0] = 1.0
        additive_w = {a: trial.suggest_float(f"add_{a}", 0.0, ADDITIVE_CAPS[a]) for a in ADDITIVES}
        temp = trial.suggest_float("process_temp_c", 100.0, 240.0)
        time_min = trial.suggest_float("process_time_min", 4.0, 90.0)
        form = vector_to_formulation(np.asarray(polymer_w), additive_w, temp, time_min)
        pred = predict_one(model, form)
        return score_prediction(pred, target, weights=weights)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_start:
        for p in _warm_start_params(model, target, weights, warm_start, 20 * warm_start, seed):
            study.enqueue_trial(p)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    finished = [t for t in study.trials if t.value is not None]
    trials = sorted(finished, key=lambda t: float(t.value))[:top_k]  # type: ignore[arg-type]
    results = []
    for t in trials:
        polymer_w = [t.params[f"w_{p}"] for p in POLYMERS]
        additive_w = {a: t.params[f"add_{a}"] for a in ADDITIVES}
        form = vector_to_formulation(
            np.asarray(polymer_w),
            additive_w,
            t.params["process_temp_c"],
            t.params["process_time_min"],
        )
        pred = predict_one(model, form)
        results.append(
            {
                "score": round(float(t.value), 4),  # type: ignore[arg-type]
                "formulation": describe(form),
                "predicted": {k: round(v["value"], 2) for k, v in pred.items()},
            }
        )
    return results
