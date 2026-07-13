"""FastAPI service exposing the forward model and inverse design.

Endpoints:
  GET  /health   -> liveness + which model version is serving
  POST /predict  -> formulation -> predicted properties (+ p10/p90 bands)
  POST /design   -> target spec  -> ranked candidate formulations

The champion model is loaded once at startup from the registry. Pydantic validates
every request and validation failures return typed 422s; a missing model returns 503.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from biopoly import TARGETS, __version__
from biopoly.api.models import Candidate, DesignRequest, DesignResponse
from biopoly.data.chemistry import Formulation
from biopoly.data.schema import FormulationInput, PredictResponse, PropertyPrediction
from biopoly.inverse import baseline, bayesopt
from biopoly.inverse.common import predict_one
from biopoly.models.registry import ModelRegistry

STATE: dict = {"model": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once at startup from the registry."""
    reg = ModelRegistry()
    if reg.champion() is not None:
        STATE["model"] = reg.load_champion()
        STATE["version"] = reg.champion()
    yield


app = FastAPI(title="biopoly formulation API", version=__version__, lifespan=lifespan)


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503, detail="no champion model registered; run biopoly-train"
        )
    return STATE["model"]


@app.get("/health")
def health() -> dict:
    """Liveness probe: status, package version and the served model version."""
    return {
        "status": "ok",
        "biopoly_version": __version__,
        "model_version": STATE["version"],
        "targets": TARGETS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: FormulationInput) -> PredictResponse:
    """Predict the five properties (with p10/p90 bands) for a formulation."""
    model = _require_model()
    warnings: list[str] = []
    total = sum(body.frac.values()) + sum(body.additives.values())
    frac, additives = body.frac, body.additives
    if abs(total - 1.0) > 1e-3 and total > 0:
        frac = {k: v / total for k, v in frac.items()}
        additives = {k: v / total for k, v in additives.items()}
        warnings.append(f"fractions summed to {total:.3f}; renormalised to 1.0")

    form = Formulation(frac, additives, body.process_temp_c, body.process_time_min)
    pred = predict_one(
        model, form, protocol=body.tensile_protocol, feedstock_quality=body.feedstock_quality
    )
    return PredictResponse(
        predictions={
            t: PropertyPrediction(value=pred[t]["value"], p10=pred[t]["p10"], p90=pred[t]["p90"])
            for t in TARGETS
        },
        warnings=warnings,
    )


@app.post("/design", response_model=DesignResponse)
def design(body: DesignRequest) -> DesignResponse:
    """Return ranked candidate formulations for a target property spec."""
    model = _require_model()
    try:
        target = body.validated_target()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if body.method == "baseline":
        cands = baseline.design(
            model, target, n_candidates=body.budget, top_k=body.top_k, weights=body.weights
        )
    else:
        cands = bayesopt.design(
            model, target, n_trials=body.budget, top_k=body.top_k, weights=body.weights
        )
    return DesignResponse(
        method=body.method,
        target=target,
        candidates=[Candidate(**c) for c in cands],
    )
