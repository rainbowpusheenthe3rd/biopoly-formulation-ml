"""FastAPI service exposing the forward model and inverse design.

Endpoints:
  GET  /health      -> liveness + which model version is serving (open)
  GET  /whoami      -> the authenticated tenant (tenant_id + role)
  POST /predict     -> formulation -> predicted properties (+ p10/p90 bands)
  POST /design      -> target spec  -> ranked candidate formulations
  GET  /history     -> this tenant's past runs (isolated)
  GET  /admin/usage -> per-tenant call counts (admin role only)

All endpoints except /health require an ``X-API-Key`` header, resolved to a
:class:`~biopoly.api.tenancy.TenantContext` at a single choke-point; every tenant read
and write goes through the tenant-scoped :class:`~biopoly.api.tenancy.RunLog` so tenants
are isolated (see [`docs/MULTI_TENANCY.md`](../../../docs/MULTI_TENANCY.md)).

The champion model is loaded once at startup from the registry. Pydantic validates
every request and validation failures return typed 422s; a missing model returns 503.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from biopoly import TARGETS, __version__
from biopoly.api.models import Candidate, DesignRequest, DesignResponse
from biopoly.api.tenancy import (
    TenantContext,
    enforce_quota,
    get_run_log,
    require_admin,
    require_tenant,
)
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
    """Liveness probe (open, no auth): status, package version and served model version."""
    return {
        "status": "ok",
        "biopoly_version": __version__,
        "model_version": STATE["version"],
        "targets": TARGETS,
    }


Tenant = Annotated[TenantContext, Depends(require_tenant)]
AdminTenant = Annotated[TenantContext, Depends(require_admin)]


@app.get("/whoami", response_model=TenantContext)
def whoami(ctx: Tenant) -> TenantContext:
    """Echo the authenticated tenant — a quick check that a key resolves as expected."""
    return ctx


def _record(tenant_id: str, kind: str, summary: dict) -> None:
    """Audit a run, fail-open: telemetry must never break the request it describes."""
    try:
        get_run_log().record(tenant_id, kind, summary)
    except Exception:  # pragma: no cover - defensive; audit is best-effort
        pass


@app.post("/predict", response_model=PredictResponse)
def predict(body: FormulationInput, ctx: Tenant) -> PredictResponse:
    """Predict the five properties (with p10/p90 bands) for a formulation."""
    model = _require_model()
    enforce_quota(ctx)
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
    _record(
        ctx.tenant_id, "predict", {"frac": frac, "tensile": round(pred[TARGETS[0]]["value"], 2)}
    )
    return PredictResponse(
        predictions={
            t: PropertyPrediction(value=pred[t]["value"], p10=pred[t]["p10"], p90=pred[t]["p90"])
            for t in TARGETS
        },
        warnings=warnings,
    )


@app.post("/design", response_model=DesignResponse)
def design(body: DesignRequest, ctx: Tenant) -> DesignResponse:
    """Return ranked candidate formulations for a target property spec."""
    model = _require_model()
    enforce_quota(ctx)
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
    _record(ctx.tenant_id, "design", {"target": target, "n_candidates": len(cands)})
    return DesignResponse(
        method=body.method,
        target=target,
        candidates=[Candidate(**c) for c in cands],
    )


@app.get("/history")
def history(ctx: Tenant, limit: int = 50) -> dict:
    """This tenant's past runs — the tenant-scoped access layer never returns another's."""
    rows = get_run_log().history(ctx.tenant_id, limit=limit)
    return {
        "tenant_id": ctx.tenant_id,
        "runs": [{"kind": r.kind, "ts": r.ts, "summary": r.summary} for r in rows],
    }


@app.get("/admin/usage")
def admin_usage(ctx: AdminTenant) -> dict:
    """Per-tenant call counts across all tenants — admin role only."""
    return {"usage": get_run_log().usage()}
