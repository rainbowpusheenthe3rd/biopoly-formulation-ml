"""Multi-tenant identity, isolation and guardrails for the API.

Implements step 1 of [`docs/MULTI_TENANCY.md`](../../../docs/MULTI_TENANCY.md) at demo
scale: resolve every request to a :class:`TenantContext` from an API key (the single
auth choke-point), and route all tenant data through one **tenant-scoped data-access
layer** (:class:`RunLog`) whose every method takes a ``tenant_id`` — so cross-tenant
reads can't be written by accident. A fail-open audit trail and per-tenant daily quotas
round out the guardrails.

There is no database in this synthetic demo, so isolation is enforced in the access
layer rather than by Postgres row-level security; RLS is the production hardening
documented in the design note (the access layer is written so RLS drops in underneath
it unchanged — every query already carries the tenant filter).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Annotated, Literal

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel

from biopoly.config import settings

Role = Literal["scientist", "admin"]

# Default demo tenants, used when BIOPOLY_API_KEYS is unset. Two tenants so isolation is
# demonstrable, plus an admin key. These are illustrative demo secrets, not real keys.
DEFAULT_API_KEYS: dict[str, tuple[str, str, Role]] = {
    "demo-acme-key": ("acme", "Acme Compostables", "scientist"),
    "demo-globex-key": ("globex", "Globex Bioplastics", "scientist"),
    "demo-admin-key": ("acme", "Acme Compostables", "admin"),
}


class TenantContext(BaseModel):
    """The resolved caller: which tenant, and their role. Injected into every endpoint."""

    tenant_id: str
    name: str
    role: Role = "scientist"


@dataclass
class RunRecord:
    """One tenant-attributed API call — the append-only audit/history row."""

    tenant_id: str
    kind: str  # "predict" | "design"
    ts: str  # ISO-8601 UTC
    day: str  # YYYY-MM-DD (for daily quota counting)
    summary: dict


class TenantStore:
    """API-key -> :class:`TenantContext`. Rotatable per tenant; the auth source of truth."""

    def __init__(self, by_key: dict[str, TenantContext]):
        self._by_key = by_key

    @classmethod
    def from_config(cls, raw: str) -> TenantStore:
        """Build from ``BIOPOLY_API_KEYS`` (``key:tenant_id:role`` CSV) or the demo default.

        The ``name`` defaults to ``tenant_id`` for env-provided keys; the baked-in demo
        keys carry friendlier names.
        """
        by_key: dict[str, TenantContext] = {}
        if raw.strip():
            for entry in raw.split(","):
                parts = [p.strip() for p in entry.split(":")]
                if len(parts) < 2 or not parts[0]:
                    continue
                key, tenant_id = parts[0], parts[1]
                role: Role = "admin" if len(parts) > 2 and parts[2] == "admin" else "scientist"
                by_key[key] = TenantContext(tenant_id=tenant_id, name=tenant_id, role=role)
        else:
            for key, (tenant_id, name, role) in DEFAULT_API_KEYS.items():
                by_key[key] = TenantContext(tenant_id=tenant_id, name=name, role=role)
        return cls(by_key)

    def resolve(self, api_key: str) -> TenantContext | None:
        return self._by_key.get(api_key)

    def tenant_ids(self) -> list[str]:
        return sorted({ctx.tenant_id for ctx in self._by_key.values()})


class RunLog:
    """The single tenant-scoped data-access layer for prediction/design runs.

    Every read and write takes a ``tenant_id`` and filters on it, so isolation lives in
    one place instead of being re-implemented (and eventually forgotten) per endpoint.
    In production this is backed by a ``tenant_id``-columned table with Postgres RLS; here
    it is an in-memory list with the same contract.
    """

    def __init__(self) -> None:
        self._rows: list[RunRecord] = []

    def record(self, tenant_id: str, kind: str, summary: dict) -> None:
        now = datetime.now(UTC)
        self._rows.append(
            RunRecord(
                tenant_id=tenant_id,
                kind=kind,
                ts=now.isoformat(timespec="seconds"),
                day=now.date().isoformat(),
                summary=summary,
            )
        )

    def history(self, tenant_id: str, limit: int = 50) -> list[RunRecord]:
        """This tenant's runs, most recent last — never any other tenant's."""
        rows = [r for r in self._rows if r.tenant_id == tenant_id]
        return rows[-limit:]

    def count_today(self, tenant_id: str) -> int:
        today = date.today().isoformat()
        return sum(1 for r in self._rows if r.tenant_id == tenant_id and r.day == today)

    def usage(self) -> dict[str, int]:
        """Per-tenant total call counts — an admin-only aggregate view."""
        return dict(Counter(r.tenant_id for r in self._rows))


@dataclass
class _Registry:
    store: TenantStore | None = None
    run_log: RunLog = field(default_factory=RunLog)


_REG = _Registry()


def get_tenant_store() -> TenantStore:
    """The process-wide tenant store, built lazily from settings on first use."""
    if _REG.store is None:
        _REG.store = TenantStore.from_config(settings.api_keys)
    return _REG.store


def get_run_log() -> RunLog:
    """The process-wide tenant-scoped run log."""
    return _REG.run_log


def reset_tenancy() -> None:
    """Rebuild the tenant store and clear the run log (used by tests)."""
    _REG.store = None
    _REG.run_log = RunLog()


# ── FastAPI dependencies (the auth choke-point) ──────────────────────────────


def require_tenant(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TenantContext:
    """Resolve the caller to a :class:`TenantContext` from ``X-API-Key`` or reject with 401."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    ctx = get_tenant_store().resolve(x_api_key)
    if ctx is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return ctx


def require_admin(ctx: Annotated[TenantContext, Depends(require_tenant)]) -> TenantContext:
    """Require the resolved tenant to hold the ``admin`` role, else 403."""
    if ctx.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return ctx


def enforce_quota(ctx: TenantContext) -> None:
    """Raise 429 if the tenant has spent its daily call quota (0 = unlimited)."""
    limit = settings.tenant_daily_quota
    if limit > 0 and get_run_log().count_today(ctx.tenant_id) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"daily quota of {limit} calls exhausted for tenant '{ctx.tenant_id}'",
        )
