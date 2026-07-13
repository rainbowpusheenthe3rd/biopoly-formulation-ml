# Multi-tenancy & a client-login frontend — design note

> **Status: mostly design; a minimal login frontend is built.** The repo ships a
> single-tenant demo API ([`api/main.py`](../src/biopoly/api/main.py)) plus a small
> login-gated Streamlit UI ([`frontend/streamlit_app.py`](../frontend/streamlit_app.py))
> that demonstrates the tenant UX (step 3 below). This note sketches how it would grow
> into a service that several client organisations ("tenants") log into and use in
> isolation — the natural next step once more than one customer is on it.

## The shape of the problem

A formulation service that multiple companies use has three hard requirements the
demo doesn't yet meet:

1. **Isolation** — Tenant A must never see Tenant B's formulations, targets, or
   results. This is the whole ballgame; everything else is secondary.
2. **Identity** — each request must be attributable to a tenant (and ideally a user
   within it) for authorisation, quotas and audit.
3. **A way in** — a login-gated frontend a non-engineer (e.g. a materials scientist)
   can actually use, not just `curl`.

## Design

### Auth & identity
- **Tenant API keys** for machine-to-machine (issued per tenant, rotatable), and
  **short-lived JWTs** for the interactive frontend (login → token carrying
  `tenant_id` + `role`). A FastAPI dependency resolves the caller to a
  `TenantContext(tenant_id, role)` and injects it into every endpoint — the single
  choke-point where auth is enforced.
- **Roles** kept minimal to start: `scientist` (predict/design) vs `admin` (manage
  keys, view usage). Add more only when a real need appears.

### Data isolation (the part to get right)
- Every stored row (formulations, targets, prediction logs) carries a non-null
  `tenant_id`. **All** queries filter on the caller's `tenant_id` — enforced in one
  data-access layer, never hand-written per endpoint, so isolation can't be
  forgotten. For stronger guarantees, Postgres **row-level security** (RLS) with a
  per-request `SET app.tenant_id` makes the database itself refuse cross-tenant reads.
- Start with a **shared schema + `tenant_id` column** (simplest, cheapest); escalate
  to schema-per-tenant or db-per-tenant only if a tenant's scale or compliance needs
  demand it. (Simplest thing that works first.)

### Model strategy
Three options, in ascending cost — pick per tenant, not globally:
1. **Shared model, isolated data** (default): one forward/inverse model serves all
   tenants; only their *data* is private. Cheapest, and fine when the physics is
   shared.
2. **Per-tenant calibration**: shared base model, but the **CQR calibrator**
   ([`models/conformal.py`](../src/biopoly/models/conformal.py)) and any thresholds are
   fit per tenant on their own data — better-calibrated uncertainty without retraining.
3. **Per-tenant model**: a tenant with enough proprietary data gets its own champion
   in the registry ([`models/registry.py`](../src/biopoly/models/registry.py)), keyed
   by `tenant_id`. The registry's versioning/promotion already generalises to this.

### Guardrails
- **Quotas / rate-limits** per tenant (predict/design calls per minute/day) to protect
  shared capacity and enable tiered plans.
- **Audit log** — append-only record of who ran what, when (fail-open: never blocks a
  request), reusing the drift/usage telemetry pattern.

### The frontend
A thin **login-gated UI** that talks to the API — nothing the API can't already do,
just made usable:
- Login → JWT; a tenant-scoped dashboard: submit a formulation → see predicted
  properties **with calibrated p10–p90 bands**; submit a target spec → ranked candidate
  formulations from inverse design; a history view of *this tenant's* past runs.
- **Streamlit** is the pragmatic first cut (fast to build, Python-native, speaks the
  same stack); the API stays the source of truth so the frontend is swappable.

## Phased rollout
1. `TenantContext` dependency + API-key auth + `tenant_id` on all rows and queries.
2. Postgres RLS + per-tenant quotas + audit log.
3. **Minimal Streamlit login frontend (built)** — `frontend/streamlit_app.py`: login →
   tenant-scoped predict / design / history over the API. Isolation is session-layer
   here (a demo); DB-level isolation is steps 1–2. Run:
   `uv run --extra frontend streamlit run frontend/streamlit_app.py` (API up separately).
4. Per-tenant calibration, then per-tenant models for tenants with the data to justify it.

## What exists vs what's next
- **Exists:** the single-tenant FastAPI service, the model registry (versioning/
  promotion that per-tenant models would reuse), calibrated intervals, and the minimal
  Streamlit login frontend (step 3, session-layer isolation).
- **Next (not built):** the API-side tenant machinery — `TenantContext` + API-key/JWT
  auth + `tenant_id` on all rows with Postgres RLS (steps 1–2), then per-tenant
  calibration/models (step 4).
