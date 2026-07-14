# biopoly roadmap

Where the project is and where it's going. This is a **synthetic-data demo** (see the
[README](README.md)); the roadmap tracks the *engineering and modelling* direction, not a product.

## Shipped
- **Forward model** — multi-target quantile GBDT with native categorical/missing handling.
- **Uncertainty** — p10–p90 quantile bands, **conformally calibrated (CQR)** for a finite-sample
  coverage guarantee ([`models/conformal.py`](src/biopoly/models/conformal.py)).
- **Inverse design** — search on the forward model (baseline sampling → warm-started Bayesian
  optimisation).
- **MLOps** — experiment tracking behind a protocol, a filesystem model registry with a
  **calibration-aware** register-if-better gate (accuracy minus an interval-coverage penalty) + rollback,
  KS/PSI **drift monitoring**, a retrain trigger, Docker, typed FastAPI.
- **Signal processing** — synthetic DSC thermograms → `scipy.signal` DSP → melt-peak feature
  extraction ([`signals.py`](src/biopoly/signals.py)).
- **Seasonality → model** — the seasonal feedstock signal ([`timeseries.py`](src/biopoly/timeseries.py))
  is wired into the generator and in as a `feedstock_quality` covariate, with an STL decomposition
  figure and a seasonal-naïve baseline; the supplier shift reads as a regime change on top.
- **Signal features → model** — a realized-crystallinity latent the recipe can't see, recovered by the
  DSC features: a with-vs-without ablation lifts clarity and biodegradation R² (see `RESULTS.md`).
- **Active learning** — query-by-committee acquisition (epistemic disagreement) + a "propose next
  experiment" search, reusing the inverse loop ([`active_learning.py`](src/biopoly/active_learning.py)).
  Benchmarked honestly: on this synthetic problem it does *not* beat random; the machinery is in place
  for domains with genuine label scarcity / shift.
- **Learned polymer representation** ([`representation.py`](src/biopoly/representation.py)) — a
  supervised per-polymer embedding whose cosine geometry is interpretable (PHA~PBS closest, PLA~TPS
  most opposed); an honest ablation shows it does not beat the descriptors predictively, so it serves
  as an interpretability tool.
- **Retrain on drift — end-to-end** ([`monitoring/retrain.py`](src/biopoly/monitoring/retrain.py)) — a
  pre-shift champion degrades on the post-shift regime (melt-flow index R² 0.73) and retraining on the
  new data recovers it (0.79), validated on held-out post-shift data; the drift alert drives the action
  and `register_if_better` promotes only on a win.
- **Testing** — a spiral-layered suite reporting a **complexity frontier**
  ([`tests/conftest.py`](tests/conftest.py)).
- **Real-data seed + sim-to-real validation** — [`DATA_STRATEGY.md`](DATA_STRATEGY.md) + a literature
  seed ([`data/real_seed.csv`](data/real_seed.csv), sourced tensile) anchoring the synthetic generator
  (~12% median gap; flags PCL tensile), plus real PLA/PBAT & PLA/PBS blends
  ([`data/real_formulations.csv`](data/real_formulations.csv)) enriched with processing metadata into
  **schema-complete rows**. The synthetic-trained model is scored on them as an out-of-distribution
  **sim-to-real** check (tensile MAE + calibrated-band coverage), with an honest leave-one-out
  augmentation experiment and an opt-in `biopoly-train --augment-real`
  ([`real_seed.py`](src/biopoly/data/real_seed.py)).
- **Multi-tenant API (isolation + guardrails)** ([`api/tenancy.py`](src/biopoly/api/tenancy.py)) — every
  endpoint but `/health` resolves an `X-API-Key` to a `TenantContext` at one auth choke-point; all tenant
  data flows through a single tenant-scoped access layer (`/history` returns only the caller's runs),
  with per-tenant daily quotas and a fail-open audit trail, and role-gated `/admin/usage`. Step 1 + the
  guardrails of step 2 of [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md).

## Next

### Data & signal realism
- **Complete the real formulations' property set.** The blends currently report tensile only; source
  the remaining targets (MFI, 60-day biodegradation, water absorption, optical clarity) with primary
  citations so they become full multi-target training rows rather than tensile-only, and revisit a
  small fine-tune / residual-correction on top of the synthetic model now that the sim-to-real gap is
  measured.

### Modelling depth
- **Inverse design — MCTS *(benched — separate stage)*.** A Monte-Carlo tree-search escalation
  alongside Bayesian optimisation for harder, multi-modal target specs. Powerful, but expensive to do
  well (a real tree policy + rollouts over the formulation space, and an honest benchmark vs the
  warm-started TPE we already have). Deliberately deferred to its own implementation stage rather than
  bolted on.

### Productionisation
- **Multi-tenancy — DB-level hardening + per-tenant models.** API-side isolation, quotas and audit are
  built (above); still to come are **Postgres row-level security** (the access layer already carries the
  tenant filter on every query, so `SET app.tenant_id` + RLS policies drop in underneath it once there
  is a real database) and short-lived **JWTs** for the frontend, then **per-tenant calibration/models**
  keyed by `tenant_id` in the registry (see [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md)).

### Engineering hygiene
- Google-style docstrings (ruff `D`), `mypy`, a coverage gate (`pytest-cov`), and expanded CI stages
  (format-check → lint → type → test).
