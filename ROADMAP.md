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

## Next

### Data & signal realism
- **Real-data seed — started.** [`DATA_STRATEGY.md`](DATA_STRATEGY.md) + a small literature seed
  ([`data/real_seed.csv`](data/real_seed.csv), sourced tensile with citations) used to *anchor* the
  synthetic generator (~12% median gap; flags PCL tensile as a calibration target). Next: primary
  citations + real *formulations* with processing metadata that can actually augment training.

### Modelling depth
- **Inverse design — MCTS *(benched — separate stage)*.** A Monte-Carlo tree-search escalation
  alongside Bayesian optimisation for harder, multi-modal target specs. Powerful, but expensive to do
  well (a real tree policy + rollouts over the formulation space, and an honest benchmark vs the
  warm-started TPE we already have). Deliberately deferred to its own implementation stage rather than
  bolted on.

### Productionisation
- **Multi-tenant frontend — minimal built** ([`frontend/streamlit_app.py`](frontend/streamlit_app.py)):
  a login-gated Streamlit UI over the API with tenant-scoped predict / design / history (session-layer
  isolation). *Next:* the API-side tenant machinery — `TenantContext` + API-key/JWT auth + `tenant_id`
  on all rows with Postgres row-level security (see [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md)).

### Engineering hygiene
- Google-style docstrings (ruff `D`), `mypy`, a coverage gate (`pytest-cov`), and expanded CI stages
  (format-check → lint → type → test).
