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
- **Inverse design — MCTS.** Add a Monte-Carlo tree-search escalation alongside Bayesian
  optimisation for harder, multi-modal target specs.
- **Learned polymer representation.** Compare learned embeddings against the current descriptor
  features (cosine-similarity analysis).

### Productionisation
- **Multi-tenant frontend.** A minimal Streamlit login demo against the API (design in
  [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md)).

### Engineering hygiene
- Google-style docstrings (ruff `D`), `mypy`, a coverage gate (`pytest-cov`), and expanded CI stages
  (format-check → lint → type → test).
