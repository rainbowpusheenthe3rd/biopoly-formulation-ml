# biopoly roadmap

Where the project is and where it's going. This is a **synthetic-data demo** (see the
[README](README.md)); the roadmap tracks the *engineering and modelling* direction, not a product.

## Shipped
- **Forward model** — multi-target quantile GBDT with native categorical/missing handling.
- **Uncertainty** — p10–p90 quantile bands, **conformally calibrated (CQR)** for a finite-sample
  coverage guarantee ([`models/conformal.py`](src/biopoly/models/conformal.py)).
- **Inverse design** — search on the forward model (baseline sampling → warm-started Bayesian
  optimisation).
- **MLOps** — experiment tracking behind a protocol, a filesystem model registry
  (register-if-better + rollback), KS/PSI **drift monitoring**, a retrain trigger, Docker, typed
  FastAPI.
- **Signal processing** — synthetic DSC thermograms → `scipy.signal` DSP → melt-peak feature
  extraction ([`signals.py`](src/biopoly/signals.py)).
- **Testing** — a spiral-layered suite reporting a **complexity frontier**
  ([`tests/conftest.py`](tests/conftest.py)).

## Next

### Data & signal realism
- **Seasonality → model.** The seasonal feedstock signal ([`timeseries.py`](src/biopoly/timeseries.py))
  is standalone today; wire it into the generator and in as a model covariate, add an STL
  decomposition figure and a seasonal-naïve forecast baseline. Drift then reads as a regime change
  *on top* of the seasonal baseline.
- **Signal features → model.** Feed the DSC-derived features into the forward model and quantify
  their lift (feature importance / with-vs-without ablation).
- **Data strategy.** A `DATA_STRATEGY.md` — synthetic → literature-derived priors → **active
  learning** (reuse the inverse-design loop to choose the next most-informative experiment) → a
  small real seed, with honest limitations.

### Modelling depth
- **Drift → retrain, end-to-end.** Exercise the full detect → retrain → validate → register loop
  against the built-in mid-2025 supplier-purity shift.
- **Inverse design — MCTS.** Add a Monte-Carlo tree-search escalation alongside Bayesian
  optimisation for harder, multi-modal target specs.
- **Learned polymer representation.** Compare learned embeddings against the current descriptor
  features (cosine-similarity analysis).

### Productionisation
- **Multi-tenant frontend.** A minimal Streamlit login demo against the API (design in
  [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md)).
- **Promotion gate.** Fold interval-coverage error into `register_if_better` — today it promotes on
  mean R² only, so a calibration-only improvement won't auto-promote.

### Engineering hygiene
- Google-style docstrings (ruff `D`), `mypy`, a coverage gate (`pytest-cov`), and expanded CI stages
  (format-check → lint → type → test).
