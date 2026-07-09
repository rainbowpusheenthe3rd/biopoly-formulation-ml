# 5-minute walkthrough

A script for demoing this repo. Everything runs on synthetic data the repo
generates itself — no real or proprietary data anywhere.

## 0. Framing (30s)
This is a working forward + inverse ML pipeline for materials formulation, built on
synthetic data. The generator encodes real biopolymer property ranges from the
literature so the ML problem is realistic, while keeping the whole thing shareable.

## 1. The data is a real ML problem, on purpose (60s)
Open `DATA_CARD.md`. Point out the deliberate difficulty:
- **Not-at-random missingness** — biodegradation@60d missing more often for slow (PLA-rich)
  samples where the 60-day test was abandoned. "Missing for a reason" → a process signal.
- **Protocol covariate** — tensile under ISO 527 vs ASTM D638; the model gets protocol as a
  feature so it doesn't mistake a protocol change for a real effect.
- **Supplier-purity drift** mid-2025 — the hook for drift monitoring + retraining.

## 2. Forward model — readable, not a black box (60s)
`uv run biopoly-train`. Talk to the metrics table: **mean R² ≈ 0.94**, and it reports two
things beyond R² that matter to a scientist — **p10–p90 uncertainty** on every prediction,
and **within-tolerance accuracy** against a *per-variable* margin (the acceptable error on
water absorption is nothing like the one on tensile). Then show feature importances:
processing temperature dominates tensile strength — physically right, and it's the readout
that makes the model trustworthy.

## 3. Inverse design — the headline loop (90s)
`POST /design` with a target spec → ranked formulations.
- **Baseline**: sample valid candidates, run the forward model, rank by distance-to-target.
- **Bayesian optimisation** (Optuna/TPE, warm-started from the best samples): refines toward
  spec while respecting the model's uncertainty.
Demo two targets: an **achievable** one (PLA-like → it nails tensile ≈ 50, clarity ≈ 80),
and a **conflicting** one (high tensile *and* high biodegradation) where it returns the best
compromise — which is itself the useful answer: it exposes the trade-off and proposes the
next experiment.

## 4. It's a system, not a notebook (60s)
- FastAPI + Pydantic (`/predict`, `/design`, `/health`), typed errors, `docker compose up`.
- Experiment tracking behind a protocol — MLflow default, **ClearML swappable** (the
  abstraction means either plugs in).
- Model **registry** with champion promote/rollback; `register_if_better` only promotes on
  improvement.
- **Drift monitor** catches the supplier shift; **retrain trigger** (CI `workflow_dispatch`)
  closes the loop: detect → retrain → validate → register.

## 5. What I'd do next (20s)
Conformal calibration of the intervals; learned polymer embeddings on top of the descriptor
features; active-learning loop feeding proposed experiments back as new training data.

---
**One-liners if asked "why this choice?"**
- *LightGBM* — trains in seconds on ~2k rows, native categorical + missing handling, feature
  importance. The simplest thing that does the job.
- *Quantile regression for uncertainty* — cheap, model-native, gives the scientist a band.
- *uv* — reproducible locked env, pinned Python.
- *Synthetic data* — lets the whole pipeline be demoed honestly without touching real IP.
