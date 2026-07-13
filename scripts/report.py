"""Generate the analysis report: figures (PNG) + a markdown summary.

Replaces a notebook with a plain, reproducible script. It builds the dataset,
trains the forward model, runs inverse design and the drift check, saves figures to
``docs/figures/`` and writes ``docs/RESULTS.md`` — so results are viewable on GitHub
without running anything, and regenerated with one command:

    uv run python scripts/report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biopoly import TARGETS
from biopoly.config import settings
from biopoly.data.generate import build_dataset
from biopoly.features import split
from biopoly.inverse import bayesopt
from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import evaluate, summary_row
from biopoly.monitoring.drift import detect_drift, format_report

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"


def _fig_target_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 2.8))
    for ax, t in zip(axes, TARGETS, strict=True):
        df[t].dropna().hist(bins=30, ax=ax, color="#4C72B0")
        ax.set_title(t.replace("_", "\n"), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Target distributions (synthetic)", fontsize=10)
    fig.tight_layout()
    path = FIG / "target_distributions.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_pred_vs_actual(df: pd.DataFrame, preds, metrics) -> str:
    t = "tensile_strength_mpa"
    y = df[t].to_numpy()
    v, lo, hi = preds[t]["value"], preds[t]["p10"], preds[t]["p90"]
    order = np.argsort(y)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(y, v, s=8, alpha=0.5)
    lim = [0, max(y.max(), v.max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("actual")
    ax[0].set_ylabel("predicted")
    ax[0].set_title(f"{t}  (R2={metrics[t]['r2']:.2f})")
    ax[1].fill_between(range(len(y)), lo[order], hi[order], alpha=0.25, label="p10-p90")
    ax[1].plot(y[order], lw=1, label="actual")
    ax[1].set_title("uncertainty band")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "pred_vs_actual.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_feature_importance(model: ForwardModel) -> str:
    imp = model.feature_importances()
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 3))
    for ax, tgt in zip(axes, TARGETS, strict=True):
        s = pd.Series(imp[tgt]).sort_values().tail(6)
        s.plot.barh(ax=ax, color="#55A868")
        ax.set_title(tgt.replace("_", "\n"), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature importance per target (gain)", fontsize=10)
    fig.tight_layout()
    path = FIG / "feature_importance.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_signal_processing() -> str:
    """Raw vs processed synthetic DSC thermogram with detected melt peaks annotated."""
    from scipy.signal import find_peaks

    from biopoly import signals

    rng = np.random.default_rng(0)
    x, raw = signals.synth_dsc({"PLA": 0.6, "PCL": 0.4}, nucleating=0.02, rng=rng)
    proc = signals.process_signal(x, raw)
    peaks, _ = find_peaks(proc, prominence=0.02)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x, raw, color="#C44E52", lw=1)
    ax[0].set_title("raw DSC thermogram (noise + sloping baseline)", fontsize=9)
    ax[0].set_xlabel("temperature (C)")
    ax[0].set_ylabel("heat flow")
    ax[1].plot(x, proc, color="#4C72B0", lw=1.2)
    ax[1].plot(x[peaks], proc[peaks], "v", color="#DD8452", ms=9)
    for p in peaks:
        ax[1].annotate(
            f"{x[p]:.0f} C", (x[p], proc[p]), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8,
        )
    ax[1].set_title("baseline-corrected + Savitzky-Golay, peaks detected", fontsize=9)
    ax[1].set_xlabel("temperature (C)")
    fig.suptitle("Signal processing: melt peaks recovered from a PLA/PCL thermogram", fontsize=10)
    fig.tight_layout()
    path = FIG / "signal_processing.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_signal_ablation() -> tuple[str, dict, dict]:
    """With-vs-without DSC signal features: do they recover the crystallinity latent?"""
    from biopoly.data.schema import FEATURE_COLS, SIGNAL_FEATURES
    from biopoly.models.metrics import evaluate

    df = build_dataset(settings, with_signal_features=True)
    tr, te = split(df, test_size=0.2, mode="random", seed=settings.seed)
    base = ForwardModel(feature_cols=FEATURE_COLS).fit(tr)
    sig = ForwardModel(feature_cols=FEATURE_COLS + SIGNAL_FEATURES).fit(tr)
    mb, ms = evaluate(te, base.predict(te)), evaluate(te, sig.predict(te))

    xb = np.arange(len(TARGETS))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(xb - 0.2, [mb[t]["r2"] for t in TARGETS], 0.4, label="recipe only", color="#C44E52")
    ax.bar(
        xb + 0.2, [ms[t]["r2"] for t in TARGETS], 0.4, label="recipe + DSC signal", color="#4C72B0"
    )
    ax.set_xticks(xb)
    ax.set_xticklabels([t.split("_")[0] for t in TARGETS], fontsize=8)
    ax.set_ylabel("R² on held-out test")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Signal ablation: DSC recovers crystallinity-driven properties", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "signal_ablation.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name, mb, ms


def _fig_seasonality(df: pd.DataFrame) -> tuple[str, float, float, float]:
    """STL decomposition of the monthly feedstock-quality signal + a seasonal-naive backtest.

    Returns the figure name plus (seasonal_strength, seasonal_naive_mae, monthly_std) so the
    forecast baseline is reported honestly next to the decomposition.
    """
    from biopoly.timeseries import monthly_mean, seasonal_naive_forecast, stl_decompose

    s = monthly_mean(df, "feedstock_quality")
    dec = stl_decompose(s.to_numpy(), period=12)

    # Honest baseline: hold out the last 12 months, forecast them by repeating the
    # prior season, and measure the error any ML forecaster would have to beat.
    naive_mae = float("nan")
    if len(s) >= 24:
        train, actual = s.to_numpy()[:-12], s.to_numpy()[-12:]
        fc = seasonal_naive_forecast(train, period=12, horizon=12)
        naive_mae = float(np.mean(np.abs(fc - actual)))

    idx = s.index
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, (name, series, colour) in zip(
        axes,
        [
            ("observed", dec.observed, "#4C72B0"),
            ("trend", dec.trend, "#8172B3"),
            ("seasonal", dec.seasonal, "#55A868"),
            ("resid", dec.resid, "#C44E52"),
        ],
        strict=True,
    ):
        ax.plot(idx, series, color=colour, lw=1.3)
        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].set_title(
        f"Feedstock-quality seasonality — STL decomposition "
        f"(seasonal strength {dec.seasonal_strength:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    path = FIG / "seasonality.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name, dec.seasonal_strength, naive_mae, float(s.std())


def _fig_active_learning(df: pd.DataFrame) -> tuple[str, list, np.ndarray, np.ndarray, dict]:
    """Active-learning-under-shift curves (active vs random, seed-averaged) + next experiment."""
    from biopoly.active_learning import Committee, active_learning_shift_curve, propose_experiment

    params = {"n_estimators": 100, "learning_rate": 0.08}
    runs = [
        active_learning_shift_curve(
            df, seed_size=40, batch=25, rounds=5, k=4, params=params, seed=s
        )
        for s in (0, 1, 2)
    ]
    labels = runs[0]["labels"]
    active = np.mean([r["active"] for r in runs], axis=0)
    random_sel = np.mean([r["random"] for r in runs], axis=0)

    train, _ = split(df, test_size=0.2, mode="random", seed=0)
    committee = Committee.fit(train, k=5, seed=1)
    proposed = propose_experiment(committee, n_candidates=2500, top_k=1, seed=7)[0]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(labels, active, "o-", color="#4C72B0", label="active (max disagreement)")
    ax.plot(labels, random_sel, "s--", color="#C44E52", label="random")
    ax.set_xlabel("labelled experiments (seed = pre-shift S1)")
    ax.set_ylabel("mean R² on post-shift (S2) test")
    ax.set_title("Committee-disagreement active learning vs random (3 seeds)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG / "active_learning.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name, labels, active, random_sel, proposed


def _metrics_table_md(metrics) -> str:
    rows = ["| target | n | MAE | RMSE | R² | coverage | within-tol |",
            "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        m = metrics[t]
        rows.append(
            f"| {t} | {m['n']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} "
            f"| {m['interval_coverage']:.2f} | {m['within_tolerance']:.2f} |"
        )
    rows.append(f"\n**mean R² = {summary_row(metrics):.3f}**")
    return "\n".join(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = build_dataset(settings)
    train_df, test_df = split(df, test_size=0.2, mode="random", seed=settings.seed)
    model = ForwardModel().fit(train_df)
    preds = model.predict(test_df)
    metrics = evaluate(test_df, preds)

    f_dist = _fig_target_distributions(df)
    f_pva = _fig_pred_vs_actual(test_df, preds, metrics)
    f_imp = _fig_feature_importance(model)
    f_sig = _fig_signal_processing()
    f_seas, seas_strength, naive_mae, seas_std = _fig_seasonality(df)
    f_abl, abl_base, abl_sig = _fig_signal_ablation()
    f_al, al_labels, al_active, al_random, al_proposed = _fig_active_learning(df)

    abl_rows = "\n".join(
        f"| {t.split('_')[0]} | {abl_base[t]['r2']:.3f} | {abl_sig[t]['r2']:.3f} "
        f"| {abl_sig[t]['r2'] - abl_base[t]['r2']:+.3f} |"
        for t in TARGETS
    )
    abl_base_mean = float(np.mean([abl_base[t]["r2"] for t in TARGETS]))
    abl_sig_mean = float(np.mean([abl_sig[t]["r2"] for t in TARGETS]))

    # inverse design
    achievable = {
        "tensile_strength_mpa": 50.0,
        "optical_clarity_pct": 80.0,
        "water_absorption_pct": 1.0,
    }
    inv = bayesopt.design(model, achievable, n_trials=400, top_k=1, seed=3)[0]
    conflict = {"tensile_strength_mpa": 55.0, "biodegradation_60d_pct": 85.0}
    inv_conf = bayesopt.design(model, conflict, n_trials=400, top_k=1, seed=5)[0]

    # drift
    ref, cur = df[df.supplier_batch == "S1"], df[df.supplier_batch == "S2"]
    drift = detect_drift(
        ref, cur, ["tensile_strength_mpa", "melt_flow_index_g10min", "frac_PBS", "primary_polymer"]
    )

    md = f"""# Results (synthetic data)

> Auto-generated by `scripts/report.py`. **Synthetic, physics-informed data — not
> real-company data.** See [`DATA_CARD.md`](../DATA_CARD.md). Numbers are illustrative.

## Data
{df.shape[0]} rows, {df.shape[1]} columns; missing per target (structured, not-at-random):
{", ".join(f"{t.split("_")[0]} {df[t].isna().mean() * 100:.0f}%" for t in TARGETS)}.

![target distributions](figures/{f_dist})

## Forward model (LightGBM, quantile uncertainty)
{_metrics_table_md(metrics)}

![predicted vs actual](figures/{f_pva})
![feature importance](figures/{f_imp})

Note: processing temperature dominates tensile strength — physically correct.

## Characterisation as signal (synthetic DSC)
Polymer characterisation is signal, not tabular. A synthetic DSC thermogram per formulation is
baseline-corrected, Savitzky-Golay smoothed and peak-detected
([`signals.py`](../src/biopoly/signals.py)) to recover melt temperatures and crystallinity — the
features a scientist actually reads off the instrument, rather than hand-waving the measurement.

![signal processing](figures/{f_sig})

### Do the signal features help? An ablation
A **realized-crystallinity latent** — batch/thermal-history variation the nominal recipe does *not*
capture — drives haze and slow degradation, so a recipe-only model cannot see it. The DSC-derived
features recover it: adding them lifts **optical clarity** and **biodegradation** R² materially, at
a small honest cost on tensile (for which the thermogram is just noise). This is what makes
characterisation signal worth extracting — made measurable.

| target | recipe only | + DSC signal | Δ R² |
|---|---|---|---|
{abl_rows}
| **mean** | {abl_base_mean:.3f} | {abl_sig_mean:.3f} | {abl_sig_mean - abl_base_mean:+.3f} |

![signal ablation](figures/{f_abl})

## Seasonality (feedstock quality over time)
Bio-feedstock purity follows an annual cycle (harvest -> storage -> depletion) on a slow
trend. It is wired into the generator
([`timeseries.py`](../src/biopoly/timeseries.py)) as a `feedstock_quality` covariate that
shifts tensile strength, so the model sees real temporal structure — and the mid-2025 supplier
shift reads as a **regime change on top** of this baseline. STL cleanly separates trend /
seasonal / residual (seasonal strength **{seas_strength:.2f}**). The honest forecast baseline
any ML model must beat is **seasonal-naive** ("next September looks like last September"): MAE
**{naive_mae:.4f}** over a 12-month backtest, against a monthly signal SD of **{seas_std:.4f}**.

![feedstock-quality seasonality](figures/{f_seas})

## Inverse design (target spec -> formulation)
**Achievable target** `{achievable}`
- predicted: `{inv["predicted"]}`
- formulation: `{inv["formulation"]}`

**Conflicting target** `{conflict}` (high tensile *and* high biodegradation pull apart) —
returns the best compromise:
- predicted: `{inv_conf["predicted"]}`
- formulation: `{inv_conf["formulation"]}`

## Active learning (choosing the most informative next experiment)
Data is the binding constraint, so the next experiment should be the most *informative* one — chosen
by **expected information gain**, estimated as disagreement across a bootstrap committee of forward
models (*epistemic* uncertainty, the reducible kind — not the aleatoric measurement noise the
quantile band already captures). This reuses the inverse-design search with its objective flipped
from "hit a target" to "learn the most"
([`active_learning.py`](../src/biopoly/active_learning.py)); its concrete output is a recommended
next formulation to run.

**Does it beat random?** Benchmarked honestly — seeding from pre-shift (S1) data and scoring on the
post-shift (S2) regime. On this synthetic problem (a strong GBDT, a pool from the same distribution
as the test) committee-disagreement selection did **not** outperform random: mean R²
**{al_active[-1]:.3f}** vs **{al_random[-1]:.3f}** on the post-shift test (3 seeds). That is the
honest result — uncertainty sampling's advantage needs genuine label sparsity or a sharper epistemic
signal than a bootstrap committee provides here. The acquisition machinery is in place for the
domains (costly labels, real distribution shift) where it pays.

![active learning vs random](figures/{f_al})

**Proposed next experiment** (highest information gain): `{al_proposed["formulation"]}`

## Drift monitoring (S1 reference vs S2 incoming)
```
{format_report(drift)}
```
"""
    (DOCS / "RESULTS.md").write_text(md, encoding="utf-8")
    print("wrote docs/RESULTS.md and 7 figures under docs/figures/")
    print(f"mean R2 = {summary_row(metrics):.3f}; drift alert = {drift['alert']}")


if __name__ == "__main__":
    main()
