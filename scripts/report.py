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

## Drift monitoring (S1 reference vs S2 incoming)
```
{format_report(drift)}
```
"""
    (DOCS / "RESULTS.md").write_text(md, encoding="utf-8")
    print("wrote docs/RESULTS.md and 5 figures under docs/figures/")
    print(f"mean R2 = {summary_row(metrics):.3f}; drift alert = {drift['alert']}")


if __name__ == "__main__":
    main()
