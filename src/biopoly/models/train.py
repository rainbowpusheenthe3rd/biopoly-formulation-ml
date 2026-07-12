"""Train the forward model, evaluate, log to the tracker, and register it.

uv run biopoly-train                 # fast: default hyper-parameters
uv run biopoly-train --hpo --trials 30   # Optuna hyper-parameter search
BIOPOLY_TRACKING_BACKEND=noop uv run biopoly-train   # CI (no server)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from biopoly import TARGETS
from biopoly.config import settings
from biopoly.features import split
from biopoly.models.conformal import ConformalCalibrator
from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import evaluate, format_table, summary_row
from biopoly.models.registry import ModelRegistry
from biopoly.tracking.base import get_tracker


def _load() -> pd.DataFrame:
    if not settings.data_path.exists():
        from biopoly.data.generate import main as gen

        gen()
    return pd.read_parquet(settings.data_path)


def run_hpo(train_df: pd.DataFrame, n_trials: int) -> dict:
    """Small Optuna search on the median models (fast proxy for the full ensemble)."""
    import optuna

    tr, va = split(train_df, test_size=0.2, mode="random", seed=1)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 40),
        )
        model = ForwardModel(params).fit(tr)
        return summary_row(evaluate(va, model.predict(va)))  # mean R2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"best mean R2 (val) = {study.best_value:.3f}  params = {study.best_params}")
    return study.best_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpo", action="store_true", help="run Optuna hyper-parameter search")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    args = ap.parse_args()

    settings.ensure_dirs()
    df = _load()
    train_df, test_df = split(df, test_size=0.2, mode=args.split, seed=settings.seed)
    # Carve a calibration split out of train for CQR — the calibrator must see
    # data the forward model never trained on, or the coverage guarantee is void.
    fit_df, cal_df = split(train_df, test_size=0.2, mode=args.split, seed=settings.seed + 1)

    params = run_hpo(fit_df, args.trials) if args.hpo else {}
    model = ForwardModel(params).fit(fit_df)

    # Raw (pre-conformal) test coverage, for the before/after comparison.
    raw_metrics = evaluate(test_df, model.predict(test_df))

    # Fit and attach the CQR calibrator; predict() now serves calibrated bands.
    model.conformal_ = ConformalCalibrator(alpha=0.20).fit(cal_df, model.predict(cal_df))

    metrics = evaluate(test_df, model.predict(test_df))
    mean_r2 = summary_row(metrics)
    print(format_table(metrics))
    print("\nconformal calibration (nominal 80% band, test coverage):")
    for t in TARGETS:
        adj = model.conformal_.adjustments_[t]
        print(
            f"  {t:26s} raw {raw_metrics[t]['interval_coverage']:.2f} "
            f"-> conformal {metrics[t]['interval_coverage']:.2f}  "
            f"(width {raw_metrics[t]['mean_interval_width']:.2f} "
            f"-> {metrics[t]['mean_interval_width']:.2f}, Q={adj:+.2f})"
        )

    model_dir = model.save(settings.artifact_dir / "forward_latest")
    model.conformal_.save(model_dir)  # human-readable adjustments alongside the pickle
    (settings.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with get_tracker(run_name=f"forward-{args.split}") as tracker:
        tracker.log_params({"split": args.split, "hpo": args.hpo, **params})
        tracker.log_metrics({f"{t}_r2": metrics[t]["r2"] for t in TARGETS})
        tracker.log_metrics({f"{t}_mae": metrics[t]["mae"] for t in TARGETS})
        tracker.log_metrics({f"{t}_coverage": metrics[t]["interval_coverage"] for t in TARGETS})
        tracker.log_metrics({"mean_r2": mean_r2})
        tracker.log_artifacts(model_dir)

    version, promoted = ModelRegistry().register_if_better(model_dir, metrics, mean_r2)
    print(f"registered v{version}; champion={'yes' if promoted else 'no'} (mean R2={mean_r2:.3f})")


if __name__ == "__main__":
    main()
