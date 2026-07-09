"""Retraining trigger: detect drift -> retrain -> validate -> register if better.

This is the CI/scheduled retraining job. It simulates "new lab records
arrived" by treating the post-2025-07 batch (S2) as the new data and the earlier
batch (S1) as the reference the champion was trained on. If drift is detected (the
supplier-purity shift), it retrains on *all* data, evaluates on a held-out split,
and promotes the model only if it beats the current champion.

    uv run python scripts/retrain.py
"""

from __future__ import annotations

import pandas as pd

from biopoly.config import settings
from biopoly.data.schema import TARGETS
from biopoly.features import split
from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import evaluate, format_table, summary_row
from biopoly.models.registry import ModelRegistry
from biopoly.monitoring.drift import detect_drift, format_report

MONITORED = [
    "tensile_strength_mpa",
    "melt_flow_index_g10min",
    "frac_PBS",
    "primary_polymer",
]


def main() -> None:
    settings.ensure_dirs()
    df = pd.read_parquet(settings.data_path)

    reference = df[df["supplier_batch"] == "S1"]
    current = df[df["supplier_batch"] == "S2"]
    print("== drift check (reference S1 vs incoming S2) ==")
    report = detect_drift(reference, current, MONITORED)
    print(format_report(report))

    if not report["alert"]:
        print("\nno drift -> no retrain needed.")
        return

    print("\ndrift detected -> retraining on all available data...")
    train_df, test_df = split(df, test_size=0.2, mode="random", seed=settings.seed)
    model = ForwardModel().fit(train_df)
    metrics = evaluate(test_df, model.predict(test_df))
    mean_r2 = summary_row(metrics)
    print(format_table(metrics))

    model_dir = model.save(settings.artifact_dir / "forward_candidate")
    version, promoted = ModelRegistry().register_if_better(model_dir, metrics, mean_r2)
    verdict = "PROMOTED to champion" if promoted else "kept as candidate (did not beat champion)"
    print(
        f"\nregistered v{version}: {verdict} (mean R2={mean_r2:.3f}) across {len(TARGETS)} targets"
    )


if __name__ == "__main__":
    main()
