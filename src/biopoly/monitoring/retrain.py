"""End-to-end retrain-on-drift cycle: the loop that fixes a regime change.

Ties the operational pieces together — detect drift (S1 vs S2) -> retrain on the
combined data -> validate on the *post-shift* regime -> (register-if-better). The
point it makes concrete: a champion trained only on pre-shift (S1) data **degrades**
on the post-shift (S2) regime, and retraining on the new data **recovers** it. That
degradation is exactly what the drift monitor flags, so the retrain trigger is
principled rather than arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import evaluate, summary_row
from biopoly.monitoring.drift import detect_drift

# Features watched for the supplier-purity shift (tensile/MFI move; frac_PBS/primary
# tell us the *composition* mix is stable, so the change is in the outputs).
MONITORED = ["tensile_strength_mpa", "melt_flow_index_g10min", "frac_PBS", "primary_polymer"]


def retrain_cycle(df: pd.DataFrame, *, test_size: float = 0.3, seed: int = 0) -> dict:
    """Run detect -> retrain -> validate against the mid-2025 supplier shift.

    The champion is trained on pre-shift (S1) data only; a slice of post-shift (S2)
    data is held out as the test set and the rest is the "newly arrived" data to
    retrain on. Both models are scored on the held-out post-shift test.

    Returns the drift report plus champion vs retrained metrics (per-target and mean
    R²) on the post-shift regime.
    """
    s1 = df[df["supplier_batch"] == "S1"]
    s2 = df[df["supplier_batch"] == "S2"].reset_index(drop=True)

    # Monitor the full incoming batch (that is what a real monitor sees) ...
    drift = detect_drift(s1, s2, MONITORED)

    # ... then hold out a post-shift validation slice and retrain on the rest.
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(s2) * test_size))
    test_pos = rng.choice(len(s2), size=n_test, replace=False)
    s2_test = s2.iloc[test_pos]
    s2_new = s2.drop(index=s2.index[test_pos])  # post-shift data available to retrain on

    champion = ForwardModel().fit(s1)  # trained on the pre-shift regime only
    retrained = ForwardModel().fit(pd.concat([s1, s2_new], ignore_index=True))

    champ_metrics = evaluate(s2_test, champion.predict(s2_test))
    retr_metrics = evaluate(s2_test, retrained.predict(s2_test))
    return {
        "drift": drift,
        "champion": champ_metrics,
        "retrained": retr_metrics,
        "champion_mean_r2": summary_row(champ_metrics),
        "retrained_mean_r2": summary_row(retr_metrics),
        "n_train_s1": int(len(s1)),
        "n_new_s2": int(len(s2_new)),
        "n_test_s2": int(len(s2_test)),
    }
