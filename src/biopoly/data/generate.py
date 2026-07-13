"""Synthetic dataset generator.

Samples realistic biopolymer formulations, runs them through the physics-informed
ground truth in :mod:`biopoly.data.chemistry`, then adds the messiness that makes
this a genuine ML problem: measurement noise + outliers, a test-protocol covariate,
structured (not-at-random) missingness, and a mid-2025 supplier-purity drift.

Run:  ``uv run biopoly-generate``  (or ``python -m biopoly.data.generate``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biopoly import TARGETS
from biopoly.config import Settings, settings
from biopoly.data.chemistry import ADDITIVES, POLYMERS, Formulation, forward_true
from biopoly.data.schema import TENSILE_PROTOCOLS
from biopoly.timeseries import seasonal_feedstock_quality

# present-probability and max fraction per additive
_ADDITIVE_SPEC = {
    "plasticizer": (0.55, 0.20),
    "nucleating": (0.30, 0.03),
    "compatibilizer": (0.25, 0.08),
    "fibre": (0.35, 0.30),
    "chain_extender": (0.20, 0.02),
}
_N_POLYMERS_P = {1: 0.30, 2: 0.50, 3: 0.20}
_TARGET_NOISE_CV = {
    "tensile_strength_mpa": 0.05,
    "melt_flow_index_g10min": 0.08,
    "biodegradation_60d_pct": 0.06,
    "water_absorption_pct": 0.07,
    "optical_clarity_pct": 0.05,
}
_ASTM_TENSILE_OFFSET = 2.0  # ASTM D638 reads slightly higher than ISO 527 here
_SHIFT_DATE = np.datetime64("2025-07-01")
# physical caps applied to *measured* values (noise must not escape physics)
_MEAS_CAP = {
    "tensile_strength_mpa": 82.0,
    "melt_flow_index_g10min": 125.0,
    "biodegradation_60d_pct": 100.0,
    "water_absorption_pct": 60.0,
    "optical_clarity_pct": 99.0,
}


def _sample_formulation(rng: np.random.Generator) -> Formulation:
    # additives first (they consume part of the mass budget)
    additive_frac: dict[str, float] = {}
    for a in ADDITIVES:
        p_present, fmax = _ADDITIVE_SPEC[a]
        if rng.random() < p_present:
            additive_frac[a] = float(rng.uniform(0.01, fmax))
    raw_add = sum(additive_frac.values())
    if raw_add > 0.40:  # keep the mass budget balanced by scaling additives down
        scale = 0.40 / raw_add
        additive_frac = {a: f * scale for a, f in additive_frac.items()}
    total_add = sum(additive_frac.values())
    matrix = 1.0 - total_add

    k = int(rng.choice(list(_N_POLYMERS_P), p=list(_N_POLYMERS_P.values())))
    chosen = list(rng.choice(POLYMERS, size=k, replace=False))
    ratios = rng.dirichlet(np.ones(k) * 1.5)
    polymer_frac = {p: float(matrix * r) for p, r in zip(chosen, ratios, strict=True)}

    from biopoly.data.chemistry import blend_optimum_temp

    opt = blend_optimum_temp(polymer_frac)
    temp = float(np.clip(rng.normal(opt, 20.0), 90.0, 255.0))
    time_min = float(np.clip(rng.exponential(18.0) + 4.0, 4.0, 110.0))
    return Formulation(polymer_frac, additive_frac, temp, time_min)


def _measure(true: dict[str, float], protocol: str, rng: np.random.Generator) -> dict[str, float]:
    out: dict[str, float] = {}
    outlier = rng.random() < 0.02
    for t in TARGETS:
        cv = _TARGET_NOISE_CV[t] * (3.0 if outlier else 1.0)
        val = true[t] * (1.0 + rng.normal(0.0, cv))
        if t == "tensile_strength_mpa" and protocol == "ASTMD638":
            val += _ASTM_TENSILE_OFFSET
        out[t] = float(np.clip(val, 0.0, _MEAS_CAP[t]))
    return out


def build_dataset(cfg: Settings | None = None) -> pd.DataFrame:
    cfg = cfg or settings
    rng = np.random.default_rng(cfg.seed)
    dates = pd.to_datetime(
        rng.uniform(
            pd.Timestamp(cfg.start_date).value,
            pd.Timestamp(cfg.end_date).value,
            size=cfg.n_samples,
        )
    )
    # Seasonal bio-feedstock quality per sample date — a real temporal covariate the
    # forward model can use, and the baseline the mid-2025 shift is a regime change on.
    # Drawn from a dedicated RNG so it does not perturb the formulation-sampling stream.
    feedstock_quality = seasonal_feedstock_quality(dates, rng=np.random.default_rng(cfg.seed + 1))

    rows: list[dict] = []
    for i in range(cfg.n_samples):
        form = _sample_formulation(rng)
        date = dates[i]
        quality = float(feedstock_quality[i])
        after_shift = np.datetime64(date) >= _SHIFT_DATE and form.polymer_frac.get("PBS", 0) > 0
        true = forward_true(form, after_supplier_shift=bool(after_shift), feedstock_quality=quality)
        protocol = str(rng.choice(TENSILE_PROTOCOLS))
        meas = _measure(true, protocol, rng)

        primary = max(form.polymer_frac, key=form.polymer_frac.get)
        row = form.as_row()
        row.update(meas)
        row["primary_polymer"] = primary
        row["tensile_protocol"] = protocol
        row["feedstock_quality"] = quality
        row["sample_id"] = f"NN-{i:05d}"
        row["date"] = date
        row["supplier_batch"] = "S2" if np.datetime64(date) >= _SHIFT_DATE else "S1"
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- structured, not-at-random missingness ---
    # biodegradation@60d: a slow, costly test; abandoned more often for slow (PLA-rich) samples.
    slow = df["biodegradation_60d_pct"] < 30
    miss_p = np.where(slow, 0.55, 0.22)
    df.loc[rng.random(len(df)) < miss_p, "biodegradation_60d_pct"] = np.nan
    # optical clarity: not recorded when the sample is visibly opaque (fillers / phase sep).
    opaque = (df["optical_clarity_pct"] < 25) | (df["add_fibre"] > 0.15)
    clar_miss_p = np.where(opaque, 0.7, 0.05)
    df.loc[rng.random(len(df)) < clar_miss_p, "optical_clarity_pct"] = np.nan

    return df.sort_values("date").reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    lines = [f"rows={len(df)}  cols={df.shape[1]}", "missing %:"]
    for t in TARGETS:
        lines.append(
            f"  {t:26s} {df[t].isna().mean() * 100:5.1f}%  [{df[t].min():.1f}, {df[t].max():.1f}]"
        )
    lines.append(f"supplier batches: {df['supplier_batch'].value_counts().to_dict()}")
    return "\n".join(lines)


def main() -> None:
    settings.ensure_dirs()
    df = build_dataset(settings)
    df.to_parquet(settings.data_path, index=False)
    print(f"wrote {settings.data_path}")
    print(summarise(df))


if __name__ == "__main__":
    main()
