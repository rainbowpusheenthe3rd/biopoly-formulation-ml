"""Domain knowledge base for compostable biopolymer formulation.

This module encodes the *physics-informed ground truth* used to synthesise the
dataset. Every anchor value is a plausible mid-range figure for the neat polymer
drawn from the biopolymer literature (see ``DATA_CARD.md`` for the citations and
ranges). Blends are combined with simple, transparent mixing rules plus
interaction terms, so the resulting dataset has *learnable, non-trivial* structure
rather than pure noise.

NOTE: these numbers are illustrative anchors for a synthetic demo, not measured
values for any real product or supplier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POLYMERS: list[str] = ["PLA", "PHA", "PBAT", "PBS", "TPS", "PCL"]
ADDITIVES: list[str] = ["plasticizer", "nucleating", "compatibilizer", "fibre", "chain_extender"]

# Neat-polymer anchors: tensile (MPa), melt-flow index (g/10min), biodegradation
# at 60 days (% mass loss, industrial composting), water absorption (%), optical
# clarity (% transmittance). See DATA_CARD.md for literature ranges per figure.
ANCHORS: dict[str, dict[str, float]] = {
    "PLA": {"tensile": 55.0, "mfi": 10.0, "biodeg": 15.0, "water": 0.7, "clarity": 90.0},
    "PHA": {"tensile": 30.0, "mfi": 15.0, "biodeg": 80.0, "water": 0.9, "clarity": 20.0},
    "PBAT": {"tensile": 12.0, "mfi": 4.0, "biodeg": 60.0, "water": 1.2, "clarity": 55.0},
    "PBS": {"tensile": 35.0, "mfi": 22.0, "biodeg": 55.0, "water": 1.0, "clarity": 40.0},
    "TPS": {"tensile": 4.0, "mfi": 8.0, "biodeg": 90.0, "water": 25.0, "clarity": 30.0},
    "PCL": {"tensile": 16.0, "mfi": 8.0, "biodeg": 45.0, "water": 1.5, "clarity": 50.0},
}

# Ideal processing (nozzle) temperature per polymer (deg C).
PROC_TEMP_OPT: dict[str, float] = {
    "PLA": 200.0,
    "PHA": 175.0,
    "PBAT": 180.0,
    "PBS": 160.0,
    "TPS": 150.0,
    "PCL": 120.0,
}

# Pairwise immiscibility (0 miscible .. 1 strongly phase-separating). Phase
# separation penalises tensile strength and optical clarity. Symmetric; unlisted
# pairs default to 0.15.
_IMMISCIBLE: dict[frozenset[str], float] = {
    frozenset({"PLA", "TPS"}): 0.80,
    frozenset({"PLA", "PCL"}): 0.60,
    frozenset({"PLA", "PBAT"}): 0.50,
    frozenset({"PLA", "PHA"}): 0.40,
    frozenset({"PBS", "TPS"}): 0.60,
    frozenset({"PHA", "TPS"}): 0.55,
    frozenset({"PBAT", "TPS"}): 0.35,
    frozenset({"PLA", "PBS"}): 0.25,
}


def immiscibility(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _IMMISCIBLE.get(frozenset({a, b}), 0.15)


@dataclass
class Formulation:
    """A single formulation: polymer fractions + additive fractions + processing."""

    polymer_frac: dict[str, float]  # fractions over POLYMERS, sum ~ (1 - total additive)
    additive_frac: dict[str, float]  # fractions over ADDITIVES
    process_temp_c: float
    process_time_min: float

    def as_row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for p in POLYMERS:
            row[f"frac_{p}"] = float(self.polymer_frac.get(p, 0.0))
        for a in ADDITIVES:
            row[f"add_{a}"] = float(self.additive_frac.get(a, 0.0))
        row["process_temp_c"] = float(self.process_temp_c)
        row["process_time_min"] = float(self.process_time_min)
        return row


def _norm_polymer(frac: dict[str, float]) -> dict[str, float]:
    total = sum(frac.get(p, 0.0) for p in POLYMERS)
    if total <= 0:
        return {p: 0.0 for p in POLYMERS}
    return {p: frac.get(p, 0.0) / total for p in POLYMERS}


def blend_optimum_temp(pfrac: dict[str, float]) -> float:
    w = _norm_polymer(pfrac)
    return sum(w[p] * PROC_TEMP_OPT[p] for p in POLYMERS)


def _weighted_anchor(w: dict[str, float], key: str) -> float:
    return sum(w[p] * ANCHORS[p][key] for p in POLYMERS)


def _mean_immiscibility(w: dict[str, float]) -> float:
    """Composition-weighted mean pairwise immiscibility of the polymer matrix."""
    active = [(p, f) for p, f in w.items() if f > 1e-6]
    if len(active) < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            (pa, fa), (pb, fb) = active[i], active[j]
            weight = fa * fb
            num += weight * immiscibility(pa, pb)
            den += weight
    return num / den if den else 0.0


def forward_true(
    form: Formulation, *, after_supplier_shift: bool = False, feedstock_quality: float = 1.0
) -> dict[str, float]:
    """Deterministic 'true' property means for a formulation (pre-measurement-noise).

    This is the physical ground truth the ML forward model must learn to
    approximate from data alone.

    Args:
        form: The formulation to evaluate.
        after_supplier_shift: Apply the mid-2025 PBS supplier-purity regime change.
        feedstock_quality: Seasonal bio-feedstock purity multiplier (~1.0), from
            :func:`biopoly.timeseries.seasonal_feedstock_quality`. Purer feedstock
            yields a stronger, slightly clearer polymer.
    """
    w = _norm_polymer(form.polymer_frac)
    add = form.additive_frac
    p_plast = add.get("plasticizer", 0.0)
    p_nucl = add.get("nucleating", 0.0)
    p_comp = add.get("compatibilizer", 0.0)
    p_fibre = add.get("fibre", 0.0)
    p_cext = add.get("chain_extender", 0.0)

    # Supplier-purity shift (mid-2025): PBS batches degrade -> weaker, runnier. Sized
    # so the regime change is a clear drift signal that stands out above the seasonal
    # feedstock-quality baseline, not a borderline one.
    pbs_tensile_mult = 0.70 if after_supplier_shift else 1.0
    pbs_mfi_mult = 1.55 if after_supplier_shift else 1.0

    # --- base weighted anchors ---
    tensile = _weighted_anchor(w, "tensile")
    tensile -= w["PBS"] * ANCHORS["PBS"]["tensile"] * (1 - pbs_tensile_mult)
    mfi = _weighted_anchor(w, "mfi")
    mfi += w["PBS"] * ANCHORS["PBS"]["mfi"] * (pbs_mfi_mult - 1)
    biodeg = _weighted_anchor(w, "biodeg")
    water = _weighted_anchor(w, "water")
    clarity = _weighted_anchor(w, "clarity")

    # --- immiscibility penalty (reduced by compatibilizer) ---
    immis = _mean_immiscibility(w) * max(0.0, 1.0 - 6.0 * p_comp)
    tensile *= 1.0 - 0.45 * immis
    clarity *= 1.0 - 0.7 * immis

    # --- processing-temperature effect ---
    opt = blend_optimum_temp(form.polymer_frac)
    temp_gap = (form.process_temp_c - opt) / 30.0
    tensile *= float(np.exp(-0.5 * temp_gap**2))  # bell curve around optimum
    mfi *= float(np.exp(0.35 * (form.process_temp_c - opt) / 30.0))  # hotter -> runnier
    # long residence at high temp degrades chains slightly (raises biodeg, lowers tensile)
    thermal_dose = max(0.0, (form.process_temp_c - opt)) * form.process_time_min / 6000.0
    tensile *= 1.0 - 0.15 * thermal_dose
    biodeg += 6.0 * thermal_dose

    # --- additive effects ---
    tensile *= 1.0 - 1.8 * p_plast  # plasticiser softens
    tensile *= 1.0 + 1.2 * p_fibre - 1.5 * p_fibre**2  # fibre reinforces then embrittles
    tensile *= 1.0 + 2.0 * p_cext  # chain extender rebuilds MW
    tensile *= 1.0 + 0.8 * p_nucl  # nucleation -> crystallinity -> stiffer

    mfi *= 1.0 + 6.0 * p_plast  # plasticiser -> flows more
    mfi *= 1.0 - 1.5 * p_fibre  # fibre -> flows less
    mfi *= 1.0 - 8.0 * p_cext  # chain extender -> higher MW -> lower MFI
    mfi *= 1.0 - 2.0 * p_nucl

    biodeg += 15.0 * p_plast + 25.0 * p_fibre
    water += 8.0 * p_plast + 20.0 * p_fibre
    clarity *= 1.0 - 1.0 * p_plast
    clarity *= 1.0 - 3.0 * p_fibre  # fillers scatter light
    clarity *= 1.0 - 8.0 * p_nucl  # crystallinity -> haze
    clarity *= 1.0 + 2.0 * p_comp  # compatibiliser -> finer morphology -> clearer

    # --- feedstock quality (seasonal bio-feedstock purity; see biopoly.timeseries) ---
    # Purer feedstock -> higher molecular weight -> a stronger, slightly clearer
    # polymer. Gentle and monotone, so it is real structure the forward model can
    # learn from the feedstock_quality covariate rather than noise.
    tensile *= feedstock_quality
    clarity *= 1.0 + 0.3 * (feedstock_quality - 1.0)

    # --- clamp to physical ranges ---
    return {
        "tensile_strength_mpa": float(np.clip(tensile, 0.5, 80.0)),
        "melt_flow_index_g10min": float(np.clip(mfi, 0.5, 120.0)),
        "biodegradation_60d_pct": float(np.clip(biodeg, 0.0, 100.0)),
        "water_absorption_pct": float(np.clip(water, 0.0, 60.0)),
        "optical_clarity_pct": float(np.clip(clarity, 0.0, 95.0)),
    }
