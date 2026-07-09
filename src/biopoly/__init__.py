"""biopoly — forward + inverse ML for compostable biopolymer formulation.

A synthetic-data MLOps demo for materials formulation:
formulation (polymers + ratios + additives + processing) -> 5 material properties,
plus inverse design (target spec -> formulation).
"""

__version__ = "0.1.0"

TARGETS = [
    "tensile_strength_mpa",
    "melt_flow_index_g10min",
    "biodegradation_60d_pct",
    "water_absorption_pct",
    "optical_clarity_pct",
]
