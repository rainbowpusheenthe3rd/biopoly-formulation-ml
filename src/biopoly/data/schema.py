"""Canonical column definitions and Pydantic I/O models.

Kept in one place so the generator, feature pipeline, model and API all agree on
the schema. ``FEATURE_COLS`` is the exact model input contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from biopoly import TARGETS
from biopoly.data.chemistry import ADDITIVES, POLYMERS

POLYMER_FRAC_COLS = [f"frac_{p}" for p in POLYMERS]
ADDITIVE_COLS = [f"add_{a}" for a in ADDITIVES]
PROCESS_COLS = ["process_temp_c", "process_time_min"]
# Temporal context covariates (seasonal feedstock purity; see biopoly.timeseries).
CONTEXT_COLS = ["feedstock_quality"]
NUMERIC_FEATURES = POLYMER_FRAC_COLS + ADDITIVE_COLS + PROCESS_COLS + CONTEXT_COLS
CATEGORICAL_FEATURES = ["primary_polymer", "tensile_protocol"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# DSC-derived signal features (see biopoly.signals). An optional ablation feature
# group — added to the dataset only when the generator is asked for them, and NOT
# part of the default FEATURE_COLS.
SIGNAL_FEATURES = [
    "dsc_n_peaks",
    "dsc_dominant_temp_c",
    "dsc_total_area",
    "dsc_mean_width_c",
    "dsc_max_height",
]

TENSILE_PROTOCOLS = ["ISO527", "ASTMD638"]

__all__ = [
    "POLYMER_FRAC_COLS",
    "ADDITIVE_COLS",
    "PROCESS_COLS",
    "CONTEXT_COLS",
    "SIGNAL_FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FEATURE_COLS",
    "TENSILE_PROTOCOLS",
    "TARGETS",
    "FormulationInput",
    "PropertyPrediction",
    "PredictResponse",
]


class FormulationInput(BaseModel):
    """A validated formulation for the /predict endpoint.

    Polymer fractions + additive fractions should sum to ~1.0 (a small tolerance
    is allowed and the vector is renormalised downstream).
    """

    frac: dict[str, float] = Field(
        ..., description="Polymer -> fraction, keys in {PLA,PHA,PBAT,PBS,TPS,PCL}"
    )
    additives: dict[str, float] = Field(
        default_factory=dict,
        description="Additive -> fraction, keys in "
        "{plasticizer,nucleating,compatibilizer,fibre,chain_extender}",
    )
    process_temp_c: float = Field(..., ge=80.0, le=260.0)
    process_time_min: float = Field(..., ge=1.0, le=120.0)
    tensile_protocol: str = Field("ISO527")
    feedstock_quality: float = Field(
        1.0,
        ge=0.5,
        le=1.5,
        description="Seasonal feedstock-purity multiplier (~1.0); 1.0 = typical batch",
    )

    @model_validator(mode="after")
    def _check_keys(self) -> FormulationInput:
        bad_p = set(self.frac) - set(POLYMERS)
        if bad_p:
            raise ValueError(f"unknown polymer(s): {sorted(bad_p)}; allowed {POLYMERS}")
        bad_a = set(self.additives) - set(ADDITIVES)
        if bad_a:
            raise ValueError(f"unknown additive(s): {sorted(bad_a)}; allowed {ADDITIVES}")
        if sum(self.frac.values()) <= 0:
            raise ValueError("at least one polymer fraction must be > 0")
        if self.tensile_protocol not in TENSILE_PROTOCOLS:
            raise ValueError(f"tensile_protocol must be one of {TENSILE_PROTOCOLS}")
        return self


class PropertyPrediction(BaseModel):
    """A single property prediction with an uncertainty band."""

    value: float
    p10: float
    p90: float


class PredictResponse(BaseModel):
    """The /predict response: per-property predictions plus any warnings."""

    predictions: dict[str, PropertyPrediction]
    warnings: list[str] = Field(default_factory=list)
