"""API request/response models for the inverse-design endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from biopoly import TARGETS


class DesignRequest(BaseModel):
    """A /design request: a target property spec plus search settings."""

    target: dict[str, float] = Field(
        ..., description=f"Desired property values; keys a subset of {TARGETS}"
    )
    method: Literal["bayesopt", "baseline"] = "bayesopt"
    top_k: int = Field(3, ge=1, le=20)
    budget: int = Field(
        300, ge=50, le=5000, description="trials (bayesopt) or candidates (baseline)"
    )
    weights: dict[str, float] | None = None

    def validated_target(self) -> dict[str, float]:
        bad = set(self.target) - set(TARGETS)
        if bad:
            raise ValueError(f"unknown target key(s): {sorted(bad)}; allowed {TARGETS}")
        if not self.target:
            raise ValueError("target must specify at least one property")
        return self.target


class Candidate(BaseModel):
    """One ranked inverse-design result: score, formulation and predicted properties."""

    score: float
    formulation: dict
    predicted: dict[str, float]


class DesignResponse(BaseModel):
    """The /design response: the method used, echoed target, and ranked candidates."""

    method: str
    target: dict[str, float]
    candidates: list[Candidate]
