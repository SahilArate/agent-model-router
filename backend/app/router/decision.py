"""The router's output: the chosen model and the evidence behind the choice."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.registry.capability import CapabilityEstimate
from app.registry.models import ModelProfile


class RoutingMode(StrEnum):
    PROVEN = "proven"  # a model's measured lower bound clears the quality floor
    CHEAP_FIRST = "cheap_first"  # nothing proven; low risk, so start cheap and verify
    DIRECT_UPGRADE = "direct_upgrade"  # cheap-first would cost more than skipping the cheap model
    STRONGEST = "strongest"  # nothing proven and risk too high for cheap-first


@dataclass(frozen=True)
class Candidate:
    """One eligible model as the router saw it, kept for explanation and the dashboard."""

    model: ModelProfile
    expected_cost_usd: float
    estimate: CapabilityEstimate
    qualified: bool


@dataclass(frozen=True)
class RoutingDecision:
    model: ModelProfile
    mode: RoutingMode
    quality_floor: float
    fallbacks: tuple[ModelProfile, ...]
    candidates: tuple[Candidate, ...]
    rejected: Mapping[str, tuple[str, ...]]
    requires_independent_verifier: bool
    reason: str