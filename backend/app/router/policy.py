"""Tunable routing policy: quality floors per risk level and cost assumptions."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.core.types import RiskLevel


def _default_floors() -> dict[RiskLevel, float]:
    return {RiskLevel.LOW: 0.80, RiskLevel.MEDIUM: 0.90, RiskLevel.HIGH: 0.95}


@dataclass(frozen=True)
class RoutingPolicy:
    """quality_floors: the credible lower bound a model must clear, per risk level.
    expected_output_fraction: share of max_output_tokens assumed to be used when costing.
    verify_cost_usd: assumed cost of one verification (0 for deterministic checks).
    max_escalations: how many stronger models may be tried after the first.
    """

    quality_floors: Mapping[RiskLevel, float] = field(default_factory=_default_floors)
    expected_output_fraction: float = 0.5
    verify_cost_usd: float = 0.0
    max_escalations: int = 2

    def __post_init__(self) -> None:
        missing = set(RiskLevel) - set(self.quality_floors)
        if missing:
            raise ValueError(f"quality_floors missing risk levels: {sorted(missing)}")
        if any(not 0.0 < floor <= 1.0 for floor in self.quality_floors.values()):
            raise ValueError("quality floors must be in (0, 1]")
        if not 0.0 < self.expected_output_fraction <= 1.0:
            raise ValueError("expected_output_fraction must be in (0, 1]")
        if self.verify_cost_usd < 0:
            raise ValueError("verify_cost_usd must be non-negative")
        if self.max_escalations < 0:
            raise ValueError("max_escalations must be non-negative")

    def floor_for(self, risk: RiskLevel) -> float:
        return self.quality_floors[risk]