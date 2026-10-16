"""Shared domain types used across the router, agents and registry."""

from enum import StrEnum


class Tier(StrEnum):
    """Model capability tiers, ordered from cheapest to most capable."""

    SMALL = "small"
    MID = "mid"
    FRONTIER = "frontier"

    @property
    def rank(self) -> int:
        return list(Tier).index(self)

    def escalate(self) -> "Tier | None":
        """Return the next tier up, or None if already at the top."""
        tiers = list(Tier)
        next_rank = self.rank + 1
        return tiers[next_rank] if next_rank < len(tiers) else None


class RiskLevel(StrEnum):
    """How costly a wrong answer is. Decides the routing process, not just the model."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def cheap_first_allowed(self) -> bool:
        """Only low-risk calls may start on a cheaper tier and escalate on failure."""
        return self is RiskLevel.LOW


class AgentRole(StrEnum):
    """The roles in our multi-agent workflow."""

    PLANNER = "planner"
    WORKER = "worker"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


class TaskCategory(StrEnum):
    """What kind of capability a call needs. Capability is not the same as complexity."""

    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODING = "coding"
    TOOL_USE = "tool_use"