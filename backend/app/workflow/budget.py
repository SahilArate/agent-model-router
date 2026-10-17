"""Per-workflow spending and call limits. This is what stops loops and runaway escalation."""

from dataclasses import dataclass


@dataclass
class Budget:
    """Checked before every model call. A single call can overshoot by its own cost."""

    max_cost_usd: float
    max_calls: int = 20
    spent_usd: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        if self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        if self.max_calls < 1:
            raise ValueError("max_calls must be at least 1")

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.max_cost_usd or self.calls >= self.max_calls

    def charge_call(self, cost_usd: float) -> None:
        """Record one model call and what it cost."""
        self.calls += 1
        self.spent_usd += cost_usd

    def charge(self, cost_usd: float) -> None:
        """Record extra spend that is not a counted call, such as verification."""
        self.spent_usd += cost_usd