"""Measured capability profiles.

For every (model, agent role, task category) we track successes and failures and treat the
true success rate as Beta-distributed. Routing uses a conservative lower quantile of that
distribution, so thin evidence cannot win against solid evidence.

The estimates describe our own task distribution, not the model in general.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from app.core.types import AgentRole, TaskCategory
from app.registry.beta import beta_ppf


class CapabilityKey(NamedTuple):
    model_id: str
    agent: AgentRole
    category: TaskCategory


@dataclass(frozen=True)
class CapabilityEstimate:
    """What we currently believe about one model on one kind of task."""

    successes: int
    failures: int
    mean: float
    lower_bound: float
    sufficient_evidence: bool

    @property
    def sample_size(self) -> int:
        return self.successes + self.failures


class CapabilityRegistry:
    def __init__(
        self,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        lower_quantile: float = 0.05,
        min_samples: int = 10,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior parameters must be positive")
        if not 0.0 < lower_quantile < 0.5:
            raise ValueError("lower_quantile must be between 0 and 0.5")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._lower_quantile = lower_quantile
        self._min_samples = min_samples
        self._counts: dict[CapabilityKey, list[int]] = {}

    def record(
        self, model_id: str, agent: AgentRole, category: TaskCategory, *, success: bool
    ) -> None:
        counts = self._counts.setdefault(CapabilityKey(model_id, agent, category), [0, 0])
        counts[0 if success else 1] += 1

    def estimate(
        self, model_id: str, agent: AgentRole, category: TaskCategory
    ) -> CapabilityEstimate:
        successes, failures = self._counts.get(CapabilityKey(model_id, agent, category), (0, 0))
        alpha = self._prior_alpha + successes
        beta = self._prior_beta + failures
        return CapabilityEstimate(
            successes=successes,
            failures=failures,
            mean=alpha / (alpha + beta),
            lower_bound=beta_ppf(self._lower_quantile, alpha, beta),
            sufficient_evidence=successes + failures >= self._min_samples,
        )

    def save(self, path: Path) -> None:
        records = [
            {
                "model_id": key.model_id,
                "agent": key.agent.value,
                "category": key.category.value,
                "successes": counts[0],
                "failures": counts[1],
            }
            for key, counts in sorted(self._counts.items())
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Replace the current counts with those stored at path."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._counts = {
            CapabilityKey(
                r["model_id"], AgentRole(r["agent"]), TaskCategory(r["category"])
            ): [int(r["successes"]), int(r["failures"])]
            for r in raw["records"]
        }