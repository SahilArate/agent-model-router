"""The decision engine: stages A to C of routing for a single LLM call.

A  hard constraints remove ineligible models (see constraints.py)
B  among the rest, pick the cheapest model whose measured lower bound clears the floor
C  build an escalation path for verification failures
"""

import math
from collections.abc import Iterable

from app.core.types import RiskLevel
from app.registry.capability import CapabilityRegistry
from app.registry.models import ModelProfile
from app.router.cascade import cascade_beats_direct
from app.router.constraints import filter_models
from app.router.decision import Candidate, RoutingDecision, RoutingMode
from app.router.policy import RoutingPolicy
from app.router.request import RoutingRequest


class ModelRouter:
    def __init__(
        self,
        models: Iterable[ModelProfile],
        registry: CapabilityRegistry,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self._models = tuple(models)
        self._registry = registry
        self._policy = policy or RoutingPolicy()

    def route(self, request: RoutingRequest) -> RoutingDecision:
        filtered = filter_models(self._models, request)
        eligible = filtered.require_any()

        floor = self._policy.floor_for(request.risk)
        candidates = tuple(self._candidate(m, request, floor) for m in eligible)

        primary, mode, reason = self._choose_primary(candidates, request, floor)
        path = self._escalation_path(primary, candidates)

        if mode is RoutingMode.CHEAP_FIRST and path:
            next_up = path[0]
            if not cascade_beats_direct(
                primary.expected_cost_usd,
                next_up.expected_cost_usd,
                self._policy.verify_cost_usd,
                escalation_prob=1.0 - primary.estimate.mean,
            ):
                reason = (
                    f"{primary.model.id} is cheapest but cheap-first would cost more than "
                    f"going straight to {next_up.model.id} "
                    f"(escalation chance {1.0 - primary.estimate.mean:.0%}); skipping it"
                )
                primary, path, mode = next_up, path[1:], RoutingMode.DIRECT_UPGRADE

        return RoutingDecision(
            model=primary.model,
            mode=mode,
            quality_floor=floor,
            fallbacks=tuple(c.model for c in path[: self._policy.max_escalations]),
            candidates=candidates,
            rejected=filtered.rejected,
            requires_independent_verifier=request.risk is not RiskLevel.LOW,
            reason=reason,
        )

    def _candidate(self, model: ModelProfile, request: RoutingRequest, floor: float) -> Candidate:
        expected_output = max(
            1, math.ceil(request.max_output_tokens * self._policy.expected_output_fraction)
        )
        estimate = self._registry.estimate(model.id, request.agent, request.category)
        return Candidate(
            model=model,
            expected_cost_usd=model.cost(request.input_tokens, expected_output),
            estimate=estimate,
            qualified=estimate.sufficient_evidence and estimate.lower_bound >= floor,
        )

    @staticmethod
    def _choose_primary(
        candidates: tuple[Candidate, ...], request: RoutingRequest, floor: float
    ) -> tuple[Candidate, RoutingMode, str]:
        qualified = [c for c in candidates if c.qualified]
        if qualified:
            best = min(
                qualified,
                key=lambda c: (c.expected_cost_usd, -c.estimate.lower_bound, c.model.id),
            )
            reason = (
                f"{best.model.id} is the cheapest model whose lower bound "
                f"{best.estimate.lower_bound:.2f} clears the {request.risk} floor {floor:.2f} "
                f"(n={best.estimate.sample_size})"
            )
            return best, RoutingMode.PROVEN, reason

        if request.risk.cheap_first_allowed:
            cheapest = min(
                candidates,
                key=lambda c: (c.expected_cost_usd, c.model.tier.rank, c.model.id),
            )
            reason = (
                f"no model has proven the {request.risk} floor {floor:.2f} yet; starting with "
                f"cheapest eligible {cheapest.model.id} and relying on verification"
            )
            return cheapest, RoutingMode.CHEAP_FIRST, reason

        strongest = max(
            candidates,
            key=lambda c: (c.model.tier.rank, -c.expected_cost_usd, c.model.id),
        )
        reason = (
            f"no model has proven the {request.risk} floor {floor:.2f} and cheap-first is not "
            f"allowed at this risk; using strongest eligible {strongest.model.id}"
        )
        return strongest, RoutingMode.STRONGEST, reason

    @staticmethod
    def _escalation_path(primary: Candidate, candidates: tuple[Candidate, ...]) -> list[Candidate]:
        """One model per higher tier, in tier order, cheapest within each tier."""
        cheapest_per_tier: dict[int, Candidate] = {}
        for c in candidates:
            rank = c.model.tier.rank
            if rank <= primary.model.tier.rank:
                continue
            current = cheapest_per_tier.get(rank)
            if current is None or (c.expected_cost_usd, c.model.id) < (
                current.expected_cost_usd,
                current.model.id,
            ):
                cheapest_per_tier[rank] = c
        return [cheapest_per_tier[rank] for rank in sorted(cheapest_per_tier)]