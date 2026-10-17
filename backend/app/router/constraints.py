"""Stage A of routing: hard constraints.

A model that breaks a hard constraint is removed before any cost or quality reasoning.
Optimisation never gets to override these rules.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from app.registry.models import ModelProfile
from app.router.request import RoutingRequest


class NoEligibleModelError(Exception):
    """Every candidate model violates at least one hard constraint."""


@dataclass(frozen=True)
class FilterResult:
    """Models that passed, plus the reasons every other model was rejected."""

    eligible: tuple[ModelProfile, ...]
    rejected: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def require_any(self) -> tuple[ModelProfile, ...]:
        """Return the eligible models, or raise with every rejection reason."""
        if self.eligible:
            return self.eligible
        details = "; ".join(f"{mid}: {', '.join(why)}" for mid, why in self.rejected.items())
        raise NoEligibleModelError(f"no model satisfies the hard constraints ({details})")


def violations(model: ModelProfile, request: RoutingRequest) -> list[str]:
    """Return every hard constraint the model breaks for this request (empty means eligible)."""
    reasons: list[str] = []

    if not model.enabled:
        reasons.append("model is disabled")

    if request.allowed_providers is not None and model.provider not in request.allowed_providers:
        reasons.append(f"provider '{model.provider}' is not allowed")

    if request.contains_sensitive_data and not model.approved_for_sensitive_data:
        reasons.append("not approved for sensitive data")

    needed_context = request.input_tokens + request.max_output_tokens
    if needed_context > model.context_window:
        reasons.append(
            f"context too small ({needed_context} tokens needed, window is {model.context_window})"
        )

    if request.needs_tools and not model.supports_tools:
        reasons.append("does not support tool calling")

    if request.needs_structured_output and not model.supports_structured_output:
        reasons.append("does not support structured output")

    if request.max_cost_usd is not None:
        # Worst case: no cache hits and the full output allowance is used.
        worst_case = model.cost(request.input_tokens, request.max_output_tokens)
        if worst_case > request.max_cost_usd:
            reasons.append(
                f"worst-case cost ${worst_case:.6f} exceeds budget ${request.max_cost_usd:.6f}"
            )

    return reasons


def filter_models(models: Iterable[ModelProfile], request: RoutingRequest) -> FilterResult:
    """Split candidates into eligible models and rejections with reasons. Order is preserved."""
    eligible: list[ModelProfile] = []
    rejected: dict[str, tuple[str, ...]] = {}

    for model in models:
        reasons = violations(model, request)
        if reasons:
            rejected[model.id] = tuple(reasons)
        else:
            eligible.append(model)

    return FilterResult(eligible=tuple(eligible), rejected=rejected)