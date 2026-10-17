import pytest

from app.core.types import AgentRole, RiskLevel, TaskCategory, Tier
from app.registry.capability import CapabilityRegistry
from app.registry.models import ModelProfile, Provider
from app.router.constraints import NoEligibleModelError
from app.router.decision import RoutingMode
from app.router.engine import ModelRouter
from app.router.policy import RoutingPolicy
from app.router.request import RoutingRequest

AGENT, CATEGORY = AgentRole.PLANNER, TaskCategory.REASONING


def make_model(model_id, tier, input_price, output_price, **overrides) -> ModelProfile:
    fields = {
        "id": model_id,
        "provider": Provider.MOCK,
        "litellm_model": f"mock/{model_id}",
        "tier": tier,
        "input_price_per_mtok": input_price,
        "output_price_per_mtok": output_price,
        "context_window": 100_000,
        "supports_tools": True,
        "supports_structured_output": True,
        "approved_for_sensitive_data": True,
    }
    return ModelProfile(**{**fields, **overrides})


def default_models(**small_overrides):
    return [
        make_model("small", Tier.SMALL, 0.1, 0.5, **small_overrides),
        make_model("mid", Tier.MID, 2.0, 10.0),
        make_model("frontier", Tier.FRONTIER, 10.0, 50.0),
    ]


def make_request(**overrides) -> RoutingRequest:
    fields = {
        "agent": AGENT,
        "category": CATEGORY,
        "input_tokens": 1_000,
        "max_output_tokens": 1_000,
    }
    return RoutingRequest(**{**fields, **overrides})


def teach(registry, model_id, successes, failures):
    for _ in range(successes):
        registry.record(model_id, AGENT, CATEGORY, success=True)
    for _ in range(failures):
        registry.record(model_id, AGENT, CATEGORY, success=False)


def router(registry, models=None, policy=None):
    return ModelRouter(models or default_models(), registry, policy)


def test_picks_cheapest_proven_model_and_lists_escalation_path():
    registry = CapabilityRegistry()
    for model_id in ("small", "mid", "frontier"):
        teach(registry, model_id, 95, 5)

    decision = router(registry).route(make_request())

    assert decision.model.id == "small"
    assert decision.mode is RoutingMode.PROVEN
    assert [m.id for m in decision.fallbacks] == ["mid", "frontier"]
    assert "cheapest model whose lower bound" in decision.reason


def test_skips_cheap_model_that_has_not_proven_itself():
    registry = CapabilityRegistry()
    teach(registry, "small", 60, 40)
    teach(registry, "mid", 95, 5)
    teach(registry, "frontier", 95, 5)

    decision = router(registry).route(make_request())

    assert decision.model.id == "mid"
    assert decision.mode is RoutingMode.PROVEN
    assert [m.id for m in decision.fallbacks] == ["frontier"]


def test_thin_evidence_never_qualifies_even_if_perfect():
    registry = CapabilityRegistry()
    teach(registry, "small", 5, 0)  # lower bound 0.61, above the 0.5 floor below, but n < 10
    policy = RoutingPolicy(
        quality_floors={RiskLevel.LOW: 0.5, RiskLevel.MEDIUM: 0.9, RiskLevel.HIGH: 0.95}
    )

    decision = router(registry, policy=policy).route(make_request())
    small = next(c for c in decision.candidates if c.model.id == "small")

    assert small.qualified is False
    assert decision.mode is RoutingMode.CHEAP_FIRST


def test_cold_start_low_risk_goes_cheap_first_with_escalation_path():
    decision = router(CapabilityRegistry()).route(make_request())

    assert decision.model.id == "small"
    assert decision.mode is RoutingMode.CHEAP_FIRST
    assert [m.id for m in decision.fallbacks] == ["mid", "frontier"]
    assert decision.requires_independent_verifier is False


def test_uneconomical_cascade_upgrades_directly():
    models = [
        make_model("small", Tier.SMALL, 1.9, 9.5),  # nearly as expensive as mid
        make_model("mid", Tier.MID, 2.0, 10.0),
        make_model("frontier", Tier.FRONTIER, 10.0, 50.0),
    ]
    registry = CapabilityRegistry()
    teach(registry, "small", 1, 9)  # fails often, so escalation is very likely

    decision = router(registry, models).route(make_request())

    assert decision.model.id == "mid"
    assert decision.mode is RoutingMode.DIRECT_UPGRADE
    assert [m.id for m in decision.fallbacks] == ["frontier"]
    assert "skipping" in decision.reason


def test_medium_risk_without_proof_uses_strongest_and_independent_verifier():
    decision = router(CapabilityRegistry()).route(make_request(risk=RiskLevel.MEDIUM))

    assert decision.model.id == "frontier"
    assert decision.mode is RoutingMode.STRONGEST
    assert decision.fallbacks == ()
    assert decision.requires_independent_verifier is True


def test_high_risk_needs_much_stronger_evidence():
    registry = CapabilityRegistry()
    teach(registry, "small", 95, 5)  # fine for low risk, not for the 0.95 floor
    assert router(registry).route(make_request(risk=RiskLevel.HIGH)).model.id == "frontier"

    teach(registry, "small", 400, 0)  # now overwhelming evidence
    decision = router(registry).route(make_request(risk=RiskLevel.HIGH))

    assert decision.model.id == "small"
    assert decision.mode is RoutingMode.PROVEN


def test_hard_constraints_beat_cost():
    registry = CapabilityRegistry()
    teach(registry, "small", 500, 0)
    teach(registry, "mid", 500, 0)
    models = default_models(approved_for_sensitive_data=False)

    decision = router(registry, models).route(make_request(contains_sensitive_data=True))

    assert decision.model.id == "mid"
    assert "small" in decision.rejected
    assert "small" not in [c.model.id for c in decision.candidates]


def test_no_eligible_model_raises():
    with pytest.raises(NoEligibleModelError):
        router(CapabilityRegistry()).route(make_request(input_tokens=500_000))


def test_max_escalations_caps_the_fallback_path():
    policy = RoutingPolicy(max_escalations=1)

    decision = router(CapabilityRegistry(), policy=policy).route(make_request())

    assert [m.id for m in decision.fallbacks] == ["mid"]


def test_equal_cost_models_tie_break_on_higher_lower_bound():
    models = [
        make_model("a", Tier.SMALL, 1.0, 1.0),
        make_model("b", Tier.SMALL, 1.0, 1.0),
    ]
    registry = CapabilityRegistry()
    teach(registry, "a", 90, 10)
    teach(registry, "b", 99, 1)

    assert router(registry, models).route(make_request()).model.id == "b"


def test_candidates_expose_estimates_for_every_eligible_model():
    registry = CapabilityRegistry()
    teach(registry, "mid", 20, 0)

    decision = router(registry).route(make_request())
    by_id = {c.model.id: c for c in decision.candidates}

    assert set(by_id) == {"small", "mid", "frontier"}
    assert by_id["mid"].estimate.sample_size == 20
    assert by_id["small"].estimate.sample_size == 0
    assert by_id["small"].expected_cost_usd < by_id["mid"].expected_cost_usd