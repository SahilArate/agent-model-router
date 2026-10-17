import pytest
from pydantic import ValidationError

from app.core.types import AgentRole, TaskCategory, Tier
from app.registry.models import ModelProfile, Provider
from app.router.constraints import NoEligibleModelError, filter_models, violations
from app.router.request import RoutingRequest


def make_model(**overrides) -> ModelProfile:
    fields = {
        "id": "m1",
        "provider": Provider.MOCK,
        "litellm_model": "mock/m1",
        "tier": Tier.SMALL,
        "input_price_per_mtok": 1.0,
        "output_price_per_mtok": 4.0,
        "context_window": 10_000,
        "supports_tools": True,
        "supports_structured_output": True,
        "approved_for_sensitive_data": True,
    }
    return ModelProfile(**{**fields, **overrides})


def make_request(**overrides) -> RoutingRequest:
    fields = {
        "agent": AgentRole.WORKER,
        "category": TaskCategory.EXTRACTION,
        "input_tokens": 1_000,
        "max_output_tokens": 1_000,
    }
    return RoutingRequest(**{**fields, **overrides})


def test_plain_request_passes():
    assert violations(make_model(), make_request()) == []


def test_context_overflow_is_rejected():
    request = make_request(input_tokens=9_500, max_output_tokens=1_000)

    reasons = violations(make_model(context_window=10_000), request)

    assert len(reasons) == 1 and "context" in reasons[0]


def test_tool_and_structured_output_requirements():
    model = make_model(supports_tools=False, supports_structured_output=False)
    request = make_request(needs_tools=True, needs_structured_output=True)

    reasons = violations(model, request)

    assert any("tool" in r for r in reasons)
    assert any("structured" in r for r in reasons)


def test_sensitive_data_requires_approved_model():
    request = make_request(contains_sensitive_data=True)

    assert violations(make_model(approved_for_sensitive_data=False), request)
    assert violations(make_model(approved_for_sensitive_data=True), request) == []


def test_provider_allowlist():
    request = make_request(allowed_providers=frozenset({Provider.ANTHROPIC}))

    assert violations(make_model(provider=Provider.OPENAI), request)
    assert violations(make_model(provider=Provider.ANTHROPIC), request) == []


def test_budget_uses_worst_case_cost():
    model = make_model()  # 1000 in + 1000 out = $0.005 worst case

    assert violations(model, make_request(max_cost_usd=0.006)) == []
    assert any("budget" in r for r in violations(model, make_request(max_cost_usd=0.004)))


def test_disabled_model_is_rejected():
    assert violations(make_model(enabled=False), make_request())


def test_all_violations_are_reported_together():
    model = make_model(supports_tools=False, approved_for_sensitive_data=False)
    request = make_request(needs_tools=True, contains_sensitive_data=True)

    assert len(violations(model, request)) == 2


def test_filter_splits_eligible_from_rejected_and_keeps_order():
    models = [
        make_model(id="a"),
        make_model(id="b", supports_tools=False),
        make_model(id="c"),
    ]

    result = filter_models(models, make_request(needs_tools=True))

    assert [m.id for m in result.eligible] == ["a", "c"]
    assert list(result.rejected) == ["b"]


def test_require_any_raises_with_reasons_when_nothing_is_eligible():
    models = [make_model(id="a", approved_for_sensitive_data=False)]
    result = filter_models(models, make_request(contains_sensitive_data=True))

    with pytest.raises(NoEligibleModelError, match="a: not approved for sensitive data"):
        result.require_any()


def test_request_rejects_negative_tokens():
    with pytest.raises(ValidationError):
        make_request(input_tokens=-1)