import pytest
from routing_helpers import ScriptedLLM, client_for, make_model

from app.core.types import AgentRole, RiskLevel, TaskCategory, Tier
from app.llm.types import ChatMessage
from app.registry.capability import CapabilityRegistry
from app.router.engine import ModelRouter
from app.verification.checks import non_empty
from app.verification.outcome import Action
from app.verification.pipeline import VerificationPipeline
from app.verification.types import CheckResult, CheckStatus
from app.workflow.budget import Budget
from app.workflow.executor import RoutedExecutor
from app.workflow.spec import CallSpec


def make_spec(pipeline=None, **overrides) -> CallSpec:
    fields = {
        "agent": AgentRole.WORKER,
        "category": TaskCategory.EXTRACTION,
        "risk": RiskLevel.LOW,
        "messages": (
            ChatMessage(role="system", content="You are the worker agent."),
            ChatMessage(role="user", content="do the thing"),
        ),
        "pipeline": pipeline or VerificationPipeline([non_empty()]),
    }
    return CallSpec(**{**fields, **overrides})


def executor_for(script, models):
    router = ModelRouter(models, CapabilityRegistry())
    return RoutedExecutor(router, client_for(script))


def big_budget() -> Budget:
    return Budget(max_cost_usd=100.0)


async def test_accepts_on_first_attempt_with_cost_tracked(tier_models, script):
    trace = await executor_for(script, tier_models).run(make_spec(), big_budget())

    assert trace.accepted and trace.output == "result from small"
    assert [a.model_id for a in trace.attempts] == ["small"]
    assert trace.escalations == 0
    assert trace.cost_usd == pytest.approx(trace.attempts[0].generation_cost_usd)
    assert trace.cost_usd > 0
    assert trace.decision is not None


async def test_failed_verification_escalates_with_evidence(tier_models):
    script = ScriptedLLM({("worker", "small"): "   "})

    trace = await executor_for(script, tier_models).run(make_spec(), big_budget())

    assert trace.accepted and trace.output == "result from mid"
    assert [a.model_id for a in trace.attempts] == ["small", "mid"]
    assert trace.escalations == 1
    retry_messages = script.calls_for("worker")[1][2]
    assert "failed verification" in retry_messages[-1].content
    assert "output is empty" in retry_messages[-1].content
    assert retry_messages[-2].role == "assistant"


async def test_abstains_when_every_model_fails_and_loop_is_bounded(tier_models):
    script = ScriptedLLM({("worker", m.id): "" for m in tier_models})

    trace = await executor_for(script, tier_models).run(make_spec(), big_budget())

    assert trace.final_action is Action.ABSTAIN and trace.output is None
    assert [a.model_id for a in trace.attempts] == ["small", "mid", "frontier"]


async def test_provider_failure_falls_back_to_the_next_model(tier_models):
    script = ScriptedLLM({("worker", "small"): RuntimeError("provider down")})

    trace = await executor_for(script, tier_models).run(make_spec(), big_budget())

    assert trace.accepted and trace.output == "result from mid"
    assert "provider down" in trace.attempts[0].error
    assert trace.attempts[0].action is None


async def test_provider_failure_with_no_fallback_abstains(tier_models):
    script = ScriptedLLM({("worker", m.id): RuntimeError("down") for m in tier_models})

    trace = await executor_for(script, tier_models).run(make_spec(), big_budget())

    assert trace.final_action is Action.ABSTAIN
    assert "no fallback remains" in trace.reason
    assert len(trace.attempts) == 3


async def test_budget_exhaustion_stops_escalation(tier_models):
    script = ScriptedLLM({("worker", "small"): ""})
    budget = Budget(max_cost_usd=100.0, max_calls=1)

    trace = await executor_for(script, tier_models).run(make_spec(), budget)

    assert trace.final_action is Action.ABSTAIN
    assert "budget" in trace.reason
    assert len(trace.attempts) == 1


async def test_no_eligible_model_abstains_without_calling_anything(script):
    models = [make_model("only", Tier.SMALL, 1, 1, approved_for_sensitive_data=False)]
    spec = make_spec(contains_sensitive_data=True)

    trace = await executor_for(script, models).run(spec, big_budget())

    assert trace.final_action is Action.ABSTAIN
    assert trace.attempts == () and trace.decision is None
    assert script.calls == []


async def test_unverifiable_output_abstains_instead_of_escalating(tier_models, script):
    class BrokenCheck:
        name = "broken"

        async def run(self, ctx):
            return CheckResult(self.name, CheckStatus.ERROR, "judge unavailable")

    spec = make_spec(pipeline=VerificationPipeline([BrokenCheck()]))

    trace = await executor_for(script, tier_models).run(spec, big_budget())

    assert trace.final_action is Action.ABSTAIN
    assert len(trace.attempts) == 1


async def test_budget_is_charged_for_generation_and_verification(tier_models, script):
    budget = big_budget()

    trace = await executor_for(script, tier_models).run(make_spec(), budget)

    assert budget.spent_usd == pytest.approx(trace.cost_usd)
    assert budget.calls == 1