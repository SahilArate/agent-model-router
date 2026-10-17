import json

import pytest
from routing_helpers import PLAN_JSON, ScriptedLLM, client_for

from app.agents.common import VerifierFactory
from app.core.types import AgentRole, RiskLevel
from app.registry.capability import CapabilityRegistry
from app.router.engine import ModelRouter
from app.workflow.budget import Budget
from app.workflow.executor import RoutedExecutor
from app.workflow.runner import MultiAgentWorkflow, compress
from app.workflow.spec import RunOptions

TASK = "Summarise the key risks in the attached contract."


def build_workflow(models, script, *, max_steps=4):
    client = client_for(script)
    router = ModelRouter(models, CapabilityRegistry())
    verifiers = VerifierFactory(client=client, judge_candidates=tuple(models))
    return MultiAgentWorkflow(RoutedExecutor(router, client), verifiers, max_steps=max_steps)


def agents_in(result):
    return [t.agent for t in result.traces]


async def test_happy_path_runs_all_agents_and_accounts_for_cost(tier_models, script):
    budget = Budget(max_cost_usd=100.0)

    result = await build_workflow(tier_models, script).run(TASK, RunOptions(), budget)

    assert result.completed and result.final_answer == "final answer"
    assert agents_in(result) == [
        AgentRole.PLANNER,
        AgentRole.WORKER,
        AgentRole.WORKER,
        AgentRole.WORKER,
        AgentRole.SYNTHESIZER,
    ]
    assert result.cost_usd > 0
    assert budget.spent_usd == pytest.approx(result.cost_usd)
    assert result.abstained_at is None


async def test_planner_failure_escalates_and_workflow_still_completes(tier_models):
    script = ScriptedLLM({("planner", "small"): "not json at all"})

    result = await build_workflow(tier_models, script).run(
        TASK, RunOptions(), Budget(max_cost_usd=100.0)
    )

    plan_trace = result.traces[0]
    assert result.completed
    assert [a.model_id for a in plan_trace.attempts] == ["small", "mid"]
    assert plan_trace.escalations == 1


async def test_worker_abstention_stops_the_workflow_before_synthesis(tier_models):
    script = ScriptedLLM({("worker", m.id): "" for m in tier_models})

    result = await build_workflow(tier_models, script).run(
        TASK, RunOptions(), Budget(max_cost_usd=100.0)
    )

    assert not result.completed and result.final_answer is None
    assert result.abstained_at is AgentRole.WORKER
    assert AgentRole.SYNTHESIZER not in agents_in(result)


async def test_planner_abstention_means_no_work_is_done(tier_models):
    script = ScriptedLLM({("planner", m.id): "nope" for m in tier_models})

    result = await build_workflow(tier_models, script).run(
        TASK, RunOptions(), Budget(max_cost_usd=100.0)
    )

    assert result.abstained_at is AgentRole.PLANNER
    assert agents_in(result) == [AgentRole.PLANNER]


async def test_plan_is_truncated_to_max_steps(tier_models):
    five_steps = json.dumps({"steps": ["a", "b", "c", "d", "e"]})
    script = ScriptedLLM({("planner", "small"): five_steps})

    result = await build_workflow(tier_models, script, max_steps=2).run(
        TASK, RunOptions(), Budget(max_cost_usd=100.0)
    )

    assert agents_in(result).count(AgentRole.WORKER) == 2


async def test_workflow_budget_stops_runaway_calls(tier_models, script):
    budget = Budget(max_cost_usd=100.0, max_calls=2)  # planner + first worker only

    result = await build_workflow(tier_models, script).run(TASK, RunOptions(), budget)

    assert not result.completed
    assert result.abstained_at is AgentRole.WORKER
    assert budget.calls == 2


async def test_above_low_risk_an_independent_judge_is_paid_for(tier_models, script):
    options = RunOptions(risk=RiskLevel.MEDIUM)

    result = await build_workflow(tier_models, script).run(
        TASK, options, Budget(max_cost_usd=100.0)
    )

    assert result.completed
    assert script.calls_for("judge"), "medium risk must run the judge"
    assert all(t.attempts[-1].verification_cost_usd > 0 for t in result.traces)


async def test_low_risk_skips_the_judge(tier_models, script):
    await build_workflow(tier_models, script).run(TASK, RunOptions(), Budget(max_cost_usd=100.0))

    assert script.calls_for("judge") == []


async def test_later_workers_see_compressed_earlier_results(tier_models):
    script = ScriptedLLM({("worker", "small"): "x" * 2_000})

    await build_workflow(tier_models, script).run(TASK, RunOptions(), Budget(max_cost_usd=100.0))

    second_worker_prompt = script.calls_for("worker")[1][2][-1].content
    assert "x" * 400 not in second_worker_prompt
    assert "..." in second_worker_prompt


def test_compress_shortens_and_normalises_whitespace():
    assert compress("a  b\n c") == "a b c"
    assert len(compress("y" * 1_000, 100)) == 100
    assert compress("y" * 1_000, 100).endswith("...")


def test_budget_validation_and_exhaustion():
    with pytest.raises(ValueError):
        Budget(max_cost_usd=0)
    with pytest.raises(ValueError):
        Budget(max_cost_usd=1, max_calls=0)

    budget = Budget(max_cost_usd=1.0, max_calls=5)
    assert not budget.exhausted
    budget.charge_call(1.0)
    assert budget.exhausted


def test_plan_fixture_is_valid_json():
    assert len(json.loads(PLAN_JSON)["steps"]) == 3