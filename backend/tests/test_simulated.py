import json

import pytest
from routing_helpers import make_model

from app.agents.planner import parse_plan
from app.core.types import Tier
from app.llm.simulated import SimulatedLLM
from app.llm.types import ChatMessage

MODEL = make_model("mock-small", Tier.SMALL, 0.1, 0.5)


def planner_messages(task="do something"):
    return [
        ChatMessage(role="system", content="You are the planning agent. Break the task down."),
        ChatMessage(role="user", content=task),
    ]


def worker_messages(step="gather facts"):
    return [
        ChatMessage(role="system", content="You are the worker agent. Carry out one step."),
        ChatMessage(role="user", content=f"Overall task:\nT\n\nYour step:\n{step}"),
    ]


def test_same_input_always_gives_the_same_output():
    llm = SimulatedLLM({"mock-small": 0.5}, seed=3)

    assert llm(MODEL, planner_messages()) == llm(MODEL, planner_messages())


def test_zero_flaw_rate_is_never_flawed_and_plan_is_valid():
    llm = SimulatedLLM({"mock-small": 0.0})

    for i in range(50):
        steps = parse_plan(llm(MODEL, planner_messages(f"task {i}")))
        assert len(steps) == 3
        assert llm(MODEL, worker_messages(f"step {i}")).strip()


def test_full_flaw_rate_is_always_flawed():
    llm = SimulatedLLM({"mock-small": 1.0})

    with pytest.raises(json.JSONDecodeError):
        json.loads(llm(MODEL, planner_messages()))
    assert llm(MODEL, worker_messages()) == ""


def test_flaw_frequency_matches_the_configured_rate():
    llm = SimulatedLLM({"mock-small": 0.3}, seed=1)

    flawed = sum(llm(MODEL, worker_messages(f"step {i}")) == "" for i in range(600))

    assert 0.22 < flawed / 600 < 0.38


def test_different_seeds_give_different_outcomes():
    prompts = [worker_messages(f"step {i}") for i in range(100)]

    first = [SimulatedLLM({"mock-small": 0.5}, seed=1)(MODEL, p) for p in prompts]
    second = [SimulatedLLM({"mock-small": 0.5}, seed=2)(MODEL, p) for p in prompts]

    assert first != second


def test_judge_always_passes():
    llm = SimulatedLLM({"mock-small": 1.0})
    messages = [
        ChatMessage(role="system", content="You are a strict evaluator."),
        ChatMessage(role="user", content="anything"),
    ]

    assert json.loads(llm(MODEL, messages))["pass"] is True


def test_unknown_models_use_the_default_rate():
    llm = SimulatedLLM({}, default_flaw_rate=1.0)

    assert llm(MODEL, worker_messages()) == ""