"""Shared test helpers: a scriptable mock LLM and a model factory."""

import json

from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.registry.models import ModelProfile, Provider

PLAN_JSON = json.dumps({"steps": ["gather facts", "analyse facts", "draft conclusion"]})

_AGENT_MARKERS = {
    "evaluator": "judge",
    "planning agent": "planner",
    "worker agent": "worker",
    "synthesis agent": "synthesizer",
}


def agent_of(messages) -> str:
    system = messages[0].content
    for marker, name in _AGENT_MARKERS.items():
        if marker in system:
            return name
    return "unknown"


class ScriptedLLM:
    """Mock backend responder. Override replies per (agent, model id); an Exception is raised."""

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.calls: list[tuple[str, str, list]] = []

    def __call__(self, model, messages):
        agent = agent_of(messages)
        self.calls.append((agent, model.id, list(messages)))
        reply = self.overrides.get((agent, model.id))
        if isinstance(reply, Exception):
            raise reply
        if reply is not None:
            return reply
        defaults = {
            "judge": '{"pass": true, "reason": "fine"}',
            "planner": PLAN_JSON,
            "worker": f"result from {model.id}",
            "synthesizer": "final answer",
        }
        return defaults.get(agent, "ok")

    def calls_for(self, agent: str) -> list[tuple[str, str, list]]:
        return [c for c in self.calls if c[0] == agent]


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


def client_for(script: ScriptedLLM) -> LLMClient:
    return LLMClient(MockBackend(responder=script))