"""A deterministic simulated LLM for demos and tests.

Each model has a flaw rate: the chance that its answer is defective (planner: not valid JSON,
worker and synthesizer: empty). Whether a given call is flawed is decided by hashing the model
and prompt, so the same input always gives the same output, while an escalated retry with a
different prompt or model gets a fresh roll.
"""

import hashlib
import json
from collections.abc import Mapping

from app.llm.types import ChatMessage
from app.registry.models import ModelProfile

DEMO_FLAW_RATES: Mapping[str, float] = {
    "mock-small": 0.35,
    "mock-mid": 0.10,
    "mock-frontier": 0.0,
}

_AGENT_MARKERS = (
    ("evaluator", "judge"),
    ("planning agent", "planner"),
    ("worker agent", "worker"),
    ("synthesis agent", "synthesizer"),
)
_PLAN_STEPS = [
    "Gather the relevant facts",
    "Analyse the facts against the task",
    "Draft the conclusion",
]
_STEP_MARKER = "Your step:\n"
_TASK_MARKER = "Task:\n"


class SimulatedLLM:
    """Use as the responder of a MockBackend."""

    def __init__(
        self,
        flaw_rates: Mapping[str, float] | None = None,
        *,
        seed: int = 0,
        default_flaw_rate: float = 0.0,
    ) -> None:
        self._flaw_rates = dict(DEMO_FLAW_RATES if flaw_rates is None else flaw_rates)
        self._seed = seed
        self._default = default_flaw_rate

    def __call__(self, model: ModelProfile, messages: list[ChatMessage]) -> str:
        agent = _agent_of(messages)
        if agent == "judge":
            return '{"pass": true, "reason": "looks fine"}'

        flawed = self._roll(model.id, messages) < self._flaw_rates.get(model.id, self._default)
        user_text = messages[0 if len(messages) < 2 else 1].content

        if agent == "planner":
            return "Here is my plan: look first, then decide." if flawed else _plan_json()
        if flawed:
            return ""
        if agent == "worker":
            step = user_text.split(_STEP_MARKER, 1)[-1].strip()
            return f"[{model.id}] completed step: {step}"
        if agent == "synthesizer":
            task = user_text.removeprefix(_TASK_MARKER).split("\n", 1)[0]
            return f"[{model.id}] Final answer for: {task[:80]}"
        return f"[{model.id}] ok"

    def _roll(self, model_id: str, messages: list[ChatMessage]) -> float:
        payload = f"{self._seed}|{model_id}|{len(messages)}|" + "|".join(
            m.content for m in messages
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64


def _agent_of(messages: list[ChatMessage]) -> str:
    system = messages[0].content if messages else ""
    for marker, name in _AGENT_MARKERS:
        if marker in system:
            return name
    return "unknown"


def _plan_json() -> str:
    return json.dumps({"steps": _PLAN_STEPS})