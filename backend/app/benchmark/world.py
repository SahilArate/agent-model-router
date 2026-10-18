"""A simulated LLM world: models whose skill depends on task category and difficulty.

This is an assumption set, not a measurement of real models. It exists so the routing and
verification machinery can be evaluated reproducibly and offline. Swapping `SimulatedWorld`
for a real backend (same `Responder` signature) runs the identical benchmark on real models.

A model answers a task correctly with probability SKILL[model][category][difficulty]. When it
is wrong, the answer is either malformed (cheap checks catch it) or well-formed but wrong
(only a judge could catch it). The split differs by model. All randomness is a hash of
(seed, model, task, conversation length), so runs are exactly reproducible, an escalated retry
gets a fresh roll, and two strategies calling the same model on the same task see the same draw.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping

from app.benchmark.tasks import Difficulty, Task
from app.core.types import TaskCategory
from app.llm.types import ChatMessage
from app.registry.models import ModelProfile

_Skill = tuple[float, float, float]  # easy, medium, hard

DEFAULT_SKILL: Mapping[str, Mapping[TaskCategory, _Skill]] = {
    "mock-small": {
        TaskCategory.EXTRACTION: (0.98, 0.92, 0.70),
        TaskCategory.SUMMARIZATION: (0.96, 0.85, 0.55),
        TaskCategory.REASONING: (0.85, 0.55, 0.20),
        TaskCategory.CODING: (0.80, 0.45, 0.15),
        TaskCategory.TOOL_USE: (0.95, 0.80, 0.35),
    },
    "mock-mid": {
        TaskCategory.EXTRACTION: (0.99, 0.97, 0.90),
        TaskCategory.SUMMARIZATION: (0.99, 0.95, 0.85),
        TaskCategory.REASONING: (0.97, 0.90, 0.70),
        TaskCategory.CODING: (0.97, 0.88, 0.65),
        TaskCategory.TOOL_USE: (0.99, 0.95, 0.80),
    },
    "mock-frontier": {
        TaskCategory.EXTRACTION: (0.995, 0.99, 0.97),
        TaskCategory.SUMMARIZATION: (0.995, 0.99, 0.95),
        TaskCategory.REASONING: (0.99, 0.97, 0.92),
        TaskCategory.CODING: (0.99, 0.96, 0.90),
        TaskCategory.TOOL_USE: (0.995, 0.99, 0.95),
    },
}

# Share of wrong answers that are malformed rather than plausible-but-wrong.
DEFAULT_MALFORMED_SHARE: Mapping[str, float] = {
    "mock-small": 0.5,
    "mock-mid": 0.35,
    "mock-frontier": 0.25,
}

# (time to first token in seconds, output tokens per second). Illustrative.
LATENCY: Mapping[str, tuple[float, float]] = {
    "mock-small": (0.3, 150.0),
    "mock-mid": (0.6, 80.0),
    "mock-frontier": (1.2, 40.0),
}

_MALFORMED_TEXT = "I believe the answer is probably what you would expect, but I am not certain."
_DIFFICULTY_INDEX = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}


def uniform(*parts: object) -> float:
    """A deterministic number in [0, 1) derived from the parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class SimulatedWorld:
    """Use as the responder of a MockBackend; also answers 'which task is this prompt?'."""

    def __init__(
        self,
        tasks: Iterable[Task],
        *,
        seed: int = 0,
        skill: Mapping[str, Mapping[TaskCategory, _Skill]] = DEFAULT_SKILL,
        malformed_share: Mapping[str, float] = DEFAULT_MALFORMED_SHARE,
    ) -> None:
        self._by_prompt = {t.prompt: t for t in tasks}
        self._seed = seed
        self._skill = skill
        self._malformed_share = malformed_share

    @property
    def seed(self) -> int:
        return self._seed

    def task_for(self, prompt: str) -> Task:
        return self._by_prompt[prompt]

    def p_correct(self, model_id: str, task: Task) -> float:
        return self._skill[model_id][task.category][_DIFFICULTY_INDEX[task.difficulty]]

    def __call__(self, model: ModelProfile, messages: list[ChatMessage]) -> str:
        task = self._by_prompt[messages[0].content]
        roll = uniform(self._seed, model.id, task.id, len(messages))
        if roll < self.p_correct(model.id, task):
            return task.expected
        kind_roll = uniform(self._seed, "kind", model.id, task.id, len(messages))
        if kind_roll < self._malformed_share[model.id] or not task.wrong_answers:
            return _MALFORMED_TEXT
        pick = int(uniform(self._seed, "pick", model.id, task.id, len(messages)) * 1e6)
        return task.wrong_answers[pick % len(task.wrong_answers)]


def describe_skill(skill: Mapping[str, Mapping[TaskCategory, _Skill]] = DEFAULT_SKILL) -> str:
    return json.dumps(
        {m: {c.value: list(v) for c, v in cats.items()} for m, cats in skill.items()}, indent=2
    )
