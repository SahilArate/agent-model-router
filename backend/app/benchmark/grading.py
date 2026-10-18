"""Ground-truth grading and cheap format validation for benchmark answers."""

import json
import re

from app.benchmark.tasks import AnswerKind, Task

_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_WORD = re.compile(r"^[A-Za-z]+$")
_NUMBER_TOLERANCE = 0.005


def _loads(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_well_formed(kind: AnswerKind, output: str) -> bool:
    """Free, deterministic check: does the answer have the right shape at all?"""
    text = output.strip()
    if kind is AnswerKind.NUMBER:
        return bool(_NUMBER.match(text))
    if kind is AnswerKind.WORD:
        return bool(_WORD.match(text))
    parsed = _loads(text)
    return parsed is not None and "tool" in parsed and isinstance(parsed.get("args"), dict)


def is_correct(task: Task, output: str | None) -> bool:
    """Exact ground-truth comparison. This is the only source of truth in the benchmark."""
    if output is None or not is_well_formed(task.kind, output):
        return False
    text = output.strip()
    if task.kind is AnswerKind.NUMBER:
        return abs(float(text) - float(task.expected)) <= _NUMBER_TOLERANCE
    if task.kind is AnswerKind.WORD:
        return text.lower() == task.expected.lower()
    return _loads(text) == json.loads(task.expected)
