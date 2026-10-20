"""Turn a model's raw reply into the answer that is graded."""

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def normalize_answer(text: str) -> str:
    """Remove only presentation wrappers: reasoning tags, surrounding whitespace, code fences.

    Anything else (extra words, units, explanations) is left in place and so counts against the
    model, because the prompts ask for the bare answer and the router's own checks are strict.
    """
    cleaned = _THINK.sub("", text).strip()
    fenced = _FENCE.match(cleaned)
    return fenced.group(1).strip() if fenced else cleaned
