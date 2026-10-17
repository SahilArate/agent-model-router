"""Deterministic checks. They cost nothing and cannot share a model's blind spots."""

import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from app.verification.types import CheckResult, CheckStatus, VerificationContext

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json_text(text: str) -> str:
    """Strip markdown fences and surrounding prose from a model's JSON answer."""
    stripped = _FENCE.sub("", text.strip()).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class SchemaCheck:
    """The output must be a JSON object that validates against a pydantic model."""

    name = "schema"

    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    async def run(self, ctx: VerificationContext) -> CheckResult:
        try:
            self._schema.model_validate_json(extract_json_text(ctx.output))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "<root>"
            return CheckResult(self.name, CheckStatus.FAIL, f"{location}: {first['msg']}")
        return CheckResult(self.name, CheckStatus.PASS, "valid")


class RuleCheck:
    """A named yes/no rule over the output text."""

    def __init__(self, name: str, predicate: Callable[[str], bool], failure: str) -> None:
        self.name = name
        self._predicate = predicate
        self._failure = failure

    async def run(self, ctx: VerificationContext) -> CheckResult:
        if self._predicate(ctx.output):
            return CheckResult(self.name, CheckStatus.PASS)
        return CheckResult(self.name, CheckStatus.FAIL, self._failure)


def non_empty() -> RuleCheck:
    return RuleCheck("non_empty", lambda text: bool(text.strip()), "output is empty")


def contains_all(*phrases: str) -> RuleCheck:
    lowered = [p.lower() for p in phrases]

    def predicate(text: str) -> bool:
        haystack = text.lower()
        return all(p in haystack for p in lowered)

    return RuleCheck("contains_all", predicate, f"output must mention: {', '.join(phrases)}")


def matches_regex(pattern: str) -> RuleCheck:
    compiled = re.compile(pattern)
    return RuleCheck(
        "matches_regex",
        lambda text: compiled.search(text) is not None,
        f"output does not match /{pattern}/",
    )


def max_chars(limit: int) -> RuleCheck:
    return RuleCheck(
        "max_chars", lambda text: len(text) <= limit, f"output is longer than {limit} characters"
    )