"""Shared types for output verification."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.registry.models import ModelProfile


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"  # the output is bad
    ERROR = "error"  # the check itself could not run, so the output is unverified


@dataclass(frozen=True)
class VerificationContext:
    """What a check gets to look at."""

    task_prompt: str
    output: str
    worker_model: ModelProfile


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    cost_usd: float = 0.0
    judge_model: str | None = None  # set by the judge check: who gave this verdict
    independent: bool | None = None  # judge from a different provider than the worker


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.status is CheckStatus.PASS for r in self.results)

    @property
    def has_error(self) -> bool:
        return any(r.status is CheckStatus.ERROR for r in self.results)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def failure_evidence(self) -> str:
        """Why verification did not pass, suitable for telling a stronger model what went wrong."""
        return "; ".join(
            f"{r.name}: {r.detail}" for r in self.results if r.status is not CheckStatus.PASS
        )


class Check(Protocol):
    name: str

    async def run(self, ctx: VerificationContext) -> CheckResult: ...
