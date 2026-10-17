"""LLM-as-judge. Useful for subjective quality, but never the only check.

Judges fail closed: if the judge cannot run or answers unparseably, the output counts as
unverified (ERROR), not as passed.
"""

import json
from collections.abc import Sequence

from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.types import ChatMessage
from app.registry.models import ModelProfile
from app.verification.checks import extract_json_text
from app.verification.types import CheckResult, CheckStatus, VerificationContext

_SYSTEM_PROMPT = (
    "You are a strict evaluator. Judge the answer against the criteria. "
    'Reply with JSON only: {"pass": true or false, "reason": "<one sentence>"}.'
)


def pick_judge(
    candidates: Sequence[ModelProfile], worker: ModelProfile
) -> tuple[ModelProfile, bool]:
    """Choose a judge, preferring a different provider than the worker.

    Returns (judge, independent). A different provider gives a better chance of independent
    errors; a same-provider judge is allowed but flagged as weaker. The worker itself is
    never its own judge.
    """
    others = [m for m in candidates if m.enabled and m.id != worker.id]
    if not others:
        raise ValueError("no judge available: need an enabled model other than the worker")

    def price(model: ModelProfile) -> tuple[float, str]:
        return (model.input_price_per_mtok + model.output_price_per_mtok, model.id)

    different_provider = [m for m in others if m.provider != worker.provider]
    if different_provider:
        return min(different_provider, key=price), True
    return min(others, key=price), False


class JudgeCheck:
    name = "judge"

    def __init__(
        self, client: LLMClient, candidates: Sequence[ModelProfile], rubric: str
    ) -> None:
        self._client = client
        self._candidates = candidates
        self._rubric = rubric

    async def run(self, ctx: VerificationContext) -> CheckResult:
        try:
            judge, independent = pick_judge(self._candidates, ctx.worker_model)
        except ValueError as exc:
            return CheckResult(self.name, CheckStatus.ERROR, str(exc))

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Task:\n{ctx.task_prompt}\n\nCriteria:\n{self._rubric}\n\n"
                    f"Answer to evaluate:\n{ctx.output}"
                ),
            ),
        ]
        try:
            result = await self._client.complete(judge, messages, max_tokens=200)
        except LLMError as exc:
            return CheckResult(self.name, CheckStatus.ERROR, f"judge {judge.id} failed: {exc}")

        note = "" if independent else " (same provider as worker: weaker independence)"
        try:
            verdict = json.loads(extract_json_text(result.text))
            passed = bool(verdict["pass"])
            reason = str(verdict.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            return CheckResult(
                self.name,
                CheckStatus.ERROR,
                f"judge {judge.id} gave an unparseable verdict{note}",
                cost_usd=result.cost_usd,
            )

        return CheckResult(
            self.name,
            CheckStatus.PASS if passed else CheckStatus.FAIL,
            f"judge {judge.id}: {reason}{note}",
            cost_usd=result.cost_usd,
        )