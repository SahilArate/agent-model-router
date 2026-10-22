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


# What one judge call is assumed to cost when the router weighs cheap-first against going direct:
# the task, the criteria and the answer in, a short JSON verdict out.
# Reasoning models spend part of this budget thinking before they write the verdict.
JUDGE_MAX_TOKENS = 600
JUDGE_ESTIMATE_INPUT_TOKENS = 300
JUDGE_ESTIMATE_OUTPUT_TOKENS = 40


def estimated_judge_cost(judge: ModelProfile) -> float:
    return judge.cost(JUDGE_ESTIMATE_INPUT_TOKENS, JUDGE_ESTIMATE_OUTPUT_TOKENS)


def pick_judge(
    candidates: Sequence[ModelProfile],
    worker: ModelProfile,
    preferred_id: str | None = None,
) -> tuple[ModelProfile, bool]:
    """Choose a judge.

    Returns (judge, independent). A configured judge (preferred_id) wins when it is enabled and
    is not the worker. Otherwise the cheapest model from a different provider is chosen, since a
    different provider gives a better chance of independent errors; a same-provider judge is
    allowed but flagged as weaker. The worker itself is never its own judge.
    """
    others = [m for m in candidates if m.enabled and m.id != worker.id]
    if not others:
        raise ValueError("no judge available: need an enabled model other than the worker")

    if preferred_id is not None:
        configured = next((m for m in others if m.id == preferred_id), None)
        if configured is not None:
            return configured, configured.provider != worker.provider

    def price(model: ModelProfile) -> tuple[float, str]:
        return (model.input_price_per_mtok + model.output_price_per_mtok, model.id)

    different_provider = [m for m in others if m.provider != worker.provider]
    if different_provider:
        return min(different_provider, key=price), True
    return min(others, key=price), False


class JudgeCheck:
    name = "judge"

    def __init__(
        self,
        client: LLMClient,
        candidates: Sequence[ModelProfile],
        rubric: str,
        preferred_id: str | None = None,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        self._max_tokens = max_tokens
        self._client = client
        self._candidates = candidates
        self._rubric = rubric
        self._preferred_id = preferred_id

    async def run(self, ctx: VerificationContext) -> CheckResult:
        try:
            judge, independent = pick_judge(self._candidates, ctx.worker_model, self._preferred_id)
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
            result = await self._client.complete(judge, messages, max_tokens=self._max_tokens)
        except LLMError as exc:
            return CheckResult(
                self.name,
                CheckStatus.ERROR,
                f"judge {judge.id} failed: {exc}",
                judge_model=judge.id,
                independent=independent,
            )

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
                judge_model=judge.id,
                independent=independent,
            )

        return CheckResult(
            self.name,
            CheckStatus.PASS if passed else CheckStatus.FAIL,
            f"judge {judge.id}: {reason}{note}",
            cost_usd=result.cost_usd,
            judge_model=judge.id,
            independent=independent,
        )
