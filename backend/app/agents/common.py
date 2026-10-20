"""Pieces shared by all agents."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.client import LLMClient
from app.registry.models import ModelProfile
from app.verification.judge import JudgeCheck
from app.verification.pipeline import VerificationPipeline
from app.verification.types import Check


@dataclass(frozen=True)
class VerifierFactory:
    """Builds the two verification pipelines every call carries.

    The basic pipeline runs deterministic checks only. The judged pipeline adds an LLM judge.
    The executor picks between them using the router's requires_independent_verifier flag, so
    the decision to pay for a judge is made once, in the router.
    """

    client: LLMClient
    judge_candidates: tuple[ModelProfile, ...]
    judge_model_id: str | None = None

    def build(
        self, checks: Sequence[Check], rubric: str
    ) -> tuple[VerificationPipeline, VerificationPipeline]:
        basic = VerificationPipeline(list(checks))
        judge = JudgeCheck(self.client, self.judge_candidates, rubric, self.judge_model_id)
        return basic, VerificationPipeline([*checks, judge])
