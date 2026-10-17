"""Pieces shared by all agents."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.types import RiskLevel
from app.llm.client import LLMClient
from app.registry.models import ModelProfile
from app.verification.judge import JudgeCheck
from app.verification.pipeline import VerificationPipeline
from app.verification.types import Check


@dataclass(frozen=True)
class VerifierFactory:
    """Builds verification pipelines. Deterministic checks always run; an independent
    judge is added above low risk, matching the router's requires_independent_verifier."""

    client: LLMClient
    judge_candidates: tuple[ModelProfile, ...]

    def pipeline(
        self, checks: Sequence[Check], risk: RiskLevel, rubric: str
    ) -> VerificationPipeline:
        ordered = list(checks)
        if risk is not RiskLevel.LOW:
            ordered.append(JudgeCheck(self.client, self.judge_candidates, rubric))
        return VerificationPipeline(ordered)