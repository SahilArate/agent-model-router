"""What to do with a verification report: accept, escalate or abstain."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.registry.models import ModelProfile
from app.verification.types import VerificationReport


class Action(StrEnum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"  # hand over to a human instead of guessing


@dataclass(frozen=True)
class Outcome:
    action: Action
    reason: str
    next_model: ModelProfile | None = None
    remaining_fallbacks: tuple[ModelProfile, ...] = ()
    failure_evidence: str = ""


def decide(report: VerificationReport, fallbacks: Sequence[ModelProfile]) -> Outcome:
    """Pass means accept. A failed output escalates if a stronger model remains.

    If verification itself broke, we abstain rather than escalate: the output is unverified,
    not known to be bad, so a more expensive model would not help.
    """
    if report.passed:
        return Outcome(Action.ACCEPT, "all checks passed")

    evidence = report.failure_evidence

    if report.has_error:
        return Outcome(
            Action.ABSTAIN, f"could not verify the output ({evidence})", failure_evidence=evidence
        )

    if fallbacks:
        return Outcome(
            Action.ESCALATE,
            f"verification failed ({evidence}); escalating to {fallbacks[0].id}",
            next_model=fallbacks[0],
            remaining_fallbacks=tuple(fallbacks[1:]),
            failure_evidence=evidence,
        )

    return Outcome(
        Action.ABSTAIN,
        f"verification failed ({evidence}) and no stronger model remains",
        failure_evidence=evidence,
    )