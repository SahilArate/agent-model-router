"""Synthesizer agent: combines the step results into the final answer."""

from collections.abc import Sequence

from app.agents.common import VerifierFactory
from app.core.types import AgentRole
from app.llm.types import ChatMessage
from app.verification.checks import non_empty
from app.workflow.spec import CallSpec, RunOptions

_SYSTEM_PROMPT = (
    "You are the synthesis agent. Combine the step results into one final answer to the task."
)
_RUBRIC = (
    "The answer must address the original task using the step results, "
    "with no contradictions or invented facts."
)


def synthesizer_spec(
    task: str,
    step_results: Sequence[str],
    options: RunOptions,
    verifiers: VerifierFactory,
) -> CallSpec:
    numbered = "\n".join(f"{i}. {result}" for i, result in enumerate(step_results, start=1))
    basic, judged = verifiers.build([non_empty()], _RUBRIC)
    return CallSpec(
        agent=AgentRole.SYNTHESIZER,
        category=options.category,
        risk=options.risk,
        messages=(
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Task:\n{task}\n\nStep results:\n{numbered}"),
        ),
        pipeline=basic,
        judged_pipeline=judged,
        contains_sensitive_data=options.contains_sensitive_data,
        allowed_providers=options.allowed_providers,
    )
