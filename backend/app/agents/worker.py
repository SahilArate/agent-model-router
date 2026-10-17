"""Worker agent: carries out one step of the plan."""

from app.agents.common import VerifierFactory
from app.core.types import AgentRole
from app.llm.types import ChatMessage
from app.verification.checks import non_empty
from app.workflow.spec import CallSpec, RunOptions

_SYSTEM_PROMPT = (
    "You are the worker agent. Carry out ONLY the step you are given, using earlier results "
    "if they help. Be concise."
)
_RUBRIC = "The result must correctly and completely carry out the given step."


def worker_spec(
    task: str,
    step: str,
    earlier_results: str,
    options: RunOptions,
    verifiers: VerifierFactory,
) -> CallSpec:
    user_prompt = (
        f"Overall task:\n{task}\n\n"
        f"Results so far:\n{earlier_results or '(none)'}\n\n"
        f"Your step:\n{step}"
    )
    return CallSpec(
        agent=AgentRole.WORKER,
        category=options.category,
        risk=options.risk,
        messages=(
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ),
        pipeline=verifiers.pipeline([non_empty()], options.risk, _RUBRIC),
        contains_sensitive_data=options.contains_sensitive_data,
        allowed_providers=options.allowed_providers,
    )