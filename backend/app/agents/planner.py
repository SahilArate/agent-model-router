"""Planner agent: breaks the task into a short ordered list of steps."""

from pydantic import BaseModel, Field

from app.agents.common import VerifierFactory
from app.core.types import AgentRole
from app.llm.types import ChatMessage
from app.verification.checks import SchemaCheck, extract_json_text, non_empty
from app.workflow.spec import CallSpec, RunOptions

_SYSTEM_PROMPT = (
    "You are the planning agent. Break the task into 2 to 4 short, concrete steps. "
    'Reply with JSON only: {"steps": ["...", "..."]}.'
)
_RUBRIC = "The plan must be a short, ordered list of concrete steps that fully covers the task."


class Plan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=8)


def parse_plan(text: str) -> list[str]:
    return Plan.model_validate_json(extract_json_text(text)).steps


def plan_spec(task: str, options: RunOptions, verifiers: VerifierFactory) -> CallSpec:
    return CallSpec(
        agent=AgentRole.PLANNER,
        category=options.category,
        risk=options.risk,
        messages=(
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=task),
        ),
        pipeline=verifiers.pipeline([non_empty(), SchemaCheck(Plan)], options.risk, _RUBRIC),
        needs_structured_output=True,
        contains_sensitive_data=options.contains_sensitive_data,
        allowed_providers=options.allowed_providers,
    )