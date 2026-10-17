"""Description of one routed, verified LLM call."""

from dataclasses import dataclass

from app.core.types import AgentRole, RiskLevel, TaskCategory
from app.llm.types import ChatMessage
from app.registry.models import Provider
from app.verification.pipeline import VerificationPipeline


@dataclass(frozen=True)
class RunOptions:
    """Workflow-level settings every agent call inherits."""

    category: TaskCategory = TaskCategory.REASONING
    risk: RiskLevel = RiskLevel.LOW
    contains_sensitive_data: bool = False
    allowed_providers: frozenset[Provider] | None = None


@dataclass(frozen=True)
class CallSpec:
    agent: AgentRole
    category: TaskCategory
    risk: RiskLevel
    messages: tuple[ChatMessage, ...]
    pipeline: VerificationPipeline
    max_output_tokens: int = 1024
    needs_structured_output: bool = False
    contains_sensitive_data: bool = False
    allowed_providers: frozenset[Provider] | None = None

    @property
    def task_prompt(self) -> str:
        """What a judge is shown as the task: the final user message."""
        return self.messages[-1].content