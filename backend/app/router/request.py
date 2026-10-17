"""What an agent asks the router for on each LLM call."""

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import AgentRole, RiskLevel, TaskCategory
from app.registry.models import Provider


class RoutingRequest(BaseModel):
    """Everything the router needs to pick a model for one call."""

    model_config = ConfigDict(frozen=True)

    agent: AgentRole
    category: TaskCategory
    risk: RiskLevel = RiskLevel.LOW

    input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(default=1024, gt=0)

    needs_tools: bool = False
    needs_structured_output: bool = False
    contains_sensitive_data: bool = False

    max_cost_usd: float | None = Field(default=None, ge=0)
    allowed_providers: frozenset[Provider] | None = None