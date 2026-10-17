"""Request and response shapes of the HTTP API. This is the contract the dashboard uses."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.api.services import Mode
from app.core.types import RiskLevel, TaskCategory

TaskText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class RunRequest(BaseModel):
    task: TaskText
    mode: Mode = Mode.MOCK
    risk: RiskLevel = RiskLevel.LOW
    category: TaskCategory = TaskCategory.REASONING
    contains_sensitive_data: bool = False
    max_cost_usd: float = Field(default=0.50, gt=0, le=10)
    max_calls: int = Field(default=20, ge=1, le=50)
    max_steps: int = Field(default=4, ge=1, le=6)


class EstimateOut(BaseModel):
    mean: float
    lower_bound: float
    sample_size: int
    sufficient_evidence: bool


class CandidateOut(BaseModel):
    model_id: str
    tier: str
    expected_cost_usd: float
    qualified: bool
    estimate: EstimateOut


class RoutingOut(BaseModel):
    model_id: str
    mode: str
    reason: str
    quality_floor: float
    fallbacks: list[str]
    requires_independent_verifier: bool
    candidates: list[CandidateOut]
    rejected: dict[str, list[str]]


class CheckOut(BaseModel):
    name: str
    status: str
    detail: str
    cost_usd: float


class AttemptOut(BaseModel):
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    generation_cost_usd: float
    verification_cost_usd: float
    action: str | None
    error: str
    checks: list[CheckOut]


class CallTraceOut(BaseModel):
    agent: str
    category: str
    risk: str
    final_action: str
    reason: str
    output: str | None
    cost_usd: float
    latency_s: float
    escalations: int
    routing: RoutingOut | None
    attempts: list[AttemptOut]


class BudgetOut(BaseModel):
    spent_usd: float
    calls: int
    max_cost_usd: float
    max_calls: int


class RunResponse(BaseModel):
    mode: Mode
    task: str
    completed: bool
    final_answer: str | None
    abstained_at: str | None
    cost_usd: float
    latency_s: float
    budget: BudgetOut
    traces: list[CallTraceOut]


class ModelOut(BaseModel):
    id: str
    provider: str
    tier: str
    enabled: bool
    input_price_per_mtok: float
    output_price_per_mtok: float
    context_window: int
    supports_tools: bool
    supports_structured_output: bool
    approved_for_sensitive_data: bool
    notes: str


class HealthOut(BaseModel):
    status: str
    version: str
    configured_providers: list[str]