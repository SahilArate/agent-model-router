"""Everything that happened during a call or a workflow, kept for the dashboard and benchmark."""

from dataclasses import dataclass

from app.core.types import AgentRole, RiskLevel, TaskCategory
from app.router.decision import RoutingDecision
from app.verification.outcome import Action
from app.verification.types import CheckResult


@dataclass(frozen=True)
class AttemptTrace:
    """One model call plus its verification."""

    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    generation_cost_usd: float = 0.0
    verification_cost_usd: float = 0.0
    checks: tuple[CheckResult, ...] = ()
    action: Action | None = None  # None when the provider call itself failed
    error: str = ""

    @property
    def cost_usd(self) -> float:
        return self.generation_cost_usd + self.verification_cost_usd


@dataclass(frozen=True)
class CallTrace:
    agent: AgentRole
    category: TaskCategory
    risk: RiskLevel
    decision: RoutingDecision | None
    attempts: tuple[AttemptTrace, ...]
    output: str | None
    final_action: Action
    reason: str

    @property
    def accepted(self) -> bool:
        return self.final_action is Action.ACCEPT

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def latency_s(self) -> float:
        return sum(a.latency_s for a in self.attempts)

    @property
    def escalations(self) -> int:
        return max(0, len(self.attempts) - 1)


@dataclass(frozen=True)
class WorkflowResult:
    task: str
    completed: bool
    final_answer: str | None
    traces: tuple[CallTrace, ...]
    abstained_at: AgentRole | None = None

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.traces)

    @property
    def latency_s(self) -> float:
        return sum(t.latency_s for t in self.traces)