"""The multi-agent workflow: plan, execute each step, then synthesize.

Every agent call is routed, verified and traced by the RoutedExecutor. If any agent abstains,
the workflow stops there instead of building on an unverified result.
"""

from pydantic import ValidationError

from app.agents.common import VerifierFactory
from app.agents.planner import parse_plan, plan_spec
from app.agents.synthesizer import synthesizer_spec
from app.agents.worker import worker_spec
from app.core.types import AgentRole
from app.workflow.budget import Budget
from app.workflow.executor import RoutedExecutor
from app.workflow.spec import CallSpec, RunOptions
from app.workflow.trace import CallTrace, WorkflowResult

_CONTEXT_CHARS_PER_RESULT = 400


def compress(text: str, limit: int = _CONTEXT_CHARS_PER_RESULT) -> str:
    """Shorten an earlier result before passing it on, so context does not grow every hop."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class MultiAgentWorkflow:
    def __init__(
        self, executor: RoutedExecutor, verifiers: VerifierFactory, *, max_steps: int = 4
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._executor = executor
        self._verifiers = verifiers
        self._max_steps = max_steps

    async def run(self, task: str, options: RunOptions, budget: Budget) -> WorkflowResult:
        traces: list[CallTrace] = []

        plan_trace = await self._run(plan_spec(task, options, self._verifiers), budget, traces)
        if not plan_trace.accepted:
            return self._abstained(task, traces, AgentRole.PLANNER)

        try:
            steps = parse_plan(plan_trace.output or "")[: self._max_steps]
        except ValidationError:  # unreachable while the schema check guards the planner
            return self._abstained(task, traces, AgentRole.PLANNER)

        results: list[str] = []
        for step in steps:
            context = "\n".join(f"- {compress(r)}" for r in results)
            spec = worker_spec(task, step, context, options, self._verifiers)
            trace = await self._run(spec, budget, traces)
            if not trace.accepted:
                return self._abstained(task, traces, AgentRole.WORKER)
            results.append(trace.output or "")

        spec = synthesizer_spec(task, results, options, self._verifiers)
        final = await self._run(spec, budget, traces)
        if not final.accepted:
            return self._abstained(task, traces, AgentRole.SYNTHESIZER)

        return WorkflowResult(
            task=task,
            completed=True,
            final_answer=final.output,
            traces=tuple(traces),
        )

    async def _run(self, spec: CallSpec, budget: Budget, traces: list[CallTrace]) -> CallTrace:
        trace = await self._executor.run(spec, budget)
        traces.append(trace)
        return trace

    @staticmethod
    def _abstained(task: str, traces: list[CallTrace], at: AgentRole) -> WorkflowResult:
        return WorkflowResult(
            task=task,
            completed=False,
            final_answer=None,
            traces=tuple(traces),
            abstained_at=at,
        )