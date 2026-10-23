"""Run the real router, verifier and judge on held-out tasks with known answers.

For every task it records which model answered, why the router chose it, whether the judge
sent it back, and whether each answer was actually right. That shows both why a cheap model
gets escalated and how good the judge really is, instead of assuming either.
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.api.serializers import escalation_steps
from app.benchmark.grading import is_correct, is_well_formed
from app.benchmark.stats import wilson_interval
from app.benchmark.tasks import Task
from app.core.types import AgentRole
from app.llm.types import ChatMessage
from app.profiling.answers import normalize_answer
from app.profiling.guarded import GuardedClient, RecordedCall
from app.profiling.runner import SYSTEM_PROMPT, RunRecord, prompt_hash
from app.registry.capability import CapabilityRegistry
from app.registry.models import ModelProfile
from app.router.engine import ModelRouter
from app.router.policy import RoutingPolicy
from app.verification.judge import JudgeCheck, estimated_judge_cost
from app.verification.outcome import Action
from app.verification.pipeline import VerificationPipeline
from app.verification.types import CheckResult, CheckStatus, VerificationContext
from app.workflow.budget import Budget
from app.workflow.executor import RoutedExecutor
from app.workflow.spec import CallSpec

RUBRIC = (
    "The answer must be exactly correct for the task and contain only the requested value, "
    "with no explanation."
)
MAX_OUTPUT_TOKENS = 600


class FormatCheck:
    """Free check: is the answer in the requested shape? Tolerates code fences only."""

    name = "format"

    def __init__(self, task: Task) -> None:
        self._kind = task.kind

    async def run(self, ctx: VerificationContext) -> CheckResult:
        if is_well_formed(self._kind, normalize_answer(ctx.output)):
            return CheckResult(self.name, CheckStatus.PASS, "well formed")
        return CheckResult(self.name, CheckStatus.FAIL, "answer is not in the requested format")


@dataclass
class AttemptResult:
    model_id: str
    answer: str | None
    correct: bool | None
    action: str | None
    error: str
    checks: list[dict]
    output_tokens: int = 0
    truncated: bool = False  # the reply used the whole token budget and may be cut off


@dataclass
class TaskResult:
    task_id: str
    prompt_sha: str
    category: str
    difficulty: str
    risk: str
    routing_mode: str
    routing_reason: str
    judged: bool
    attempts: list[AttemptResult]
    escalation_steps: list[dict]
    final_model: str | None
    accepted: bool
    correct: bool
    cost_usd: float
    latency_s: float
    notes: list[str] = field(default_factory=list)


def build_pipelines(
    task: Task, client: GuardedClient, judge: ModelProfile, workers: Sequence[ModelProfile]
) -> tuple[VerificationPipeline, VerificationPipeline]:
    basic = VerificationPipeline([FormatCheck(task)])
    judge_check = JudgeCheck(client, (judge, *workers), RUBRIC, preferred_id=judge.id)  # type: ignore[arg-type]
    return basic, VerificationPipeline([FormatCheck(task), judge_check])


def _attempts(task: Task, trace, calls: list[RecordedCall], max_tokens: int) -> list[AttemptResult]:
    results = []
    for index, attempt in enumerate(trace.attempts):
        call = calls[index] if index < len(calls) else None
        text = call.text if call else None
        tokens = call.output_tokens if call else 0
        results.append(
            AttemptResult(
                model_id=attempt.model_id,
                answer=None if text is None else normalize_answer(text)[:300],
                correct=None if text is None else is_correct(task, normalize_answer(text)),
                action=attempt.action.value if attempt.action else None,
                error=attempt.error,
                output_tokens=tokens,
                truncated=tokens >= max_tokens > 0,
                checks=[
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "detail": c.detail,
                        "judge_model": c.judge_model,
                        "independent": c.independent,
                    }
                    for c in attempt.checks
                ],
            )
        )
    return results


async def trace_task(
    task: Task,
    executor: RoutedExecutor,
    client: GuardedClient,
    judge: ModelProfile,
    workers: Sequence[ModelProfile],
    risk_override=None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> TaskResult:
    basic, judged = build_pipelines(task, client, judge, workers)
    spec = CallSpec(
        agent=AgentRole.WORKER,
        category=task.category,
        risk=risk_override or task.risk,
        messages=(
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=task.prompt),
        ),
        pipeline=basic,
        judged_pipeline=judged,
        max_output_tokens=max_output_tokens,
    )
    client.calls.clear()
    trace = await executor.run(spec, Budget(max_cost_usd=1.0, max_calls=8))
    calls = [c for c in client.calls if c.model_id != judge.id]
    final = normalize_answer(trace.output) if trace.output is not None else None
    return TaskResult(
        task_id=task.id,
        prompt_sha=prompt_hash(task),
        category=task.category.value,
        difficulty=task.difficulty.value,
        risk=spec.risk.value,
        routing_mode=trace.decision.mode.value if trace.decision else "none",
        routing_reason=trace.decision.reason if trace.decision else trace.reason,
        judged=bool(trace.decision and trace.decision.requires_independent_verifier),
        attempts=_attempts(task, trace, calls, max_output_tokens),
        escalation_steps=[s.model_dump() for s in escalation_steps(trace)],
        final_model=trace.attempts[-1].model_id if trace.attempts else None,
        accepted=trace.final_action is Action.ACCEPT,
        correct=final is not None and is_correct(task, final),
        cost_usd=trace.cost_usd,
        latency_s=trace.latency_s,
    )


class TraceLog:
    """Append-only JSON-lines record of finished tasks, so an interrupted run resumes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[TaskResult]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                raw["attempts"] = [AttemptResult(**a) for a in raw["attempts"]]
                rows.append(TaskResult(**raw))
        return rows

    def append(self, result: TaskResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")


async def run_trace(
    tasks: Sequence[Task],
    workers: Sequence[ModelProfile],
    judge: ModelProfile,
    client: GuardedClient,
    registry: CapabilityRegistry,
    log: TraceLog,
    *,
    risk=None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    stop_reason: Callable[[], str | None] = lambda: None,
    say: Callable[[str], None] = print,
) -> list[TaskResult]:
    policy = RoutingPolicy(verify_cost_usd=estimated_judge_cost(judge))
    executor = RoutedExecutor(ModelRouter(workers, registry, policy), client)  # type: ignore[arg-type]
    results = log.load()
    done = {(r.task_id, r.prompt_sha) for r in results}
    for task in tasks:
        if (task.id, prompt_hash(task)) in done:
            continue
        if stop_reason() is not None:
            say(f"stopping: {stop_reason()}")
            break
        result = await trace_task(task, executor, client, judge, workers, risk, max_output_tokens)
        log.append(result)
        results.append(result)
        done.add((task.id, result.prompt_sha))
        if len(results) % 5 == 0:
            say(f"  {len(results)} tasks done")
    wanted = {(t.id, prompt_hash(t)) for t in tasks}
    return [r for r in results if (r.task_id, r.prompt_sha) in wanted]


def _rate(hits: int, n: int) -> dict:
    lo, hi = wilson_interval(hits, n) if n else (0.0, 0.0)
    return {"hits": hits, "n": n, "rate": hits / n if n else None, "ci95": [lo, hi]}


def summarise_trace(results: Sequence[TaskResult]) -> dict:
    """Outcome, cost, escalation behaviour and measured judge quality."""
    n = len(results)
    correct = sum(r.correct for r in results)
    wrong_accepted = sum(r.accepted and not r.correct for r in results)
    total_cost = sum(r.cost_usd for r in results)

    caught = false_accept = false_reject = correct_passed = judge_errors = 0
    for r in results:
        for a in r.attempts:
            verdict = next((c for c in a.checks if c["name"] == "judge"), None)
            if verdict is None or a.correct is None:
                continue
            if verdict["status"] == "error":
                judge_errors += 1
            elif a.correct:
                correct_passed += verdict["status"] == "pass"
                false_reject += verdict["status"] == "fail"
            else:
                caught += verdict["status"] == "fail"
                false_accept += verdict["status"] == "pass"

    return {
        "tasks": n,
        "success": _rate(correct, n),
        "accepted": sum(r.accepted for r in results),
        "abstained": sum(not r.accepted for r in results),
        "wrong_but_accepted": wrong_accepted,
        "escalated_tasks": _rate(sum(bool(r.escalation_steps) for r in results), n),
        "routing_modes": dict(Counter(r.routing_mode for r in results)),
        "final_models": dict(Counter(r.final_model or "none" for r in results)),
        "truncated_attempts": sum(a.truncated for r in results for a in r.attempts),
        "total_cost_usd": total_cost,
        "cost_per_success_usd": total_cost / correct if correct else None,
        "judge": {
            "wrong_answers_judged": caught + false_accept,
            "detect_rate": _rate(caught, caught + false_accept),
            "correct_answers_judged": correct_passed + false_reject,
            "false_reject_rate": _rate(false_reject, correct_passed + false_reject),
            "unusable_verdicts": judge_errors,
        },
    }


def first_escalation(
    results: Sequence[TaskResult], *, judge_only: bool = True
) -> TaskResult | None:
    """A task where a cheaper model's answer was sent back. By default only judge rejections
    count (the Q3 exhibit); with judge_only=False any escalation qualifies."""
    return next(
        (
            r
            for r in results
            if any(s["check"] == "judge" or not judge_only for s in r.escalation_steps)
        ),
        None,
    )


def describe(result: TaskResult, task_prompt: str) -> str:
    lines = [
        f"Task {result.task_id} ({result.category}, {result.difficulty}, {result.risk} risk)",
        f"  prompt: {task_prompt.splitlines()[-1][:160]}",
        f"  router: {result.routing_mode}: {result.routing_reason}",
    ]
    for i, a in enumerate(result.attempts, 1):
        verdict = next((c for c in a.checks if c["name"] == "judge"), None)
        label = "correct" if a.correct else "WRONG" if a.correct is not None else "no answer"
        if a.truncated:
            label += ", reply cut off at the token limit"
        lines.append(f"  attempt {i}: {a.model_id} answered {a.answer!r} ({label}) -> {a.action}")
        if verdict:
            lines.append(
                f"      judge {verdict['judge_model']}: {verdict['status']}: {verdict['detail']}"
            )
    for s in result.escalation_steps:
        lines.append(
            f"  escalated {s['from_model']} -> {s['to_model']} because {s['check']} failed"
        )
    lines.append(
        f"  final: {result.final_model}, accepted={result.accepted}, "
        f"correct={result.correct}, cost ${result.cost_usd:.6f}"
    )
    return "\n".join(lines)


def baseline_costs(records: Sequence[RunRecord], categories: set[str]) -> dict[str, dict]:
    """What each model costs and achieves on its own, with no judge and no routing, from the
    profiling run. A fair yardstick for the routed system on the same task types."""
    grouped: dict[str, list[RunRecord]] = {}
    for r in records:
        if r.category in categories:
            grouped.setdefault(r.model_id, []).append(r)
    result = {}
    for model_id, rows in sorted(grouped.items()):
        wins = sum(r.correct for r in rows)
        total = sum(r.cost_usd for r in rows)
        result[model_id] = {
            "n": len(rows),
            "success": _rate(wins, len(rows)),
            "cost_per_success_usd": total / wins if wins else None,
        }
    return result
