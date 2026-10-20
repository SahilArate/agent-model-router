"""Measure real models on the benchmark tasks and turn the results into capability profiles.

Safe to run on a free tier: calls are throttled per provider, spend and call counts are capped,
transient API errors are retried and never counted as a model failure, and every call is
appended to a log so an interrupted run resumes where it stopped.
"""

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.benchmark.grading import is_correct
from app.benchmark.stats import wilson_interval
from app.benchmark.tasks import Task
from app.core.types import AgentRole, TaskCategory
from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.types import ChatMessage
from app.profiling.answers import normalize_answer
from app.profiling.limiter import Limits, RateLimiter, limits_for
from app.registry.capability import CapabilityRegistry
from app.registry.models import ModelProfile

Sleep = Callable[[float], Awaitable[None]]

SYSTEM_PROMPT = "Follow the instructions exactly. Reply with only the final answer, no explanation."
TYPICAL_OUTPUT_TOKENS = 100  # assumption used only for the "typical" cost estimate
RETRY_DELAYS_S = (5.0, 15.0, 45.0)
MAX_CONSECUTIVE_ERRORS = 5
# Errors that retrying cannot fix: a wrong key, or a model id the provider does not know.
_FATAL = re.compile(
    r"api key|api_key|authentication|unauthorized|permission denied|"
    r"\b(?:401|403|404)\b|not found|does not exist|unknown model|invalid model",
    re.IGNORECASE,
)
# Rate-limit wording always wins: those errors are worth waiting out.
_RATE_LIMITED = re.compile(r"rate.?limit|\b429\b|quota|too many requests", re.IGNORECASE)


def is_fatal(message: str) -> bool:
    return bool(_FATAL.search(message)) and not _RATE_LIMITED.search(message)


def prompt_hash(task: Task) -> str:
    return hashlib.sha256(task.prompt.encode()).hexdigest()[:12]


def estimated_input_tokens(task: Task) -> int:
    return (len(task.prompt) + len(SYSTEM_PROMPT)) // 3 + 10


@dataclass(frozen=True)
class RunRecord:
    model_id: str
    task_id: str
    prompt_sha: str
    category: str
    difficulty: str
    correct: bool
    output: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    truncated: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model_id, self.task_id, self.prompt_sha)


class RunLog:
    """Append-only JSON-lines log of finished calls."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(RunRecord(**json.loads(line)))
        return rows

    def append(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")


@dataclass(frozen=True)
class RunEstimate:
    calls: dict[str, int]
    worst_cost_usd: float
    typical_cost_usd: float
    minutes: float

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())


def estimate_run(
    models: Sequence[ModelProfile],
    tasks: Sequence[Task],
    *,
    max_tokens: int,
    done: set[tuple[str, str, str]],
    limits: dict[str, Limits] | None = None,
) -> RunEstimate:
    """Cost, call count and wall-clock time of what is still to run, before spending anything."""
    calls: dict[str, int] = {}
    worst = typical = 0.0
    provider_calls: dict[str, int] = defaultdict(int)
    provider_tokens: dict[str, int] = defaultdict(int)
    for model in models:
        todo = [t for t in tasks if (model.id, t.id, prompt_hash(t)) not in done]
        calls[model.id] = len(todo)
        for task in todo:
            tokens_in = estimated_input_tokens(task)
            worst += model.cost(tokens_in, max_tokens)
            typical += model.cost(tokens_in, TYPICAL_OUTPUT_TOKENS)
            provider_calls[model.provider.value] += 1
            provider_tokens[model.provider.value] += tokens_in + max_tokens
    minutes = 0.0
    for provider, n_calls in provider_calls.items():
        lim = (limits or {}).get(provider) or limits_for(provider)
        minutes = max(minutes, n_calls / lim.rpm, provider_tokens[provider] / lim.tpm)
    return RunEstimate(calls, worst, typical, minutes)


class Guard:
    """Hard caps on notional spend and on the number of calls, shared by all models."""

    def __init__(self, max_cost_usd: float, max_calls: int) -> None:
        self.max_cost_usd = max_cost_usd
        self.max_calls = max_calls
        self.spent_usd = 0.0
        self.calls = 0
        self.reason: str | None = None

    def begin_call(self, worst_case_usd: float) -> bool:
        if self.reason is not None:
            return False
        if self.calls >= self.max_calls:
            self.reason = f"call cap reached ({self.max_calls})"
        elif self.spent_usd + worst_case_usd > self.max_cost_usd:
            self.reason = f"cost cap reached (${self.spent_usd:.4f} of ${self.max_cost_usd:.2f})"
        else:
            self.calls += 1
        return self.reason is None

    def charge(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd


class FatalModelError(Exception):
    """The provider rejected the request in a way retrying cannot fix."""


@dataclass
class ModelStatus:
    model_id: str
    completed: int = 0
    errors: int = 0
    aborted: str | None = None


@dataclass(frozen=True)
class ProfileConfig:
    max_tokens: int = 600
    max_cost_usd: float = 0.50
    max_calls: int = 300
    retry_delays_s: tuple[float, ...] = RETRY_DELAYS_S


@dataclass
class ProfileRun:
    records: list[RunRecord]
    statuses: dict[str, ModelStatus]
    guard: Guard
    notes: list[str] = field(default_factory=list)


async def _call(
    model: ModelProfile,
    task: Task,
    client: LLMClient,
    limiter: RateLimiter,
    guard: Guard,
    config: ProfileConfig,
    sleep: Sleep,
):
    tokens_in = estimated_input_tokens(task)
    worst = model.cost(tokens_in, config.max_tokens)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=task.prompt),
    ]
    for attempt in range(len(config.retry_delays_s) + 1):
        if not guard.begin_call(worst):
            return None
        reservation = await limiter.acquire(tokens_in + config.max_tokens)
        try:
            result = await client.complete(
                model, messages, max_tokens=config.max_tokens, temperature=0.0
            )
        except LLMError as exc:
            if is_fatal(str(exc)):
                raise FatalModelError(str(exc)) from exc
            if attempt < len(config.retry_delays_s):
                await sleep(config.retry_delays_s[attempt])
                continue
            return None
        RateLimiter.settle(reservation, result.input_tokens + result.output_tokens)
        guard.charge(result.cost_usd)
        return result
    return None


async def _run_model(
    model: ModelProfile,
    tasks: Sequence[Task],
    client: LLMClient,
    limiter: RateLimiter,
    guard: Guard,
    log: RunLog,
    config: ProfileConfig,
    done: set[tuple[str, str, str]],
    status: ModelStatus,
    say: Callable[[str], None],
    sleep: Sleep,
) -> list[RunRecord]:
    fresh: list[RunRecord] = []
    consecutive_errors = 0
    for task in tasks:
        if (model.id, task.id, prompt_hash(task)) in done:
            continue
        try:
            result = await _call(model, task, client, limiter, guard, config, sleep)
        except FatalModelError as exc:
            status.aborted = f"provider rejected the request: {exc}"
            break
        if result is None:
            if guard.reason is not None:
                status.aborted = guard.reason
                break
            status.errors += 1
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status.aborted = f"{MAX_CONSECUTIVE_ERRORS} consecutive failed calls"
                break
            continue
        consecutive_errors = 0
        answer = normalize_answer(result.text)
        record = RunRecord(
            model_id=model.id,
            task_id=task.id,
            prompt_sha=prompt_hash(task),
            category=task.category.value,
            difficulty=task.difficulty.value,
            correct=is_correct(task, answer),
            output=answer[:300],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_s=round(result.latency_s, 3),
            cost_usd=result.cost_usd,
            truncated=result.output_tokens >= config.max_tokens,
        )
        log.append(record)
        fresh.append(record)
        status.completed += 1
        if status.completed % 10 == 0:
            say(f"  {model.id}: {status.completed} calls done")
    return fresh


async def profile_models(
    models: Sequence[ModelProfile],
    tasks: Sequence[Task],
    client: LLMClient,
    log: RunLog,
    config: ProfileConfig,
    *,
    limits: dict[str, Limits] | None = None,
    say: Callable[[str], None] = print,
    sleep: Sleep = asyncio.sleep,
    limiter_sleep: Sleep = asyncio.sleep,
) -> ProfileRun:
    """Run every model over the tasks (providers in parallel) and return all known records."""
    existing = log.load()
    done = {r.key for r in existing}
    guard = Guard(config.max_cost_usd, config.max_calls)
    limiters: dict[str, RateLimiter] = {}
    for model in models:
        provider = model.provider.value
        if provider not in limiters:
            limiters[provider] = RateLimiter(
                (limits or {}).get(provider) or limits_for(provider),
                sleep=limiter_sleep,
            )
    statuses = {m.id: ModelStatus(m.id) for m in models}

    batches = await asyncio.gather(
        *(
            _run_model(
                m,
                tasks,
                client,
                limiters[m.provider.value],
                guard,
                log,
                config,
                done,
                statuses[m.id],
                say,
                sleep,
            )
            for m in models
        )
    )
    records = [*existing, *(r for batch in batches for r in batch)]
    return ProfileRun(records=records, statuses=statuses, guard=guard)


def select_records(records: Sequence[RunRecord], tasks: Sequence[Task]) -> list[RunRecord]:
    """Records measured on exactly these tasks (runs on other tasks are left out)."""
    wanted = {(t.id, prompt_hash(t)) for t in tasks}
    return [r for r in records if (r.task_id, r.prompt_sha) in wanted]


def build_registry(records: Sequence[RunRecord]) -> CapabilityRegistry:
    """Worker-role capability profiles from measured outcomes."""
    registry = CapabilityRegistry()
    for r in records:
        registry.record(r.model_id, AgentRole.WORKER, TaskCategory(r.category), success=r.correct)
    return registry


def summarise(records: Sequence[RunRecord], registry: CapabilityRegistry) -> dict:
    """Per model: overall, per category and per difficulty results with intervals."""
    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(r)

    def block(rows: Sequence[RunRecord]) -> dict:
        n = len(rows)
        wins = sum(r.correct for r in rows)
        lo, hi = wilson_interval(wins, n) if n else (0.0, 0.0)
        return {
            "n": n,
            "successes": wins,
            "success_rate": wins / n if n else 0.0,
            "ci95": [lo, hi],
            "mean_latency_s": sum(r.latency_s for r in rows) / n if n else 0.0,
            "mean_cost_usd": sum(r.cost_usd for r in rows) / n if n else 0.0,
            "truncated": sum(r.truncated for r in rows),
        }

    summary: dict[str, dict] = {}
    for model_id, rows in sorted(by_model.items()):
        categories = {}
        for category in sorted({r.category for r in rows}):
            cell = block([r for r in rows if r.category == category])
            est = registry.estimate(model_id, AgentRole.WORKER, TaskCategory(category))
            cell["lower_bound"] = est.lower_bound
            cell["sufficient_evidence"] = est.sufficient_evidence
            categories[category] = cell
        summary[model_id] = {
            "overall": block(rows),
            "by_category": categories,
            "by_difficulty": {
                d: block([r for r in rows if r.difficulty == d])
                for d in sorted({r.difficulty for r in rows})
            },
        }
    return summary
