"""The benchmark: five strategies on the same tasks, scored against ground truth.

Protocol
  1. Generate tasks and split them 40/20/40 into train / validation / test.
  2. Profile: every model answers the train tasks directly; successes and failures feed the
     capability registry (this is how the router learns what each model is good at).
  3. Tune: the low-risk quality floor is chosen on the validation split only.
  4. Evaluate: every strategy runs on the held-out test split, `repeats` times with different
     random draws. The test split is touched once, after all choices are fixed.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.benchmark.grading import is_correct
from app.benchmark.stats import (
    Outcome,
    PairedComparison,
    Scorecard,
    compare,
    scorecard,
)
from app.benchmark.tasks import Split, Task, generate_tasks, split_tasks
from app.benchmark.verifiers import build_pipeline
from app.benchmark.world import LATENCY, SimulatedWorld
from app.core.types import AgentRole, RiskLevel, Tier
from app.llm.backends import MockBackend, estimate_tokens
from app.llm.client import LLMClient
from app.llm.types import ChatMessage
from app.registry.capability import CapabilityRegistry
from app.registry.catalog import ModelCatalog
from app.registry.models import ModelProfile
from app.router.engine import ModelRouter
from app.router.policy import RoutingPolicy
from app.workflow.budget import Budget
from app.workflow.executor import RoutedExecutor
from app.workflow.spec import CallSpec

MAX_OUTPUT_TOKENS = 64
JUDGE_LATENCY_S = 0.5
PROFILING_SEED_OFFSET = 10_000

FIXED_CHEAP = "fixed_cheap"
FIXED_STRONGEST = "fixed_strongest"
COMPLEXITY_ONLY = "complexity_only"
ROUTER_COLD = "router_cold_start"
ROUTER_LEARNED = "router_learned"
STRATEGIES = (FIXED_CHEAP, FIXED_STRONGEST, COMPLEXITY_ONLY, ROUTER_COLD, ROUTER_LEARNED)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_tasks: int = 1000
    repeats: int = 5
    seed: int = 0
    judge_detect_rate: float = 0.85
    judge_false_reject_rate: float = 0.03
    judge_tier: Tier = Tier.MID
    quality_target: float = 0.93  # validation success the tuned router must reach
    floor_grid: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    validation_repeats: int = 2
    bootstrap_resamples: int = 2000


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    split_sizes: dict[str, int]
    chosen_low_floor: float
    validation: list[dict[str, float]]
    scorecards: dict[str, Scorecard]
    comparisons: list[PairedComparison]
    by_category: dict[str, dict[str, dict[str, float]]]
    routing_mix: dict[str, dict[str, int]]
    registry: CapabilityRegistry = field(repr=False, default_factory=CapabilityRegistry)


def simulated_latency(model_id: str, output_tokens: int) -> float:
    ttft, tokens_per_s = LATENCY[model_id]
    return ttft + output_tokens / tokens_per_s


def _models(catalog: ModelCatalog) -> list[ModelProfile]:
    return sorted(catalog.all(enabled_only=True), key=lambda m: m.tier.rank)


def _message(task: Task) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=task.prompt)]


async def _run_direct(
    strategy: str,
    pick_model: Callable[[Task], ModelProfile],
    task: Task,
    client: LLMClient,
    repeat: int,
) -> Outcome:
    """One call to one model, no verification: what most deployments do today."""
    model = pick_model(task)
    result = await client.complete(model, _message(task), max_tokens=MAX_OUTPUT_TOKENS)
    return Outcome(
        strategy=strategy,
        task_id=task.id,
        repeat=repeat,
        accepted=True,
        correct=is_correct(task, result.text),
        cost_usd=result.cost_usd,
        latency_s=simulated_latency(model.id, result.output_tokens),
        escalations=0,
        final_model=model.id,
    )


async def _run_routed(
    strategy: str,
    executor: RoutedExecutor,
    task: Task,
    world: SimulatedWorld,
    judge_model: ModelProfile,
    config: BenchmarkConfig,
    repeat: int,
) -> Outcome:
    spec = CallSpec(
        agent=AgentRole.WORKER,
        category=task.category,
        risk=task.risk,
        messages=tuple(_message(task)),
        pipeline=build_pipeline(
            task,
            world,
            judge_model,
            detect_rate=config.judge_detect_rate,
            false_reject_rate=config.judge_false_reject_rate,
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    trace = await executor.run(spec, Budget(max_cost_usd=1.0, max_calls=6))
    latency = sum(
        simulated_latency(a.model_id, a.output_tokens)
        + JUDGE_LATENCY_S * sum(c.name == "judge" for c in a.checks)
        for a in trace.attempts
    )
    return Outcome(
        strategy=strategy,
        task_id=task.id,
        repeat=repeat,
        accepted=trace.accepted,
        correct=is_correct(task, trace.output),
        cost_usd=trace.cost_usd,
        latency_s=latency,
        escalations=trace.escalations,
        final_model=trace.attempts[-1].model_id if trace.attempts else None,
    )


def _judge_model(models: list[ModelProfile], config: BenchmarkConfig) -> ModelProfile:
    return next(m for m in models if m.tier is config.judge_tier)


def _client_for(world: SimulatedWorld) -> LLMClient:
    return LLMClient(MockBackend(world))


async def learn_profiles(
    train: list[Task], models: list[ModelProfile], all_tasks: list[Task], seed: int
) -> CapabilityRegistry:
    """Profiling run: each model answers every train task once; ground truth decides success."""
    world = SimulatedWorld(all_tasks, seed=seed + PROFILING_SEED_OFFSET)
    client = _client_for(world)
    registry = CapabilityRegistry()
    for model in models:
        for task in train:
            result = await client.complete(model, _message(task), max_tokens=MAX_OUTPUT_TOKENS)
            registry.record(
                model.id, AgentRole.WORKER, task.category, success=is_correct(task, result.text)
            )
    return registry


def _policy(low_floor: float, judge_model: ModelProfile) -> RoutingPolicy:
    return RoutingPolicy(
        quality_floors={
            RiskLevel.LOW: low_floor,
            RiskLevel.MEDIUM: 0.90,
            RiskLevel.HIGH: 0.95,
        },
        verify_cost_usd=judge_model.cost(120, 20),
    )


def _complexity_thresholds(train: list[Task]) -> tuple[int, int]:
    """Prompt-length terciles from the train split: the 'complexity only' baseline's rule."""
    lengths = sorted(estimate_tokens(t.prompt) for t in train)
    return lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]


async def _run_strategy(
    name: str,
    tasks: list[Task],
    all_tasks: list[Task],
    models: list[ModelProfile],
    registry: CapabilityRegistry,
    low_floor: float,
    thresholds: tuple[int, int],
    config: BenchmarkConfig,
    repeats: int,
) -> list[Outcome]:
    cheap, strongest = models[0], models[-1]
    judge_model = _judge_model(models, config)
    low_cut, high_cut = thresholds

    def by_complexity(task: Task) -> ModelProfile:
        tokens = estimate_tokens(task.prompt)
        return models[0] if tokens <= low_cut else models[1] if tokens <= high_cut else models[-1]

    direct: dict[str, Callable[[Task], ModelProfile]] = {
        FIXED_CHEAP: lambda _t: cheap,
        FIXED_STRONGEST: lambda _t: strongest,
        COMPLEXITY_ONLY: by_complexity,
    }

    def make_runner(
        repeat: int, world: SimulatedWorld, client: LLMClient
    ) -> Callable[[Task], Awaitable[Outcome]]:
        if name in direct:
            pick_model = direct[name]
            return lambda task: _run_direct(name, pick_model, task, client, repeat)
        registry_used = registry if name == ROUTER_LEARNED else CapabilityRegistry()
        router = ModelRouter(models, registry_used, _policy(low_floor, judge_model))
        executor = RoutedExecutor(router, client)
        return lambda task: _run_routed(name, executor, task, world, judge_model, config, repeat)

    outcomes: list[Outcome] = []
    for repeat in range(repeats):
        world = SimulatedWorld(all_tasks, seed=config.seed + repeat)
        run_one = make_runner(repeat, world, _client_for(world))
        for task in tasks:
            outcomes.append(await run_one(task))
    return outcomes


def _tune_floor(validation_results: dict[float, Scorecard], quality_target: float) -> float:
    """Cheapest cost-per-success among floors that reach the quality target; else best success."""
    meeting = {f: s for f, s in validation_results.items() if s.success_rate >= quality_target}
    if meeting:
        return min(meeting, key=lambda f: (meeting[f].cost_per_success_usd, -f))
    return max(validation_results, key=lambda f: (validation_results[f].success_rate, f))


def _category_table(
    outcomes: list[Outcome], tasks_by_id: Mapping[str, Task]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        grouped[tasks_by_id[o.task_id].category.value].append(o)
    table: dict[str, dict[str, float]] = {}
    for category, group in sorted(grouped.items()):
        card = scorecard("", group)
        table[category] = {
            "success_rate": card.success_rate,
            "cost_per_success_usd": card.cost_per_success_usd,
        }
    return table


async def run_benchmark(
    config: BenchmarkConfig | None = None, catalog: ModelCatalog | None = None
) -> BenchmarkResult:
    config = config or BenchmarkConfig()
    models = _models(catalog or ModelCatalog.from_file())
    all_tasks = generate_tasks(config.n_tasks, seed=config.seed)
    tasks_by_id = {t.id: t for t in all_tasks}
    splits = split_tasks(all_tasks, seed=config.seed)
    train, validation, test = splits[Split.TRAIN], splits[Split.VALIDATION], splits[Split.TEST]

    registry = await learn_profiles(train, models, all_tasks, config.seed)
    thresholds = _complexity_thresholds(train)

    validation_cards: dict[float, Scorecard] = {}
    for floor in config.floor_grid:
        outcomes = await _run_strategy(
            ROUTER_LEARNED, validation, all_tasks, models, registry, floor, thresholds,
            config, config.validation_repeats,
        )  # fmt: skip
        validation_cards[floor] = scorecard(ROUTER_LEARNED, outcomes)
    chosen = _tune_floor(validation_cards, config.quality_target)

    outcomes_by_strategy: dict[str, list[Outcome]] = {}
    for name in STRATEGIES:
        outcomes_by_strategy[name] = await _run_strategy(
            name, test, all_tasks, models, registry, chosen, thresholds, config, config.repeats
        )

    cards = {name: scorecard(name, outs) for name, outs in outcomes_by_strategy.items()}
    comparisons = [
        compare(
            ROUTER_LEARNED,
            outcomes_by_strategy[ROUTER_LEARNED],
            other,
            outcomes_by_strategy[other],
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        for other in STRATEGIES
        if other != ROUTER_LEARNED
    ]
    mix: dict[str, Counter[str]] = defaultdict(Counter)
    for o in outcomes_by_strategy[ROUTER_LEARNED]:
        if o.final_model:
            mix[tasks_by_id[o.task_id].category.value][o.final_model] += 1

    return BenchmarkResult(
        config=config,
        split_sizes={s.value: len(v) for s, v in splits.items()},
        chosen_low_floor=chosen,
        validation=[
            {
                "low_floor": f,
                "success_rate": c.success_rate,
                "cost_per_success_usd": c.cost_per_success_usd,
            }
            for f, c in validation_cards.items()
        ],
        scorecards=cards,
        comparisons=comparisons,
        by_category={
            name: _category_table(outs, tasks_by_id) for name, outs in outcomes_by_strategy.items()
        },
        routing_mix={cat: dict(counter) for cat, counter in sorted(mix.items())},
        registry=registry,
    )


def to_jsonable(value: Any) -> Any:
    """Recursively convert results to JSON-safe values (infinity and NaN become null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: to_jsonable(getattr(value, k)) for k in value.__dataclass_fields__ if k != "registry"
        }
    return value
