import asyncio
import json

import pytest

from app.benchmark.grading import is_correct, is_well_formed
from app.benchmark.runner import (
    FIXED_CHEAP,
    FIXED_STRONGEST,
    ROUTER_LEARNED,
    BenchmarkConfig,
    run_benchmark,
    to_jsonable,
)
from app.benchmark.stats import Outcome, compare, mcnemar_exact, scorecard, wilson_interval
from app.benchmark.tasks import AnswerKind, Split, generate_tasks, split_tasks
from app.benchmark.world import SimulatedWorld
from app.core.types import AgentRole, TaskCategory
from app.llm.types import ChatMessage
from app.registry.catalog import ModelCatalog


def test_tasks_are_deterministic_unique_and_balanced():
    a, b = generate_tasks(200, seed=3), generate_tasks(200, seed=3)
    assert a == b
    assert len({t.prompt for t in a}) == 200
    assert {t.category for t in a} == set(TaskCategory)
    assert all(t.expected not in t.wrong_answers for t in a)


def test_every_task_expected_answer_grades_correct_and_wrong_ones_do_not():
    for task in generate_tasks(150, seed=1):
        assert is_correct(task, task.expected)
        for wrong in task.wrong_answers:
            assert is_well_formed(task.kind, wrong)
            assert not is_correct(task, wrong)


def test_grading_rejects_malformed_and_missing_answers():
    task = generate_tasks(1, seed=0)[0]
    assert not is_correct(task, None)
    assert not is_correct(task, "no idea")
    assert not is_well_formed(AnswerKind.JSON, "[1, 2]")


def test_split_is_disjoint_complete_and_stable():
    tasks = generate_tasks(300, seed=0)
    parts = split_tasks(tasks, seed=0)
    ids = [t.id for split in Split for t in parts[split]]
    assert sorted(ids) == sorted(t.id for t in tasks)
    assert len(ids) == len(set(ids))
    assert parts == split_tasks(tasks, seed=0)
    assert len(parts[Split.VALIDATION]) < len(parts[Split.TRAIN])


def test_world_is_reproducible_and_escalation_gets_a_fresh_roll():
    tasks = generate_tasks(60, seed=0)
    model = ModelCatalog.from_file().get("mock-small")
    world = SimulatedWorld(tasks, seed=1)
    first = [world(model, [ChatMessage(role="user", content=t.prompt)]) for t in tasks]
    again = [world(model, [ChatMessage(role="user", content=t.prompt)]) for t in tasks]
    assert first == again
    longer = [world(model, [ChatMessage(role="user", content=t.prompt)] * 2) for t in tasks]
    assert first != longer


def test_world_small_model_is_worse_than_frontier_on_hard_reasoning():
    tasks = [
        t for t in generate_tasks(900, seed=0)
        if t.category is TaskCategory.REASONING and t.difficulty.value == "hard"
    ]  # fmt: skip
    world = SimulatedWorld(tasks, seed=0)
    assert world.p_correct("mock-small", tasks[0]) < world.p_correct("mock-frontier", tasks[0])


def test_wilson_interval_contains_the_estimate_and_handles_empty():
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.9 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def _outcome(strategy, task_id, repeat, success, cost):
    return Outcome(strategy, task_id, repeat, True, success, cost, 1.0, 0, "m")


def test_scorecard_metrics():
    outcomes = [
        _outcome("s", "a", 0, True, 1.0),
        _outcome("s", "b", 0, False, 3.0),
        Outcome("s", "c", 0, False, False, 2.0, 1.0, 1, None),
    ]
    card = scorecard("s", outcomes)
    assert card.success_rate == pytest.approx(1 / 3)
    assert card.cost_per_success_usd == pytest.approx(6.0)
    assert card.abstain_rate == pytest.approx(1 / 3)
    assert card.false_acceptance_rate == pytest.approx(0.5)
    assert card.escalation_rate == pytest.approx(1 / 3)


def test_compare_detects_a_clear_winner():
    ids = [f"t{i}" for i in range(80)]
    good = [_outcome("a", t, 0, True, 1.0) for t in ids]
    bad = [_outcome("b", t, 0, i % 4 != 0, 1.0) for i, t in enumerate(ids)]
    result = compare("a", good, "b", bad, resamples=300)
    assert result.a_only == 20 and result.b_only == 0
    assert result.mcnemar_p < 0.001
    assert result.success_diff_ci[0] > 0


@pytest.fixture(scope="module")
def small_run():
    config = BenchmarkConfig(n_tasks=300, repeats=2, bootstrap_resamples=100)
    return asyncio.run(run_benchmark(config))


def test_benchmark_is_reproducible(small_run):
    config = BenchmarkConfig(n_tasks=300, repeats=2, bootstrap_resamples=100)
    again = asyncio.run(run_benchmark(config))
    assert to_jsonable(again) == to_jsonable(small_run)


def test_benchmark_ordering_makes_sense(small_run):
    cards = small_run.scorecards
    assert cards[FIXED_STRONGEST].success_rate > cards[FIXED_CHEAP].success_rate
    assert cards[ROUTER_LEARNED].success_rate > cards[FIXED_CHEAP].success_rate
    assert cards[ROUTER_LEARNED].false_acceptance_rate < cards[FIXED_CHEAP].false_acceptance_rate
    assert cards[FIXED_CHEAP].cost_per_success_usd < cards[FIXED_STRONGEST].cost_per_success_usd


def test_benchmark_learns_profiles_from_the_train_split_only(small_run):
    train_total = sum(
        small_run.registry.estimate("mock-small", AgentRole.WORKER, c).sample_size
        for c in TaskCategory
    )
    assert train_total == small_run.split_sizes["train"]


def test_results_serialise_to_valid_json(small_run):
    text = json.dumps(to_jsonable(small_run))
    assert "Infinity" not in text and "NaN" not in text
