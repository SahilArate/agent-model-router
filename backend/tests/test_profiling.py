import json

import pytest
from routing_helpers import make_model

from app.benchmark.tasks import generate_tasks
from app.core.config import Settings
from app.core.types import AgentRole, TaskCategory, Tier
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.profiling import cli
from app.profiling.answers import normalize_answer
from app.profiling.limiter import Limits, RateLimiter, limits_for
from app.profiling.runner import (
    Guard,
    ProfileConfig,
    RunLog,
    build_registry,
    estimate_run,
    is_fatal,
    profile_models,
    prompt_hash,
    select_records,
    summarise,
)
from app.registry.catalog import ModelCatalog
from app.registry.models import Provider

TASKS = generate_tasks(10, seed=3)
EXPECTED = {t.prompt: t.expected for t in TASKS}


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


FAST = Limits(rpm=100_000, tpm=100_000_000)


async def no_sleep(_seconds):
    return None


def live_model(model_id="gem", provider=Provider.GEMINI, **kw):
    return make_model(model_id, Tier.SMALL, 1.0, 2.0, provider=provider, **kw)


def client_with(responder):
    return LLMClient(MockBackend(responder=responder))


def right(model, messages):
    return EXPECTED[messages[-1].content]


async def run(models, responder, tmp_path, *, config=None, tasks=TASKS):
    return await profile_models(
        models,
        tasks,
        client_with(responder),
        RunLog(tmp_path / "runs.jsonl"),
        config or ProfileConfig(retry_delays_s=(0.0, 0.0)),
        limits={"gemini": FAST, "groq": FAST},
        say=lambda _m: None,
        sleep=no_sleep,
        limiter_sleep=no_sleep,
    )


# ---- limiter ----


async def test_limiter_waits_when_requests_per_minute_are_used_up():
    clock = FakeClock()
    limiter = RateLimiter(Limits(rpm=2, tpm=10_000), clock=clock, sleep=clock.sleep)

    for _ in range(3):
        await limiter.acquire(10)

    assert len(clock.sleeps) == 1
    assert clock.now >= 60.0


async def test_limiter_waits_when_tokens_per_minute_are_used_up():
    clock = FakeClock()
    limiter = RateLimiter(Limits(rpm=100, tpm=1_000), clock=clock, sleep=clock.sleep)

    await limiter.acquire(800)
    await limiter.acquire(800)

    assert clock.now >= 60.0


async def test_limiter_lets_one_oversized_call_through_an_empty_window():
    clock = FakeClock()
    limiter = RateLimiter(Limits(rpm=5, tpm=100), clock=clock, sleep=clock.sleep)

    await limiter.acquire(5_000)

    assert clock.sleeps == []


async def test_settle_replaces_the_estimate_with_real_usage():
    clock = FakeClock()
    limiter = RateLimiter(Limits(rpm=100, tpm=1_000), clock=clock, sleep=clock.sleep)

    reservation = await limiter.acquire(900)
    RateLimiter.settle(reservation, 50)
    await limiter.acquire(900)

    assert clock.sleeps == []


def test_limits_have_a_fallback_and_reject_nonsense():
    assert limits_for("groq").rpm < 30
    assert limits_for("someone-new").rpm >= 1
    with pytest.raises(ValueError):
        RateLimiter(Limits(rpm=0, tpm=10))


# ---- answers ----


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  42 \n", "42"),
        ("```\n42\n```", "42"),
        ('```json\n{"tool": "x"}\n```', '{"tool": "x"}'),
        ("<think>hmm</think>\nriver", "river"),
        ("The answer is 42", "The answer is 42"),
    ],
)
def test_normalize_answer_strips_wrappers_only(raw, clean):
    assert normalize_answer(raw) == clean


# ---- estimate and guard ----


def test_estimate_counts_only_work_still_to_do():
    model = live_model()
    done = {(model.id, t.id, prompt_hash(t)) for t in TASKS[:4]}

    estimate = estimate_run([model], TASKS, max_tokens=500, done=done)

    assert estimate.calls == {"gem": 6}
    assert 0 < estimate.typical_cost_usd < estimate.worst_cost_usd
    assert estimate.minutes > 0


def test_guard_stops_on_call_cap_and_on_cost_cap():
    calls = Guard(max_cost_usd=10, max_calls=2)
    assert calls.begin_call(0.0) and calls.begin_call(0.0)
    assert not calls.begin_call(0.0) and "call cap" in calls.reason

    money = Guard(max_cost_usd=0.01, max_calls=100)
    money.charge(0.009)
    assert not money.begin_call(0.005) and "cost cap" in money.reason


# ---- profiling runs ----


async def test_correct_and_wrong_answers_are_recorded_against_ground_truth(tmp_path):
    def sometimes_wrong(model, messages):
        task_prompt = messages[-1].content
        return "definitely wrong" if task_prompt == TASKS[0].prompt else EXPECTED[task_prompt]

    result = await run([live_model()], sometimes_wrong, tmp_path)

    assert len(result.records) == len(TASKS)
    assert sum(not r.correct for r in result.records) == 1
    assert result.guard.spent_usd > 0


async def test_a_rerun_resumes_and_makes_no_new_calls(tmp_path):
    seen = []

    def counting(model, messages):
        seen.append(messages[-1].content)
        return right(model, messages)

    await run([live_model()], counting, tmp_path)
    first = len(seen)
    again = await run([live_model()], counting, tmp_path)

    assert first == len(TASKS) and len(seen) == first
    assert len(again.records) == len(TASKS)


async def test_transient_errors_are_retried_and_not_counted_as_model_failures(tmp_path):
    attempts = {"n": 0}

    def flaky(model, messages):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RuntimeError("429 rate limit, slow down")
        return right(model, messages)

    result = await run([live_model()], flaky, tmp_path)

    assert all(r.correct for r in result.records)
    assert result.statuses["gem"].errors == 0


async def test_a_call_that_keeps_failing_is_skipped_not_graded_wrong(tmp_path):
    def always_down(model, messages):
        raise RuntimeError("503 overloaded")

    result = await run([live_model()], always_down, tmp_path)

    assert result.records == []
    assert result.statuses["gem"].errors > 0
    assert "consecutive" in result.statuses["gem"].aborted


async def test_a_rejected_key_or_unknown_model_stops_that_model_at_once(tmp_path):
    calls = []

    def bad_key(model, messages):
        calls.append(1)
        raise RuntimeError("401 invalid API key")

    result = await run([live_model()], bad_key, tmp_path)

    assert len(calls) == 1
    assert "rejected" in result.statuses["gem"].aborted


async def test_the_call_cap_stops_the_run_and_it_can_resume(tmp_path):
    config = ProfileConfig(max_calls=4, retry_delays_s=(0.0,))
    first = await run([live_model()], right, tmp_path, config=config)
    assert len(first.records) == 4 and "call cap" in first.statuses["gem"].aborted

    second = await run([live_model()], right, tmp_path)
    assert len(second.records) == len(TASKS)


async def test_replies_cut_off_by_the_token_limit_are_flagged(tmp_path):
    config = ProfileConfig(max_tokens=1, retry_delays_s=(0.0,))

    result = await run([live_model()], lambda m, msgs: "x" * 400, tmp_path, config=config)

    assert result.records and all(r.truncated for r in result.records)


async def test_providers_run_together_and_each_model_is_measured(tmp_path):
    models = [live_model("gem"), live_model("grq", Provider.GROQ)]

    result = await run(models, right, tmp_path)

    assert {r.model_id for r in result.records} == {"gem", "grq"}
    assert len(result.records) == 2 * len(TASKS)


# ---- profiles and summary ----


async def test_profiles_and_summary_come_from_measured_results(tmp_path):
    tasks = generate_tasks(50, seed=1)
    expected = {t.prompt: t.expected for t in tasks}

    result = await run(
        [live_model()],
        lambda m, msgs: expected[msgs[-1].content],
        tmp_path,
        tasks=tasks,
    )
    records = select_records(result.records, tasks)
    registry = build_registry(records)
    summary = summarise(records, registry)

    estimate = registry.estimate("gem", AgentRole.WORKER, TaskCategory.EXTRACTION)
    assert estimate.successes == 10 and estimate.failures == 0
    assert estimate.sufficient_evidence is True
    cell = summary["gem"]["by_category"]["extraction"]
    assert cell["n"] == 10 and cell["sufficient_evidence"] is True
    assert summary["gem"]["overall"]["success_rate"] == 1.0
    assert set(summary["gem"]["by_difficulty"]) == {"easy", "medium", "hard"}


def test_records_from_other_tasks_are_left_out(tmp_path):
    other = generate_tasks(10, seed=99)
    log = RunLog(tmp_path / "runs.jsonl")
    assert log.load() == []

    from app.profiling.runner import RunRecord

    stray = RunRecord("gem", other[0].id, prompt_hash(other[0]), "extraction", "easy",
                      True, "1", 1, 1, 0.1, 0.0, False)  # fmt: skip
    assert select_records([stray], TASKS) == []


# ---- cli ----


def test_cli_rejects_unknown_and_mock_models():
    catalog = ModelCatalog.from_file()

    with pytest.raises(SystemExit):
        cli.resolve_models(catalog, ["nope"])
    with pytest.raises(SystemExit):
        cli.resolve_models(catalog, ["mock-small"])
    assert cli.resolve_models(catalog, ["gemini-3.8-flash"])[0].provider is Provider.GEMINI


def test_cli_dry_run_needs_no_key_and_calls_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    cli.main(["--models", "gemini-3.8-flash", "--dry-run", "--out-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "50 calls to make" in out and "GEMINI_API_KEY" in out
    assert not (tmp_path / "live_profile_runs.jsonl").exists()


def test_cli_real_run_without_a_key_fails_clearly(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--models", "gemini-3.8-flash", "--yes", "--out-dir", str(tmp_path)])

    assert exc.value.code == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_cli_end_to_end_writes_profiles_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    tasks = generate_tasks(10, seed=5)
    answers = {t.prompt: t.expected for t in tasks}

    monkeypatch.setattr(
        cli,
        "build_default_client",
        lambda settings, timeout_s: client_with(lambda m, msgs: answers[msgs[-1].content]),
    )
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))

    cli.main(
        ["--models", "gemini-3.8-flash", "--tasks", "10", "--seed", "5", "--yes",
         "--rpm", "100000", "--tpm", "100000000", "--out-dir", str(tmp_path)]
    )  # fmt: skip

    report = json.loads((tmp_path / "live_profile_report.json").read_text())
    assert report["summary"]["gemini-3.8-flash"]["overall"]["n"] == 10
    profiles = json.loads((tmp_path / "live_capability_profiles.json").read_text())
    assert {r["model_id"] for r in profiles["records"]} == {"gemini-3.8-flash"}


def test_rpm_and_tpm_flags_override_the_defaults():
    models = [live_model("a", Provider.GROQ)]

    assert cli.limit_overrides(models, None, None) is None
    assert cli.limit_overrides(models, 5, None)["groq"] == Limits(rpm=5, tpm=limits_for("groq").tpm)
    assert cli.limit_overrides(models, None, 99)["groq"].tpm == 99


@pytest.mark.parametrize(
    ("message", "fatal"),
    [
        ("401 invalid API key", True),
        ("model gemini-9 not found (404)", True),
        ("429 Rate limit reached. Limit 8000, Used 4040, Requested 404", False),
        ("RESOURCE_EXHAUSTED: quota exceeded, not found in free tier", False),
        ("503 service overloaded", False),
        ("call took 4010 ms", False),
    ],
)
def test_fatal_errors_are_told_apart_from_rate_limits(message, fatal):
    assert is_fatal(message) is fatal
