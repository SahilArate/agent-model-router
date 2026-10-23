import json

import pytest
from routing_helpers import make_model

from app.benchmark.tasks import generate_tasks
from app.core.config import Settings
from app.core.types import Tier
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.profiling import judge_cli
from app.profiling.guarded import GuardedClient
from app.profiling.judge_eval import (
    build_items,
    evaluate_judge,
    format_summary,
    summarise_judges,
)
from app.profiling.limiter import Limits
from app.profiling.runner import Guard
from app.registry.models import Provider

TASKS = generate_tasks(10, seed=31)
EXPECTED = {t.prompt: t.expected for t in TASKS}
FAST = {"groq": Limits(100_000, 10**9), "gemini": Limits(100_000, 10**9)}


async def no_sleep(_s):
    return None


def judge_model(model_id="j", provider=Provider.GROQ):
    return make_model(model_id, Tier.MID, 0.15, 0.6, provider=provider)


def client_for(respond, guard=None):
    return GuardedClient(
        LLMClient(MockBackend(responder=respond)),
        guard or Guard(10.0, 10_000),
        limits=FAST,
        retry_delays_s=(0.0,),
        sleep=no_sleep,
        limiter_sleep=no_sleep,
    )


def verdict_json(ok):
    return json.dumps({"pass": ok, "reason": "ok" if ok else "wrong"})


def perfect(model, messages):
    last = messages[-1].content
    prompt = last.split("Task:\n", 1)[1].split("\n\nCriteria:", 1)[0]
    answer = last.rsplit("Answer to evaluate:\n", 1)[1].strip()
    return verdict_json(answer == EXPECTED[prompt])


async def run(respond, *, guard=None):
    items = build_items(TASKS)
    verdicts = await evaluate_judge(
        judge_model(), items, client_for(respond, guard), say=lambda _m: None,
        stop_reason=(lambda: guard.reason) if guard else (lambda: None),
    )  # fmt: skip
    return verdicts, summarise_judges(verdicts)["j"]


def test_every_task_gives_a_right_and_a_different_wrong_item():
    items = build_items(TASKS)

    assert len(items) == 2 * len(TASKS)
    for right, wrong in zip(items[::2], items[1::2], strict=True):
        assert right.task is wrong.task
        assert right.correct and not wrong.correct and right.output != wrong.output


async def test_a_perfect_judge_catches_everything_and_rejects_nothing_right():
    _, s = await run(perfect)

    assert s["catches_wrong_answers"]["rate"] == 1.0
    assert s["wrongly_rejects_right_answers"]["rate"] == 0.0
    assert s["unusable_verdicts"] == 0 and s["cost_usd"] > 0
    assert set(s["by_category"]) == {t.category.value for t in TASKS}


async def test_a_judge_that_accepts_everything_catches_nothing():
    _, s = await run(lambda m, msgs: verdict_json(True))

    assert s["catches_wrong_answers"]["rate"] == 0.0
    assert s["wrongly_rejects_right_answers"]["rate"] == 0.0


async def test_a_judge_that_rejects_everything_is_caught_rejecting_right_answers():
    _, s = await run(lambda m, msgs: verdict_json(False))

    assert s["catches_wrong_answers"]["rate"] == 1.0
    assert s["wrongly_rejects_right_answers"]["rate"] == 1.0


async def test_unusable_verdicts_are_counted_but_not_scored():
    _, s = await run(lambda m, msgs: "I cannot decide")

    assert s["unusable_verdicts"] == 2 * len(TASKS)
    assert s["catches_wrong_answers"]["n"] == 0


async def test_the_run_stops_when_a_cap_is_hit():
    verdicts, _ = await run(perfect, guard=Guard(10.0, 5))

    assert 0 < len(verdicts) < 2 * len(TASKS)


async def test_the_summary_is_readable():
    _, _ = await run(perfect)
    verdicts, _ = await run(perfect)

    text = format_summary(summarise_judges(verdicts))

    assert "catches 10/10 wrong answers" in text and "extraction" in text


def test_cli_dry_run_needs_no_keys(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    judge_cli.main(["--judges", "groq-gpt-oss-120b", "--dry-run", "--tasks", "5"])

    out = capsys.readouterr().out
    assert "10 items" in out and "GROQ_API_KEY" in out


def test_cli_real_run_without_keys_fails_clearly(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        judge_cli.main(["--judges", "groq-gpt-oss-120b", "--yes", "--out-dir", str(tmp_path)])

    assert exc.value.code == 2 and "GROQ_API_KEY" in capsys.readouterr().err


def test_cli_end_to_end_writes_the_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    expected = {t.prompt: t.expected for t in generate_tasks(6, seed=11)}

    def respond(model, messages):
        last = messages[-1].content
        prompt = last.split("Task:\n", 1)[1].split("\n\nCriteria:", 1)[0]
        answer = last.rsplit("Answer to evaluate:\n", 1)[1].strip()
        return verdict_json(answer == expected[prompt])

    monkeypatch.setattr(
        judge_cli,
        "build_default_client",
        lambda settings, timeout_s: LLMClient(MockBackend(responder=respond)),
    )
    monkeypatch.setattr(judge_cli, "Settings", lambda: Settings(_env_file=None))

    judge_cli.main(
        ["--judges", "groq-gpt-oss-120b", "gemini-3.5-flash-lite", "--tasks", "6", "--yes",
         "--rpm", "100000", "--tpm", "100000000", "--out-dir", str(tmp_path)]
    )  # fmt: skip

    data = json.loads((tmp_path / "live_judge_eval.json").read_text())
    assert set(data["summary"]) == {"groq-gpt-oss-120b", "gemini-3.5-flash-lite"}
    assert data["summary"]["groq-gpt-oss-120b"]["catches_wrong_answers"]["n"] == 6
