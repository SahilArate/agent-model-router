import json

import pytest
from routing_helpers import make_model

from app.benchmark.tasks import generate_tasks
from app.core.config import Settings
from app.core.types import RiskLevel, Tier
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.types import ChatMessage
from app.profiling import trace_cli
from app.profiling.guarded import GuardedClient
from app.profiling.limiter import Limits
from app.profiling.runner import Guard, RunLog, RunRecord, prompt_hash
from app.profiling.trace import (
    AttemptResult,
    TaskResult,
    TraceLog,
    describe,
    first_escalation,
    run_trace,
    summarise_trace,
)
from app.registry.capability import CapabilityRegistry
from app.registry.models import Provider

FAST = {"gemini": Limits(100_000, 10**9), "groq": Limits(100_000, 10**9)}
TASKS = generate_tasks(10, seed=21)
EXPECTED = {t.prompt: t.expected for t in TASKS}


async def no_sleep(_s):
    return None


def models():
    cheap = make_model("cheap", Tier.SMALL, 0.01, 0.01, provider=Provider.GEMINI)
    mid = make_model("mid", Tier.MID, 2.0, 10.0, provider=Provider.GEMINI)
    judge = make_model("grader", Tier.MID, 0.15, 0.6, provider=Provider.GROQ)
    return cheap, mid, judge


def responder(counter=None, *, judge_always=None):
    """cheap gives a well-formed wrong number; mid is right; the judge knows the answer."""

    def respond(model, messages):
        system, last = messages[0].content, messages[-1].content
        if counter is not None:
            counter.append(model.id)
        if "strict evaluator" in system:
            prompt = last.split("Task:\n", 1)[1].split("\n\nCriteria:", 1)[0]
            answer = last.rsplit("Answer to evaluate:\n", 1)[1]
            ok = answer.strip() == EXPECTED[prompt]
            if judge_always is not None:
                ok = judge_always
            return json.dumps({"pass": ok, "reason": "matches" if ok else "wrong value"})
        return "999999" if model.id == "cheap" else EXPECTED[messages[1].content]

    return respond


def guarded(responder_fn, guard=None, **kw):
    return GuardedClient(
        LLMClient(MockBackend(responder=responder_fn)),
        guard or Guard(10.0, 10_000),
        limits=FAST,
        retry_delays_s=kw.pop("retry_delays_s", (0.0,)),
        sleep=no_sleep,
        limiter_sleep=no_sleep,
        **kw,
    )


# ---- guarded client ----


async def test_replies_are_recorded_including_failures():
    cheap, _, _ = models()
    calls = []

    def sometimes(model, messages):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("401 invalid API key")  # fatal: not retried
        return "42"

    client = guarded(sometimes)
    with pytest.raises(LLMError):
        await client.complete(cheap, [ChatMessage(role="user", content="hi")])
    await client.complete(cheap, [ChatMessage(role="user", content="hi")])

    assert [c.text for c in client.calls] == [None, "42"]
    assert len(calls) == 2


async def test_transient_errors_are_retried_then_succeed():
    cheap, _, _ = models()
    attempts = []

    def flaky(model, messages):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503 overloaded")
        return "ok"

    client = guarded(flaky, retry_delays_s=(0.0, 0.0))

    result = await client.complete(cheap, [ChatMessage(role="user", content="hi")])

    assert result.text == "ok" and len(attempts) == 3
    assert [c.text for c in client.calls] == ["ok"]


async def test_the_call_cap_stops_calls_with_a_clear_error():
    cheap, _, _ = models()
    client = guarded(lambda m, msgs: "x", Guard(10.0, 1))
    await client.complete(cheap, [ChatMessage(role="user", content="hi")])

    with pytest.raises(LLMError, match="run stopped"):
        await client.complete(cheap, [ChatMessage(role="user", content="hi")])


# ---- the traced run ----


async def run(tmp_path, responder_fn=None, *, tasks=TASKS, guard=None):
    cheap, mid, judge = models()
    client = guarded(responder_fn or responder(), guard)
    log = TraceLog(tmp_path / "trace.jsonl")
    results = await run_trace(
        tasks,
        [cheap, mid],
        judge,
        client,
        CapabilityRegistry(),
        log,
        risk=RiskLevel.LOW,
        stop_reason=lambda: client_guard(client).reason,
        say=lambda _m: None,
    )
    return results, client


def client_guard(client):
    return client._guard


async def test_the_judge_sends_a_wrong_cheap_answer_back_and_the_stronger_model_fixes_it(tmp_path):
    results, _ = await run(tmp_path)

    number_tasks = [r for r in results if r.category in {"extraction", "reasoning", "coding"}]
    assert number_tasks
    for r in number_tasks:
        assert r.routing_mode == "cheap_first" and r.judged is True
        step = r.escalation_steps[0]
        assert (step["from_model"], step["to_model"]) == ("cheap", "mid")
        assert step["check"] == "judge" and step["judge_model"] == "grader"
        assert r.attempts[0].correct is False and r.attempts[1].correct is True
        assert r.accepted and r.correct and r.final_model == "mid"

    summary = summarise_trace(results)
    assert summary["wrong_but_accepted"] == 0
    assert summary["judge"]["detect_rate"]["rate"] == 1.0
    assert summary["judge"]["false_reject_rate"]["rate"] == 0.0
    assert summary["escalated_tasks"]["hits"] >= len(number_tasks)


async def test_malformed_answers_are_caught_by_the_free_check_without_the_judge(tmp_path):
    results, client = await run(tmp_path)

    word_task = next(r for r in results if r.category == "summarization")
    format_failure = word_task.escalation_steps[0]
    assert format_failure["check"] == "format" and format_failure["judge_model"] is None
    assert word_task.attempts[0].checks[-1]["name"] == "format"


async def test_a_blind_judge_that_accepts_everything_lets_wrong_answers_through(tmp_path):
    results, _ = await run(tmp_path, responder(judge_always=True))

    summary = summarise_trace(results)
    assert summary["wrong_but_accepted"] > 0
    assert summary["judge"]["detect_rate"]["rate"] == 0.0


async def test_a_rerun_resumes_without_new_calls(tmp_path):
    counter: list[str] = []
    await run(tmp_path, responder(counter))
    first = len(counter)

    results, _ = await run(tmp_path, responder(counter))

    assert first > 0 and len(counter) == first and len(results) == len(TASKS)


async def test_the_run_stops_when_the_cap_is_reached(tmp_path):
    results, _ = await run(tmp_path, guard=Guard(10.0, 6))

    assert 0 < len(results) < len(TASKS)


async def test_first_escalation_and_description_tell_the_story(tmp_path):
    results, _ = await run(tmp_path)

    example = first_escalation(results)
    text = describe(example, EXPECTED and next(t.prompt for t in TASKS if t.id == example.task_id))

    assert example is not None
    assert "escalated cheap -> mid because judge failed" in text
    assert "WRONG" in text and "grader" in text


def test_summary_counts_judge_mistakes_both_ways():
    def attempt(correct, status):
        return AttemptResult(
            "m", "1", correct, None, "",
            [{"name": "judge", "status": status, "detail": "", "judge_model": "j",
              "independent": True}],
        )  # fmt: skip

    def task(attempts):
        return TaskResult("t", "h", "reasoning", "easy", "low", "proven", "", True, attempts,
                          [], "m", True, True, 0.01, 1.0)  # fmt: skip

    results = [
        task([attempt(False, "fail")]),  # caught a wrong answer
        task([attempt(False, "pass")]),  # let a wrong answer through
        task([attempt(True, "pass")]),  # correctly accepted
        task([attempt(True, "fail")]),  # wrongly rejected a right answer
        task([attempt(True, "error")]),  # unusable verdict
    ]

    judge = summarise_trace(results)["judge"]

    assert judge["detect_rate"]["hits"] == 1 and judge["wrong_answers_judged"] == 2
    assert judge["false_reject_rate"]["hits"] == 1 and judge["correct_answers_judged"] == 2
    assert judge["unusable_verdicts"] == 1


# ---- cli ----


def test_cli_needs_two_workers_and_a_separate_judge():
    with pytest.raises(SystemExit):
        trace_cli.main(["--workers", "gemini-3.8-flash", "--judge", "groq-gpt-oss-120b"])
    with pytest.raises(SystemExit):
        trace_cli.main(
            ["--workers", "gemini-3.8-flash", "groq-gpt-oss-120b", "--judge", "groq-gpt-oss-120b"]
        )


def test_cli_real_run_without_keys_fails_clearly(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        trace_cli.main(
            ["--workers", "gemini-3.5-flash-lite", "gemini-3.8-flash", "--judge",
             "groq-gpt-oss-120b", "--yes", "--out-dir", str(tmp_path)]
        )  # fmt: skip

    assert exc.value.code == 2 and "GROQ_API_KEY" in capsys.readouterr().err


def test_cli_end_to_end_skips_profiled_tasks_and_writes_the_report(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    pool = generate_tasks(40, seed=11)
    answers = {t.prompt: t.expected for t in pool}
    seen: list[str] = []
    profiled = pool[0]
    RunLog(tmp_path / "live_profile_runs.jsonl").append(
        RunRecord("x", profiled.id, prompt_hash(profiled), "extraction", "easy",
                  True, "1", 1, 1, 0.1, 0.0, False)
    )  # fmt: skip

    def respond(model, messages):
        system = messages[0].content
        if "strict evaluator" in system:
            return '{"pass": true, "reason": "ok"}'
        seen.append(messages[1].content)
        return answers[messages[1].content]

    monkeypatch.setattr(
        trace_cli,
        "build_default_client",
        lambda settings, timeout_s: LLMClient(MockBackend(responder=respond)),
    )
    monkeypatch.setattr(trace_cli, "Settings", lambda: Settings(_env_file=None))

    trace_cli.main(
        ["--workers", "gemini-3.5-flash-lite", "gemini-3.8-flash", "--judge", "groq-gpt-oss-120b",
         "--tasks", "5", "--yes", "--rpm", "100000", "--tpm", "100000000",
         "--out-dir", str(tmp_path)]
    )  # fmt: skip

    report = json.loads((tmp_path / "live_routing_trace.json").read_text())
    assert report["summary"]["tasks"] == 5 and len(report["tasks"]) == 5
    assert profiled.prompt not in seen
