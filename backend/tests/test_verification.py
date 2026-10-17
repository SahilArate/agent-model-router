import pytest
from pydantic import BaseModel

from app.core.types import Tier
from app.llm.backends import MockBackend
from app.llm.client import LLMClient
from app.registry.models import ModelProfile, Provider
from app.verification.checks import (
    SchemaCheck,
    contains_all,
    matches_regex,
    max_chars,
    non_empty,
)
from app.verification.judge import JudgeCheck, pick_judge
from app.verification.outcome import Action, decide
from app.verification.pipeline import VerificationPipeline
from app.verification.types import (
    CheckResult,
    CheckStatus,
    VerificationContext,
    VerificationReport,
)


def make_model(model_id="worker", provider=Provider.MOCK, price=1.0, **overrides) -> ModelProfile:
    fields = {
        "id": model_id,
        "provider": provider,
        "litellm_model": f"mock/{model_id}",
        "tier": Tier.SMALL,
        "input_price_per_mtok": price,
        "output_price_per_mtok": price,
        "context_window": 100_000,
    }
    return ModelProfile(**{**fields, **overrides})


def ctx(output: str, worker=None) -> VerificationContext:
    return VerificationContext(
        task_prompt="Extract the invoice total.", output=output, worker_model=worker or make_model()
    )


class Invoice(BaseModel):
    total: float
    currency: str


class StubCheck:
    """A check that returns a fixed status and counts how often it ran."""

    def __init__(self, name, status, cost=0.0):
        self.name = name
        self._status = status
        self._cost = cost
        self.runs = 0

    async def run(self, context):
        self.runs += 1
        return CheckResult(self.name, self._status, "stub", cost_usd=self._cost)


# ---- deterministic checks ----


async def test_schema_check_accepts_valid_json_and_fenced_json():
    check = SchemaCheck(Invoice)

    plain = await check.run(ctx('{"total": 12.5, "currency": "USD"}'))
    fenced = await check.run(ctx('```json\n{"total": 12.5, "currency": "USD"}\n```'))

    assert plain.status is CheckStatus.PASS
    assert fenced.status is CheckStatus.PASS


async def test_schema_check_reports_the_bad_field():
    result = await SchemaCheck(Invoice).run(ctx('{"total": "lots", "currency": "USD"}'))

    assert result.status is CheckStatus.FAIL
    assert "total" in result.detail


async def test_schema_check_rejects_non_json():
    result = await SchemaCheck(Invoice).run(ctx("The total is 12.5 dollars."))

    assert result.status is CheckStatus.FAIL


async def test_rule_checks():
    assert (await non_empty().run(ctx("   "))).status is CheckStatus.FAIL
    assert (await non_empty().run(ctx("hi"))).status is CheckStatus.PASS
    assert (await contains_all("total", "USD").run(ctx("Total: 5 usd"))).status is CheckStatus.PASS
    assert (await contains_all("total", "EUR").run(ctx("Total: 5 usd"))).status is CheckStatus.FAIL
    assert (await matches_regex(r"\d+\.\d{2}").run(ctx("pay 12.50"))).status is CheckStatus.PASS
    assert (await max_chars(5).run(ctx("too long"))).status is CheckStatus.FAIL


# ---- pipeline ----


async def test_pipeline_runs_all_checks_when_everything_passes():
    first = StubCheck("a", CheckStatus.PASS, cost=0.001)
    second = StubCheck("b", CheckStatus.PASS, cost=0.002)

    report = await VerificationPipeline([first, second]).verify(ctx("x"))

    assert report.passed
    assert [r.name for r in report.results] == ["a", "b"]
    assert report.cost_usd == pytest.approx(0.003)


async def test_pipeline_stops_at_first_failure_so_the_judge_is_never_paid_for():
    cheap = StubCheck("rule", CheckStatus.FAIL)
    expensive = StubCheck("judge", CheckStatus.PASS, cost=0.5)

    report = await VerificationPipeline([cheap, expensive]).verify(ctx("x"))

    assert not report.passed
    assert expensive.runs == 0
    assert report.cost_usd == 0.0


async def test_pipeline_stops_on_error_and_flags_it():
    report = await VerificationPipeline(
        [StubCheck("judge", CheckStatus.ERROR), StubCheck("later", CheckStatus.PASS)]
    ).verify(ctx("x"))

    assert report.has_error and not report.passed
    assert len(report.results) == 1


def test_empty_pipeline_is_rejected_and_empty_report_never_passes():
    with pytest.raises(ValueError):
        VerificationPipeline([])
    assert VerificationReport(results=()).passed is False


# ---- judge selection ----


def test_judge_prefers_a_different_provider_then_the_cheapest():
    worker = make_model("w", Provider.ANTHROPIC)
    candidates = [
        make_model("same", Provider.ANTHROPIC, price=0.1),
        make_model("other-pricey", Provider.OPENAI, price=5.0),
        make_model("other-cheap", Provider.OPENAI, price=1.0),
    ]

    judge, independent = pick_judge(candidates, worker)

    assert judge.id == "other-cheap" and independent is True


def test_judge_falls_back_to_same_provider_and_flags_it():
    worker = make_model("w", Provider.MOCK)
    judge, independent = pick_judge([worker, make_model("j", Provider.MOCK)], worker)

    assert judge.id == "j" and independent is False


def test_worker_and_disabled_models_are_never_judges():
    worker = make_model("w")

    with pytest.raises(ValueError):
        pick_judge([worker, make_model("off", enabled=False)], worker)


# ---- judge check ----


def judge_check(reply, seen=None):
    def responder(model, messages):
        if seen is not None:
            seen.append(messages[-1].content)
        return reply

    client = LLMClient(MockBackend(responder=responder))
    return JudgeCheck(client, [make_model("worker"), make_model("judge")], "Total must be correct")


async def test_judge_pass_and_fail_carry_cost_and_reason():
    passed = await judge_check('{"pass": true, "reason": "correct"}').run(ctx("12.50"))
    failed = await judge_check('{"pass": false, "reason": "wrong total"}').run(ctx("99"))

    assert passed.status is CheckStatus.PASS and passed.cost_usd > 0
    assert failed.status is CheckStatus.FAIL and "wrong total" in failed.detail


async def test_judge_accepts_fenced_json_and_sees_the_answer_and_rubric():
    seen = []
    check = judge_check('```json\n{"pass": true, "reason": "ok"}\n```', seen)

    result = await check.run(ctx("12.50"))

    assert result.status is CheckStatus.PASS
    assert "12.50" in seen[0] and "Total must be correct" in seen[0]


async def test_unparseable_judge_output_fails_closed_as_error():
    result = await judge_check("looks fine to me!").run(ctx("12.50"))

    assert result.status is CheckStatus.ERROR


async def test_judge_infrastructure_failure_is_an_error_not_a_pass():
    def broken(model, messages):
        raise RuntimeError("provider down")

    check = JudgeCheck(
        LLMClient(MockBackend(responder=broken)),
        [make_model("worker"), make_model("judge")],
        "anything",
    )

    result = await check.run(ctx("12.50"))

    assert result.status is CheckStatus.ERROR


async def test_no_available_judge_is_an_error():
    check = JudgeCheck(LLMClient(MockBackend()), [make_model("worker")], "anything")

    result = await check.run(ctx("12.50"))

    assert result.status is CheckStatus.ERROR


# ---- outcome ----


def report(*statuses):
    return VerificationReport(
        results=tuple(CheckResult(f"c{i}", s, f"detail{i}") for i, s in enumerate(statuses))
    )


def test_pass_means_accept():
    assert decide(report(CheckStatus.PASS), [make_model("up")]).action is Action.ACCEPT


def test_failure_escalates_to_the_next_model_with_evidence():
    fallbacks = [make_model("mid"), make_model("top")]

    outcome = decide(report(CheckStatus.PASS, CheckStatus.FAIL), fallbacks)

    assert outcome.action is Action.ESCALATE
    assert outcome.next_model.id == "mid"
    assert [m.id for m in outcome.remaining_fallbacks] == ["top"]
    assert "c1: detail1" in outcome.failure_evidence


def test_failure_with_nothing_left_abstains():
    outcome = decide(report(CheckStatus.FAIL), [])

    assert outcome.action is Action.ABSTAIN
    assert "no stronger model" in outcome.reason


def test_unverifiable_output_abstains_instead_of_escalating():
    outcome = decide(report(CheckStatus.ERROR), [make_model("mid")])

    assert outcome.action is Action.ABSTAIN
    assert outcome.next_model is None