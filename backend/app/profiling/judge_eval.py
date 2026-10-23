"""Measure a judge on its own: given the right answer or a plausible wrong one, does it tell?

Every judge sees the same items, so differences come from the judge alone. Wrong answers are
the benchmark's well-formed but incorrect values (off-by-a-little numbers, a runner-up word,
one altered argument). They are a controlled test, not a sample of any model's real mistakes.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.benchmark.stats import wilson_interval
from app.benchmark.tasks import Task
from app.core.types import Tier
from app.profiling.guarded import GuardedClient
from app.profiling.trace import RUBRIC
from app.registry.models import ModelProfile, Provider
from app.verification.judge import JudgeCheck
from app.verification.types import CheckStatus, VerificationContext

# Stands in for "whoever wrote the answer", so no judge ever grades itself.
AUTHOR = ModelProfile(
    id="answer-author",
    provider=Provider.MOCK,
    litellm_model="mock/author",
    tier=Tier.SMALL,
    input_price_per_mtok=0,
    output_price_per_mtok=0,
    context_window=1_000_000,
)


@dataclass(frozen=True)
class JudgeItem:
    task: Task
    output: str
    correct: bool


@dataclass(frozen=True)
class Verdict:
    judge_id: str
    task_id: str
    category: str
    answer_correct: bool
    status: str  # pass, fail or error
    reason: str
    cost_usd: float


def build_items(tasks: Sequence[Task]) -> list[JudgeItem]:
    """For every task, the correct answer and one plausible wrong answer."""
    items = []
    for task in tasks:
        wrong = next((w for w in task.wrong_answers if w != task.expected), None)
        if wrong is None:
            continue
        items.append(JudgeItem(task, task.expected, True))
        items.append(JudgeItem(task, wrong, False))
    return items


async def evaluate_judge(
    judge: ModelProfile,
    items: Sequence[JudgeItem],
    client: GuardedClient,
    *,
    stop_reason: Callable[[], str | None] = lambda: None,
    say: Callable[[str], None] = print,
) -> list[Verdict]:
    check = JudgeCheck(client, (judge, AUTHOR), RUBRIC, preferred_id=judge.id)  # type: ignore[arg-type]
    verdicts: list[Verdict] = []
    for index, item in enumerate(items, 1):
        if stop_reason() is not None:
            say(f"  {judge.id}: stopping: {stop_reason()}")
            break
        result = await check.run(VerificationContext(item.task.prompt, item.output, AUTHOR))
        verdicts.append(
            Verdict(
                judge_id=judge.id,
                task_id=item.task.id,
                category=item.task.category.value,
                answer_correct=item.correct,
                status=result.status.value,
                reason=result.detail,
                cost_usd=result.cost_usd,
            )
        )
        if index % 20 == 0:
            say(f"  {judge.id}: {index} of {len(items)} judged")
    return verdicts


def _rate(hits: int, n: int) -> dict:
    lo, hi = wilson_interval(hits, n) if n else (0.0, 0.0)
    return {"hits": hits, "n": n, "rate": hits / n if n else None, "ci95": [lo, hi]}


def _block(verdicts: Sequence[Verdict]) -> dict:
    usable = [v for v in verdicts if v.status != CheckStatus.ERROR.value]
    wrong = [v for v in usable if not v.answer_correct]
    right = [v for v in usable if v.answer_correct]
    return {
        "catches_wrong_answers": _rate(sum(v.status == "fail" for v in wrong), len(wrong)),
        "wrongly_rejects_right_answers": _rate(sum(v.status == "fail" for v in right), len(right)),
        "unusable_verdicts": len(verdicts) - len(usable),
    }


def summarise_judges(verdicts: Sequence[Verdict]) -> dict:
    """Per judge: overall and per task type catch rate and false-reject rate, with intervals."""
    summary: dict[str, dict] = {}
    for judge_id in sorted({v.judge_id for v in verdicts}):
        rows = [v for v in verdicts if v.judge_id == judge_id]
        summary[judge_id] = _block(rows) | {
            "cost_usd": sum(v.cost_usd for v in rows),
            "by_category": {
                c: _block([v for v in rows if v.category == c])
                for c in sorted({v.category for v in rows})
            },
        }
    return summary


def format_summary(summary: dict) -> str:
    lines = []
    for judge_id, s in summary.items():
        c, f = s["catches_wrong_answers"], s["wrongly_rejects_right_answers"]
        lines.append(
            f"\n{judge_id}: catches {c['hits']}/{c['n']} wrong answers "
            f"(95% CI {c['ci95'][0]:.0%}-{c['ci95'][1]:.0%}); wrongly rejects "
            f"{f['hits']}/{f['n']} right answers (95% CI {f['ci95'][0]:.0%}-{f['ci95'][1]:.0%}); "
            f"unusable verdicts {s['unusable_verdicts']}; notional ${s['cost_usd']:.4f}"
        )
        lines.append(f"  {'category':<15}{'catches wrong':>15}{'wrongly rejects right':>24}")
        for category, b in s["by_category"].items():
            cw, fr = b["catches_wrong_answers"], b["wrongly_rejects_right_answers"]
            caught = f"{cw['hits']}/{cw['n']}"
            rejected = f"{fr['hits']}/{fr['n']}"
            lines.append(f"  {category:<15}{caught:>15}{rejected:>24}")
    return "\n".join(lines)


def to_json(verdicts: Sequence[Verdict], summary: dict) -> str:
    return json.dumps({"summary": summary, "verdicts": [v.__dict__ for v in verdicts]}, indent=2)
