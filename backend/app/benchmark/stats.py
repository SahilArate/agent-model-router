"""Metrics and significance tests, implemented directly so every number can be audited."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """The result of one strategy on one task in one repeat."""

    strategy: str
    task_id: str
    repeat: int
    accepted: bool  # the strategy returned an answer (did not abstain)
    correct: bool  # the answer matches ground truth
    cost_usd: float
    latency_s: float
    escalations: int
    final_model: str | None

    @property
    def success(self) -> bool:
        return self.accepted and self.correct


@dataclass(frozen=True)
class Scorecard:
    strategy: str
    n: int
    success_rate: float
    success_ci: tuple[float, float]
    total_cost_usd: float
    cost_per_success_usd: float
    mean_latency_s: float
    escalation_rate: float
    abstain_rate: float
    false_acceptance_rate: float  # accepted answers that were wrong, among accepted answers


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def scorecard(strategy: str, outcomes: Sequence[Outcome]) -> Scorecard:
    n = len(outcomes)
    if n == 0:
        raise ValueError("no outcomes to score")
    successes = sum(o.success for o in outcomes)
    accepted = [o for o in outcomes if o.accepted]
    wrong_accepted = sum(not o.correct for o in accepted)
    total_cost = sum(o.cost_usd for o in outcomes)
    return Scorecard(
        strategy=strategy,
        n=n,
        success_rate=successes / n,
        success_ci=wilson_interval(successes, n),
        total_cost_usd=total_cost,
        cost_per_success_usd=total_cost / successes if successes else math.inf,
        mean_latency_s=sum(o.latency_s for o in outcomes) / n,
        escalation_rate=sum(o.escalations > 0 for o in outcomes) / n,
        abstain_rate=1 - len(accepted) / n,
        false_acceptance_rate=wrong_accepted / len(accepted) if accepted else 0.0,
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    a_only: tasks only strategy A solved. b_only: tasks only strategy B solved.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class PairedComparison:
    a: str
    b: str
    a_only: int
    b_only: int
    mcnemar_p: float
    success_diff: float  # a minus b, mean over tasks
    success_diff_ci: tuple[float, float]
    cost_per_success_diff: float  # a minus b (negative means a is cheaper)
    cost_per_success_diff_ci: tuple[float, float]


def _per_task(outcomes: Sequence[Outcome]) -> dict[str, tuple[float, float]]:
    """task id -> (mean success over repeats, mean cost over repeats)."""
    grouped: dict[str, list[Outcome]] = {}
    for o in outcomes:
        grouped.setdefault(o.task_id, []).append(o)
    return {
        task_id: (
            sum(o.success for o in group) / len(group),
            sum(o.cost_usd for o in group) / len(group),
        )
        for task_id, group in grouped.items()
    }


def _cost_per_success(rows: list[tuple[float, float]]) -> float:
    successes = sum(s for s, _ in rows)
    return sum(c for _, c in rows) / successes if successes else math.inf


def _percentile_ci(values: list[float], level: float = 0.95) -> tuple[float, float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return (math.nan, math.nan)
    lo = finite[int((1 - level) / 2 * (len(finite) - 1))]
    hi = finite[int((1 + level) / 2 * (len(finite) - 1))]
    return (lo, hi)


def compare(
    a: str,
    a_outcomes: Sequence[Outcome],
    b: str,
    b_outcomes: Sequence[Outcome],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedComparison:
    """Paired comparison on the same tasks.

    McNemar uses repeat 0 only, so each task is one independent pair. The bootstrap resamples
    tasks (not individual runs), keeping all repeats of a task together.
    """
    first = min(o.repeat for o in a_outcomes)
    a0 = {o.task_id: o.success for o in a_outcomes if o.repeat == first}
    b0 = {o.task_id: o.success for o in b_outcomes if o.repeat == first}
    shared = sorted(a0.keys() & b0.keys())
    a_only = sum(a0[t] and not b0[t] for t in shared)
    b_only = sum(b0[t] and not a0[t] for t in shared)

    pa, pb = _per_task(a_outcomes), _per_task(b_outcomes)
    ids = sorted(pa.keys() & pb.keys())
    rng = random.Random(seed)

    def stat(sample: list[str]) -> tuple[float, float]:
        rows_a, rows_b = [pa[t] for t in sample], [pb[t] for t in sample]
        success = sum(s for s, _ in rows_a) / len(sample) - sum(s for s, _ in rows_b) / len(sample)
        return success, _cost_per_success(rows_a) - _cost_per_success(rows_b)

    point_success, point_cps = stat(ids)
    boot: list[tuple[float, float]] = [
        stat([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(resamples)
    ]
    return PairedComparison(
        a=a,
        b=b,
        a_only=a_only,
        b_only=b_only,
        mcnemar_p=mcnemar_exact(a_only, b_only),
        success_diff=point_success,
        success_diff_ci=_percentile_ci([s for s, _ in boot]),
        cost_per_success_diff=point_cps,
        cost_per_success_diff_ci=_percentile_ci([c for _, c in boot]),
    )
