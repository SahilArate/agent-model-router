"""One command to reproduce the benchmark:  python -m app.benchmark.cli

Writes data/benchmark_results.json and data/capability_profiles.json at the repository root.
"""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from app.benchmark.runner import (
    ROUTER_LEARNED,
    STRATEGIES,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    to_jsonable,
)
from app.core.types import Tier

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "data"

# (label, judge tier, detect rate, false reject rate): how the saving depends on the judge.
SENSITIVITY = (
    ("mid-tier judge, detects 85% (main result)", Tier.MID, 0.85, 0.03),
    ("cheap judge, detects 70%", Tier.SMALL, 0.70, 0.05),
    ("cheap judge, detects 50%", Tier.SMALL, 0.50, 0.05),
    ("mid-tier judge, detects 60%", Tier.MID, 0.60, 0.05),
)


def format_report(result: BenchmarkResult) -> str:
    cfg = result.config
    lines = [
        f"Benchmark: {cfg.n_tasks} tasks, {cfg.repeats} repeats, seed {cfg.seed}",
        f"Split sizes: {result.split_sizes}; low-risk floor chosen on validation: "
        f"{result.chosen_low_floor}",
        "",
        f"{'strategy':<20}{'success':>9}{'  95% CI':>16}{'$/success':>12}"
        f"{'latency':>9}{'escal.':>8}{'abstain':>9}{'false-acc':>10}",
    ]
    for name in STRATEGIES:
        c = result.scorecards[name]
        lo, hi = c.success_ci
        lines.append(
            f"{name:<20}{c.success_rate:>9.1%}{f'[{lo:.1%}, {hi:.1%}]':>16}"
            f"{c.cost_per_success_usd:>12.6f}{c.mean_latency_s:>8.2f}s"
            f"{c.escalation_rate:>8.1%}{c.abstain_rate:>9.1%}{c.false_acceptance_rate:>10.1%}"
        )
    lines += ["", f"{ROUTER_LEARNED} vs each baseline (paired, same tasks):"]
    for p in result.comparisons:
        lo, hi = p.cost_per_success_diff_ci
        slo, shi = p.success_diff_ci
        lines.append(
            f"  vs {p.b:<18} success {p.success_diff:+.1%} [{slo:+.1%}, {shi:+.1%}]  "
            f"$/success {p.cost_per_success_diff:+.6f} [{lo:+.6f}, {hi:+.6f}]  "
            f"McNemar p={p.mcnemar_p:.3g} (only-router {p.a_only}, only-baseline {p.b_only})"
        )
    lines += ["", "Which model produced the final answer, by category (router_learned):"]
    for category, counts in result.routing_mix.items():
        lines.append(
            f"  {category:<14}" + "  ".join(f"{m}: {n}" for m, n in sorted(counts.items()))
        )
    return "\n".join(lines)


async def _sensitivity(base: BenchmarkConfig) -> list[dict]:
    rows = []
    for label, tier, detect, false_reject in SENSITIVITY:
        cfg = replace(
            base,
            judge_tier=tier,
            judge_detect_rate=detect,
            judge_false_reject_rate=false_reject,
        )
        result = await run_benchmark(cfg)
        router = result.scorecards[ROUTER_LEARNED]
        frontier = result.scorecards["fixed_strongest"]
        rows.append(
            {
                "judge": label,
                "router_success": router.success_rate,
                "router_cost_per_success_usd": router.cost_per_success_usd,
                "router_false_acceptance": router.false_acceptance_rate,
                "frontier_success": frontier.success_rate,
                "frontier_cost_per_success_usd": frontier.cost_per_success_usd,
                "saving_vs_frontier": 1
                - router.cost_per_success_usd / frontier.cost_per_success_usd,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the model-routing benchmark.")
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-sensitivity", action="store_true", help="skip the judge sweep")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(n_tasks=args.tasks, repeats=args.repeats, seed=args.seed)
    result = asyncio.run(run_benchmark(config))
    print(format_report(result))

    sensitivity: list[dict] = []
    if not args.no_sensitivity:
        sensitivity = asyncio.run(_sensitivity(config))
        print("\nSensitivity to the judge (router_learned vs fixed_strongest):")
        for row in sensitivity:
            print(
                f"  {row['judge']:<44} success {row['router_success']:.1%} "
                f"(frontier {row['frontier_success']:.1%})  false-acc "
                f"{row['router_false_acceptance']:.1%}  saving {row['saving_vs_frontier']:+.1%}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(result) | {"sensitivity": to_jsonable(sensitivity)}
    (args.out_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2))
    result.registry.save(args.out_dir / "capability_profiles.json")
    print(f"\nWrote results and capability profiles to {args.out_dir}")


if __name__ == "__main__":
    main()
