"""Run the real router and judge on held-out tasks and record what happened.

    python -m app.profiling.trace_cli --workers gemini-3.5-flash-lite gemini-3.8-flash ^
        --judge groq-gpt-oss-120b --dry-run

Run from the backend folder after profiling (python -m app.profiling.cli). Uses the measured
profiles in data/live_capability_profiles.json and tasks the profiler never saw.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.benchmark.tasks import generate_tasks
from app.core.config import Settings
from app.core.types import RiskLevel, TaskCategory
from app.llm.client import build_default_client
from app.profiling.cli import DEFAULT_OUT_DIR, fail, limit_overrides, missing_keys, resolve_models
from app.profiling.guarded import GuardedClient
from app.profiling.runner import Guard, RunLog, prompt_hash
from app.profiling.trace import (
    MAX_OUTPUT_TOKENS,
    TraceLog,
    baseline_costs,
    describe,
    first_escalation,
    run_trace,
    summarise_trace,
)
from app.registry.capability import CapabilityRegistry
from app.registry.catalog import ModelCatalog


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace the live router and judge on real tasks.")
    parser.add_argument("--workers", nargs="+", required=True, help="models that write answers")
    parser.add_argument("--judge", required=True, help="dedicated judge model (never a worker)")
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11, help="differs from the profiling seed")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in TaskCategory],
        help="only trace these task types (declare this when reporting results)",
    )
    parser.add_argument("--label", default="", help="suffix for the output file names")
    parser.add_argument("--risk", choices=[r.value for r in RiskLevel], default="low")
    parser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-cost", type=float, default=0.50, help="notional USD cap")
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--rpm", type=int)
    parser.add_argument("--tpm", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    if len(args.workers) < 2:
        fail("give at least two workers, otherwise there is nothing to escalate to")
    if args.judge in args.workers:
        fail("the judge must not also be a worker: it would grade its own answers")

    settings = Settings()
    catalog = ModelCatalog.from_file()
    workers = [
        m.model_copy(update={"enabled": True}) for m in resolve_models(catalog, args.workers)
    ]
    judge = resolve_models(catalog, [args.judge])[0].model_copy(update={"enabled": True})

    profile_log = RunLog(args.out_dir / "live_profile_runs.jsonl")
    profiled = {r.prompt_sha for r in profile_log.load()}
    wanted = set(args.categories) if args.categories else {c.value for c in TaskCategory}
    pool = generate_tasks(args.tasks * 12, args.seed)
    tasks = [t for t in pool if t.category.value in wanted and prompt_hash(t) not in profiled]
    tasks = tasks[: args.tasks]
    suffix = f"_{args.label}" if args.label else ""
    registry = CapabilityRegistry()
    profiles_path = args.out_dir / "live_capability_profiles.json"
    if profiles_path.exists():
        registry.load(profiles_path)
        print(f"Using measured profiles from {profiles_path}")
    else:
        print("No measured profiles found: routing starts cold (run the profiler first).")

    missing = missing_keys([*workers, judge], settings)
    print(f"{len(tasks)} held-out tasks, {args.risk} risk; workers {', '.join(args.workers)}; "
          f"judge {args.judge}")  # fmt: skip
    print("Roughly 1 to 4 model calls per task; throttled to stay inside free-tier limits.")
    if args.dry_run:
        if missing:
            print(f"Missing before a real run: {', '.join(missing)} (put them in backend/.env)")
        return
    if missing:
        fail(f"missing API key(s): {', '.join(missing)}. Put them in backend/.env and retry.")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("cancelled")

    guard = Guard(args.max_cost, args.max_calls)
    client = GuardedClient(
        build_default_client(settings, timeout_s=120.0),
        guard,
        limits=limit_overrides([*workers, judge], args.rpm, args.tpm),
    )
    log = TraceLog(args.out_dir / f"live_routing_trace{suffix}.jsonl")
    results = asyncio.run(
        run_trace(
            tasks, workers, judge, client, registry, log,
            risk=RiskLevel(args.risk), max_output_tokens=args.max_tokens,
            stop_reason=lambda: guard.reason,
        )
    )  # fmt: skip

    summary = summarise_trace(results)
    print(json.dumps(summary, indent=2))
    prompts = {t.id: t.prompt for t in tasks}
    example = first_escalation(results)
    if example:
        print("\nFirst escalation caused by the judge:")
    else:
        example = first_escalation(results, judge_only=False)
        print(
            "\nNo judge-driven escalation occurred."
            + (" First escalation of another kind:" if example else "")
        )
    if example:
        print(describe(example, prompts[example.task_id]))

    own = [r for r in profile_log.load() if r.model_id in set(args.workers)]
    baselines = baseline_costs(own, wanted)
    print("\nFor comparison, each worker alone with no judge (profiling run, same task types):")
    for model_id, b in baselines.items():
        cps = b["cost_per_success_usd"]
        print(
            f"  {model_id}: {b['success']['hits']}/{b['n']} correct, ${cps:.6f} per success"
            if cps
            else f"  {model_id}: no correct answers"
        )
    cps = summary["cost_per_success_usd"]
    if cps:
        print(f"  routed system with judge: ${cps:.6f} per success")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "workers": args.workers,
        "judge": args.judge,
        "risk": args.risk,
        "seed": args.seed,
        "categories": sorted(wanted),
        "max_tokens": args.max_tokens,
        "baselines": baselines,
        "summary": summary,
        "tasks": [r.__dict__ | {"attempts": [a.__dict__ for a in r.attempts]} for r in results],
    }
    (args.out_dir / f"live_routing_trace{suffix}.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out_dir / f'live_routing_trace{suffix}.json'}")


if __name__ == "__main__":
    main()
