"""Measure real models and write capability profiles.

    python -m app.profiling.cli --models gemini-3.5-flash-lite gemini-3.8-flash --dry-run
    python -m app.profiling.cli --models gemini-3.5-flash-lite gemini-3.8-flash

Run from the backend folder. API keys are read from backend/.env (GEMINI_API_KEY,
GROQ_API_KEY, ...). Output goes to data/live_capability_profiles.json and
data/live_profile_report.json; the simulated profiles are never touched.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.benchmark.tasks import generate_tasks
from app.core.config import API_KEY_ENV, Settings
from app.llm.client import build_default_client
from app.profiling.limiter import Limits, limits_for
from app.profiling.runner import (
    ProfileConfig,
    RunLog,
    build_registry,
    estimate_run,
    profile_models,
    select_records,
    summarise,
)
from app.registry.catalog import ModelCatalog
from app.registry.models import ModelProfile

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "data"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def resolve_models(catalog: ModelCatalog, ids: list[str]) -> list[ModelProfile]:
    known = {m.id: m for m in catalog.all(enabled_only=False)}
    unknown = [i for i in ids if i not in known]
    if unknown:
        fail(f"not in the catalog: {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    models = [known[i] for i in dict.fromkeys(ids)]
    if any(m.provider.value == "mock" for m in models):
        fail("mock models have nothing to measure; pick real models")
    return models


def missing_keys(models: list[ModelProfile], settings: Settings) -> list[str]:
    configured = set(settings.configured_providers)
    return sorted(
        {API_KEY_ENV[m.provider.value] for m in models if m.provider.value not in configured}
    )


def limit_overrides(
    models: list[ModelProfile], rpm: int | None, tpm: int | None
) -> dict[str, Limits] | None:
    """Replace the default limits with --rpm/--tpm, keeping the default for whichever is unset."""
    if rpm is None and tpm is None:
        return None
    result = {}
    for model in models:
        base = limits_for(model.provider.value)
        result[model.provider.value] = Limits(rpm=rpm or base.rpm, tpm=tpm or base.tpm)
    return result


def format_summary(summary: dict) -> str:
    lines = []
    for model_id, data in summary.items():
        o = data["overall"]
        lo, hi = o["ci95"]
        lines.append(
            f"\n{model_id}: {o['successes']}/{o['n']} correct ({o['success_rate']:.1%}, "
            f"95% CI {lo:.1%}-{hi:.1%}), mean latency {o['mean_latency_s']:.2f}s, "
            f"notional ${o['mean_cost_usd'] * o['n']:.4f}, truncated replies {o['truncated']}"
        )
        lines.append(f"  {'category':<15}{'n':>4}{'correct':>9}{'lower bound':>13}  evidence")
        for category, cell in data["by_category"].items():
            enough = "enough" if cell["sufficient_evidence"] else "too few samples"
            lines.append(
                f"  {category:<15}{cell['n']:>4}{cell['success_rate']:>9.0%}"
                f"{cell['lower_bound']:>13.2f}  {enough}"
            )
        by_diff = "  ".join(
            f"{d}: {c['successes']}/{c['n']}" for d, c in data["by_difficulty"].items()
        )
        lines.append(f"  by difficulty  {by_diff}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure real models into capability profiles.")
    parser.add_argument("--models", nargs="+", required=True, help="catalog model ids")
    parser.add_argument("--tasks", type=int, default=50, help="tasks per model (10 per category)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-cost", type=float, default=0.50, help="notional USD cap")
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--rpm", type=int, help="override requests per minute for every provider")
    parser.add_argument("--tpm", type=int, help="override tokens per minute for every provider")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="estimate only, call nothing")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    settings = Settings()
    models = resolve_models(ModelCatalog.from_file(), args.models)
    tasks = generate_tasks(args.tasks, args.seed)
    log = RunLog(args.out_dir / "live_profile_runs.jsonl")
    done = {r.key for r in log.load()}

    limits = limit_overrides(models, args.rpm, args.tpm)
    estimate = estimate_run(models, tasks, max_tokens=args.max_tokens, done=done, limits=limits)
    print(f"Tasks per model: {len(tasks)} (seed {args.seed}); already measured are skipped")
    for model_id, calls in estimate.calls.items():
        print(f"  {model_id}: {calls} calls to make")
    print(
        f"Estimated notional cost: about ${estimate.typical_cost_usd:.4f}, "
        f"worst case ${estimate.worst_cost_usd:.4f} (cap ${args.max_cost:.2f}); "
        f"calls {estimate.total_calls} (cap {args.max_calls}); "
        f"about {estimate.minutes:.0f} min at the throttled rate"
    )
    print("Free tiers cost $0 in practice; the figure is what the calls would cost at list price.")
    if estimate.worst_cost_usd > args.max_cost or estimate.total_calls > args.max_calls:
        print(
            "Note: the caps are below the worst case, so the run may stop early and can be resumed."
        )

    missing = missing_keys(models, settings)
    if args.dry_run:
        if missing:
            print(f"Missing before a real run: {', '.join(missing)} (put them in backend/.env)")
        return
    if missing:
        fail(f"missing API key(s): {', '.join(missing)}. Put them in backend/.env and retry.")
    if estimate.total_calls and not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("cancelled")

    config = ProfileConfig(
        max_tokens=args.max_tokens, max_cost_usd=args.max_cost, max_calls=args.max_calls
    )
    client = build_default_client(settings, timeout_s=120.0)
    run = asyncio.run(profile_models(models, tasks, client, log, config, limits=limits))

    for status in run.statuses.values():
        if status.aborted:
            print(f"! {status.model_id} stopped early: {status.aborted}")
        if status.errors:
            print(f"  {status.model_id}: {status.errors} calls failed and were not counted")

    records = select_records(run.records, tasks)
    registry = build_registry(records)
    summary = summarise(records, registry)
    print(format_summary(summary))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry.save(args.out_dir / "live_capability_profiles.json")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "tasks_per_model": len(tasks),
        "max_tokens": args.max_tokens,
        "models": {
            m.id: {
                "litellm_model": m.litellm_model,
                "input_price_per_mtok": m.input_price_per_mtok,
                "output_price_per_mtok": m.output_price_per_mtok,
                "pricing_verified_on": str(m.pricing_verified_on),
            }
            for m in models
        },
        "notional_cost_usd": run.guard.spent_usd,
        "summary": summary,
    }
    (args.out_dir / "live_profile_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote live profiles and report to {args.out_dir}")


if __name__ == "__main__":
    main()
