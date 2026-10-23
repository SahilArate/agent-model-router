"""Compare judges on identical items with known right and wrong answers.

    python -m app.profiling.judge_cli --judges groq-gpt-oss-120b gemini-3.5-flash-lite

Run from the backend folder. Writes data/live_judge_eval.json.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.benchmark.tasks import generate_tasks
from app.core.config import Settings
from app.llm.client import build_default_client
from app.profiling.cli import DEFAULT_OUT_DIR, fail, limit_overrides, missing_keys, resolve_models
from app.profiling.guarded import GuardedClient
from app.profiling.judge_eval import (
    build_items,
    evaluate_judge,
    format_summary,
    summarise_judges,
    to_json,
)
from app.profiling.runner import Guard
from app.registry.catalog import ModelCatalog
from app.verification.judge import estimated_judge_cost


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure how well judges tell right from wrong.")
    parser.add_argument("--judges", nargs="+", required=True, help="catalog model ids")
    parser.add_argument("--tasks", type=int, default=40, help="tasks; each gives 2 items")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-cost", type=float, default=0.50, help="notional USD cap")
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--rpm", type=int)
    parser.add_argument("--tpm", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    judges = [
        m.model_copy(update={"enabled": True})
        for m in resolve_models(ModelCatalog.from_file(), args.judges)
    ]
    items = build_items(generate_tasks(args.tasks, args.seed))
    calls = len(items) * len(judges)
    notional = sum(estimated_judge_cost(j) for j in judges) * len(items)
    print(f"{len(items)} items ({args.tasks} tasks, each right and wrong) per judge; {calls} calls")
    print(f"Rough notional cost ${notional:.4f}; throttled to stay inside free-tier limits.")

    missing = missing_keys(judges, settings)
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
        limits=limit_overrides(judges, args.rpm, args.tpm),
    )

    async def run_all():
        return await asyncio.gather(
            *(evaluate_judge(j, items, client, stop_reason=lambda: guard.reason) for j in judges)
        )

    verdicts = [v for batch in asyncio.run(run_all()) for v in batch]
    summary = summarise_judges(verdicts)
    print(format_summary(summary))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "live_judge_eval.json"
    path.write_text(to_json(verdicts, summary))
    print(f"\nWrote {path}", file=sys.stdout)


if __name__ == "__main__":
    main()
