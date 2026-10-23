# Validation on real models

This document answers two questions raised in review, using measurements from real models rather than the simulated world:

- **Q3.** Turn on a real model in the router, change the judge, and explain why the cheaper model gets escalated.
- **Q4.** Where did the numbers in the simulated world come from, and what would you measure instead?

Everything below was produced by the tools in [`backend/app/profiling`](../backend/app/profiling) and saved in [`data/`](../data). Commands to reproduce are at the end.

## 1. Setup

| Role | Model | List price per Mtok (in / out) | Why |
|---|---|---|---|
| Cheap worker | `groq-gpt-oss-20b` | $0.075 / $0.30 | Cheapest model that answered reliably |
| Strong worker | `groq-gpt-oss-120b` | $0.15 / $0.60 | Next tier up, same provider |
| Dedicated judge | `gemini-3.5-flash-lite` | $0.30 / $2.50 | Different provider and model family from both workers |

Everything ran on free tiers, so the real cost was $0. Costs below are what the calls would cost at list price. `gemini-3.8-flash` was the intended stronger model, but Google returned repeated `503 UNAVAILABLE: high demand` errors and a profile could not be completed (1 successful call), so it was replaced. Prices were taken from each provider's pricing page on 2026-10-22.

## 2. Q3: why the cheaper model gets escalated

### What had to change in the router first

Reproducing the scenario on the original code showed six gaps:

| Finding | Fix |
|---|---|
| At low risk only free format checks ran, so changing the judge changed nothing, and a plausible wrong answer from a cheap model was never escalated by a judge | A judge now runs whenever the chosen model is not proven for the task, at every risk level |
| With one real model and no second model, medium risk abstained ("no judge available") | A dedicated judge can be configured (`JUDGE_MODEL` or `judge_model` on `/run`); it is removed from the routing pool so it never grades its own answers |
| The live cascade priced verification at $0 | Verification is priced at what one judge call costs |
| Live mode could route real calls using simulated skill numbers | Live runs use only measured profiles (`live_capability_profiles.json`) |
| A missing API key surfaced as a provider error mid-run | `409` naming the missing variable before any call |
| Escalations were not explained | Each trace states which check failed, which judge said so and why, and what the rejected attempt cost |

### The decision chain, on real data

1. **Evidence.** Profiling put `gpt-oss-20b` at 19/20 on coding. With 20 samples the 5th-percentile lower bound is 0.79, just under the 0.80 floor for low risk, so no model is proven for coding.
2. **Route.** Nothing proven, low risk: the router starts with the cheapest eligible model (`20b`) and requires a judge, because an unproven answer has no evidence behind it. For the other four task types the lower bound was 0.86 to 0.87, so those were routed to `20b` as proven, with no judge. In a 40-task mixed run that was 32 tasks proven and 8 cheap-first.
3. **Verify.** A free format check runs first, then the judge. A pass accepts. A failure escalates to the next tier with the failure evidence attached. A judge that cannot answer abstains rather than guessing.
4. **Escalate.** The next model sees its predecessor's answer and the judge's reason.

### What actually happened, including where it went wrong

A 60-task coding run (600-token budget) escalated 6 tasks. The clearest exhibit, task `coding-0077`, `sum(i * i for i in range(171))`:

- `20b` answered `1652145`. This is correct (checked independently).
- The judge replied that the value is 1667635. It was **wrong**.
- The router escalated as designed. `120b` returned an empty reply and the call abstained.

So the cheaper model was escalated because the judge rejected it, and the judge was mistaken. The same 60 tasks rerun with a 1,500-token budget gave 59/60 correct, **0 escalations** and no truncated replies. The 600-token budget cut off some replies of these reasoning models, which probably explains most of the first run's escalations; per-attempt truncation was not recorded then, so this is consistent with the evidence rather than proven. The judge's mistake on `coding-0077` also did not repeat, which shows judge verdicts vary between runs.

### Changing the judge: measured head to head

Both judges saw the same 80 items (40 tasks, each with its correct answer and a plausible wrong one). The wrong answers are synthetic, so this is a controlled test, not a sample of any model's real mistakes.

| Judge | Catches wrong answers | Wrongly rejects right answers | Coding catch |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 29/40 (73%, 95% CI 57–84%) | 2/40 (5%, CI 1–17%) | 2/8 |
| `groq-gpt-oss-120b` | 39/39 (100%, CI 91–100%) | 0/40 (0%, CI 0–9%) | 7/7 |

(`120b` had one unusable verdict, excluded.) The weak judge is weakest exactly where it is needed: as a worker it scored 2/10 on coding. A judge cannot verify what it cannot solve.

### What it costs

Coding, per correct answer:

| Setup | Cost per success | Correct |
|---|---|---|
| `20b` alone, no judge | $0.000107 | 19/20 |
| `120b` alone, no judge | $0.000201 | 19/20 |
| Router with the `flash-lite` judge | $0.000248 | 59/60 |

The judge call is the most expensive part of the pipeline, so verifying every coding answer costs more than simply using the strong model. This is the sensitivity the main benchmark already predicted: a costly, weak judge can erase the saving. The router's cascade check handles part of this by skipping the cheap model when escalation would cost more, but it does not decide whether a judge is worth having at all.

### What I would do next

- Choose the judge by measured catch rate on labelled pairs, as done here, not by price. `gpt-oss-120b` is the better judge by a wide margin.
- Where output can be executed or checked exactly (code, arithmetic, schemas), use a deterministic check instead of an LLM judge. It is free and cannot be wrong in the way the judge was.
- A judge cannot also be a worker, so using the best judge means a third model for escalation, or accepting that the best judge sits outside the routing pool.

## 3. Q4: where the simulated numbers came from

**They were assumptions, not measurements.** In [`world.py`](../backend/app/benchmark/world.py) the skill of each simulated model per task type and difficulty (for example, the small model at 0.80 / 0.45 / 0.15 on easy / medium / hard coding), the share of wrong answers that are malformed, and latency were chosen by hand to give three plausible tiers. In [`verifiers.py`](../backend/app/benchmark/verifiers.py) the judge catches 85% of wrong answers and wrongly rejects 3%. They exist to test the machinery reproducibly and offline, and the README says so.

**How far assumptions can be from reality**, measured here (different models than the simulated ones, so this shows the size of the gap, not an error in any single number):

| Quantity | Simulated assumption | Measured |
|---|---|---|
| Cheap model on coding | "small" weak: 80% easy, 15% hard | `gpt-oss-20b` 19/20 (95%) overall; `gemini-3.5-flash-lite` 2/10 |
| Judge catch rate | 85% | 73% overall for `flash-lite` and only 2/8 on coding; 100% (CI 91–100%) for `120b` |
| Judge false-reject rate | 3% | 5% for `flash-lite`; 0% (CI 0–9%) for `120b` |
| Verification cost | A mid-tier call | Larger than the generation it checks |

**What I would measure instead, in production:**

1. Success rate per model, task type and difficulty on labelled tasks that look like production traffic, with intervals. A model earns trust only after enough clean samples: with 10 samples even a perfect score has a lower bound of 0.76, and about 13 clean samples are needed to clear 0.80.
2. Judge catch rate and false-reject rate on labelled right and wrong pairs, taken from real failures, per task type, and re-measured when the judge model changes.
3. Truncation and empty-reply rate per model at the token budget in use. This turned out to matter: reasoning models can spend the budget thinking and return nothing.
4. Latency percentiles, real token counts and billed cost, not list prices.
5. Drift: re-profile on a schedule and when a provider updates a model.

## 4. Limits of this evidence

- Small samples: 20 per task type per model, 40 to 60 traced tasks. Intervals are wide and shown.
- One synthetic task family with exact answers. It does not cover open-ended work.
- The judge test's wrong answers are synthetic. Real mistakes may be subtler.
- Free-tier availability shaped the setup: `gemini-3.8-flash` was unavailable, and free limits are unpublished or small for some models.
- Prices are public list prices; free-tier calls cost nothing. Some prices change on 2027-01-01.
- Judge verdicts vary between runs, and the 600-token and 1,500-token coding runs differ for that reason as well as the budget.
- The hosted demo stays simulated-only on purpose: it holds no API keys.

## 5. Reproduce

From the `backend` folder, with `GEMINI_API_KEY` and `GROQ_API_KEY` in `backend/.env`:

```
python -m app.profiling.cli --models groq-gpt-oss-20b groq-gpt-oss-120b --tasks 100 --yes
python -m app.profiling.trace_cli --workers groq-gpt-oss-20b groq-gpt-oss-120b --judge gemini-3.5-flash-lite --tasks 40 --yes
python -m app.profiling.trace_cli --workers groq-gpt-oss-20b groq-gpt-oss-120b --judge gemini-3.5-flash-lite --categories coding --tasks 60 --max-tokens 1500 --label coding_1500 --yes
python -m app.profiling.judge_cli --judges groq-gpt-oss-120b gemini-3.5-flash-lite --yes
```

Saved results, all under `data/`: `live_capability_profiles.json`, `live_profile_report.json`, `live_profile_runs.jsonl`, `live_routing_trace*.json`, `live_judge_eval.json`.
