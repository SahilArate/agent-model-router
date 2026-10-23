# Technical reference

How the system is built, how each part behaves, and how it was evaluated. For the reasoning behind the design, see [LOGIC.md](LOGIC.md). For an overview, see the [README](../README.md).

## Contents

1. [Stack and why](#1-stack-and-why)
2. [Module guide](#2-module-guide)
3. [Core algorithms](#3-core-algorithms)
4. [Execution model](#4-execution-model)
5. [API contract](#5-api-contract)
6. [Evaluation methodology](#6-evaluation-methodology)
7. [Testing](#7-testing)
8. [Configuration, security and privacy](#8-configuration-security-and-privacy)
9. [Failure behaviour](#9-failure-behaviour)
10. [Known limitations](#10-known-limitations)
11. [Extending the system](#11-extending-the-system)

---

## 1. Stack and why

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Strong typing support, `StrEnum`, mature async. The workload is I/O-bound (waiting on providers) so Python is not a bottleneck |
| API | FastAPI + uvicorn | Async handlers, automatic OpenAPI docs at `/docs`, tight Pydantic integration |
| Models and validation | Pydantic v2, pydantic-settings | Request/response contracts, typed settings, secrets held as `SecretStr` |
| Provider access | LiteLLM (imported lazily) | One call shape across OpenAI, Anthropic and Gemini. Lazy import keeps offline runs and tests free of provider dependencies |
| Statistics | Implemented in the repository (`beta.py`, `stats.py`) | No scientific stack required at runtime, and every number in the report can be audited in a few dozen lines |
| Tests | pytest, pytest-asyncio (auto mode) | Async executor and API tests without boilerplate |
| Lint and format | ruff (line length 100; rules E, F, I, UP, B) | One fast tool for lint, import order and formatting |
| Frontend | React 18, TypeScript (strict), Vite | Typed contract mirroring the API schemas |
| Graph view | React Flow (`@xyflow/react`) | Purpose-built node/edge rendering with selection |
| Charts | Plain SVG and CSS | The cost-quality chart and bars need no charting dependency |

Dependency versions in the frontend are pinned exactly in `package.json` and `package-lock.json` for reproducible installs.

## 2. Module guide

```
backend/app/
├── core/
│   ├── config.py        Settings (env-driven), cached get_settings(); API keys held as SecretStr
│   ├── logging.py       One named stdout handler, idempotent configuration
│   └── types.py         Tier, RiskLevel, AgentRole, TaskCategory
├── registry/
│   ├── models.py        ModelProfile: prices per million tokens, cached-input price, context window,
│   │                    capability flags, privacy approval, enabled flag. cost() computes a call's price
│   ├── models.json      The catalog. Three simulated models enabled, real models listed but disabled
│   ├── catalog.py       ModelCatalog: load, get, all(enabled_only), by_tier
│   ├── beta.py          beta_cdf and beta_ppf (continued fraction plus bisection)
│   └── capability.py    CapabilityRegistry: record outcomes, estimate lower bounds, save/load JSON
├── router/
│   ├── request.py       RoutingRequest: agent, task type, risk, token counts, constraints
│   ├── constraints.py   Stage A hard constraints with a reason for every rejection
│   ├── policy.py        RoutingPolicy: risk floors, expected output fraction, verify cost, escalations
│   ├── cascade.py       Expected-cost maths and the cascade-versus-direct test
│   ├── decision.py      RoutingDecision, Candidate, RoutingMode (the explanation object)
│   └── engine.py        ModelRouter: stages A, B, C
├── llm/
│   ├── types.py         ChatMessage, LLMResult (text, tokens, latency, cost)
│   ├── backends.py      MockBackend, LiteLLMBackend, DispatchBackend, token estimate
│   ├── client.py        LLMClient: timeout, error wrapping, cost computation
│   ├── errors.py        LLMError hierarchy
│   └── simulated.py     Deterministic simulated responder for demos and API tests
├── verification/
│   ├── types.py         CheckStatus (pass, fail, error), CheckResult, VerificationReport
│   ├── checks.py        SchemaCheck (JSON plus Pydantic), RuleCheck (non-empty, contains, regex, length)
│   ├── judge.py         pick_judge, JudgeCheck (LLM-as-judge, fails closed), estimated_judge_cost
│   ├── pipeline.py      Cheapest-first pipeline that stops at the first non-pass
│   └── outcome.py       Action (accept, escalate, abstain) and decide()
├── workflow/
│   ├── budget.py        Budget: cost cap and call cap
│   ├── spec.py          CallSpec and RunOptions
│   ├── trace.py         AttemptTrace, CallTrace, WorkflowResult
│   ├── executor.py      RoutedExecutor: the bounded route, generate, verify loop
│   └── runner.py        MultiAgentWorkflow: planner, workers, synthesizer
├── agents/              Prompt and verifier definitions for planner, worker, synthesizer
├── api/                 routes, schemas, serializers, services, application factory
├── benchmark/           tasks, grading, simulated world, verifiers, runner, statistics, CLI
└── profiling/           limiter, runner (live profiling), guarded client, trace, CLIs
```

Two rules keep the structure honest. Dependencies point downward: the router knows nothing about the API, and the registry knows nothing about the router. And only `RoutedExecutor` calls a model for the workflow, so a call cannot skip routing, verification or budgeting.

## 3. Core algorithms

### 3.1 Capability estimation

For each key `(model, agent role, task category)` the registry stores a success count `s` and failure count `f`. The posterior for the true success rate is `Beta(1 + s, 1 + f)` (a uniform prior). The router reads:

- `lower_bound`: the 5th percentile of that Beta distribution
- `mean`: `(1 + s) / (2 + s + f)`
- `sufficient_evidence`: `s + f ≥ 10`

`beta_cdf` uses the regularised incomplete beta function evaluated by a continued fraction. `beta_ppf` inverts it by bisection. Both are tested against reference values from SciPy (for example, 9 successes in 10 trials gives a lower bound of about 0.64, while 170 in 200 gives about 0.80).

A model is **qualified** for a call when it has sufficient evidence *and* `lower_bound ≥ floor(risk)`, where the floor is 0.80 (low), 0.90 (medium) or 0.95 (high). Because 0.95 is high for a lower bound, a model needs a long clean record to be proven for high-risk work. When none is, high-risk calls use the strongest eligible model with an independent judge.

### 3.2 Hard constraints (Stage A)

A model is rejected, with a recorded reason, when it is disabled, outside the provider allow-list, not approved for sensitive data while the call contains it, too small for `input_tokens + max_output_tokens`, missing tool or structured-output support when needed, or when its **worst-case** cost (no cache hits, full output allowance) exceeds the call's cap. If every model is rejected, the call abstains with all reasons, and no model is invoked.

### 3.3 Selection (Stage B)

```
qualified = [c for c in candidates if c.qualified]
if qualified:        pick min by (expected cost, -lower bound, id)            -> PROVEN
elif risk is LOW:    pick the cheapest eligible                               -> CHEAP_FIRST
else:                pick the strongest eligible                              -> STRONGEST
```

Expected cost uses the request's token estimate and `expected_output_fraction × max_output_tokens`.

### 3.4 Escalation path and cascade test (Stage C)

The escalation path holds one model per higher tier, in tier order, the cheapest within each tier, capped by `max_escalations` (default 2). For `CHEAP_FIRST`, the router then tests whether starting cheap is worth it:

```
cascade = c_cheap + c_verify + P(escalate) · (c_strong + c_verify)
direct  = c_strong + c_verify
```

with `P(escalate) = 1 − mean` for the cheap model. If `cascade ≥ direct`, the cheap model is skipped and the decision becomes `DIRECT_UPGRADE`, with the reason recorded.

### 3.5 Verification and outcome

The pipeline runs checks in order and stops at the first result that is not `pass`. Statuses are `pass`, `fail` (the output is bad) and `error` (the check itself could not run). `decide()` then returns:

| Report | Action |
|---|---|
| All pass | Accept |
| Contains an `error` | Abstain, because the output is unverified rather than known bad |
| Fail and a fallback remains | Escalate to the next model, carrying the failure evidence |
| Fail and no fallback | Abstain |

The judge is chosen by `pick_judge`. A configured judge (the `JUDGE_MODEL` setting or `judge_model` on the request) wins when it is enabled and is not the worker; otherwise the cheapest enabled model from a different provider is used, and a same-provider judge is allowed but flagged as weaker. It is never the worker itself. A configured judge is also removed from the routing pool, so it never writes answers. Each judge verdict carries the judge's id and whether it was independent, and these appear in the trace.

The verifier factory builds two pipelines per call: deterministic checks only, and deterministic checks plus the judge. The executor picks the judged one when the router sets `requires_independent_verifier`, which is true whenever risk is above low **or** the chosen model is not proven (mode other than PROVEN). Before this rule, a low-risk call to an unproven model ran only free checks, so a plausible wrong answer was accepted and a configured judge had no effect. Now the cascade is priced with what one judge call really costs (`estimated_judge_cost`), not zero.

## 4. Execution model

### 4.1 The executor loop

```
route(request) -> decision                      (abstain if no eligible model)
loop:
    if budget exhausted:                  abstain
    result = client.complete(model, messages)
        on provider error: take next fallback, or abstain if none
    charge budget for generation
    report = pipeline.verify(...)
    charge budget for verification
    outcome = decide(report, fallbacks)
    accept | abstain | escalate (rebuild messages with the failed answer and the evidence)
```

Termination is guaranteed: every pass consumes one model from a finite fallback list, and the budget is checked before every call. A single call can overshoot the cost cap by its own cost, which is documented rather than hidden.

### 4.2 The workflow

1. The planner returns JSON steps, validated by a schema check. A failure escalates like any other call.
2. Workers run one per step, sequentially. Each receives the task, its step, and earlier results **compressed** to 400 characters each, so context does not grow with every hop.
3. The synthesizer assembles the final answer.
4. If any agent abstains, the workflow stops and reports where. It never builds on an unverified result.

Every call is recorded as a `CallTrace` (decision, attempts, checks, costs, final action), and the whole run as a `WorkflowResult`. These traces feed the API and dashboard unchanged.

### 4.3 Cost accounting

`LLMClient` computes cost from measured token usage and the model's price, accounting for cached input tokens. Verification cost is charged to the same budget. Every strategy in the benchmark is costed by the same code path, so comparisons are like for like.

## 5. API contract

`POST /run`

```json
{
  "task": "Summarise the key obligations in a 12-page supplier contract.",
  "mode": "mock",
  "risk": "medium",
  "category": "summarization",
  "contains_sensitive_data": false,
  "max_cost_usd": 0.5,
  "max_calls": 20,
  "max_steps": 3
}
```

Constraints: `task` 1 to 4000 characters, `max_cost_usd` in (0, 10], `max_calls` 1 to 50, `max_steps` 1 to 6. `mode` is `mock` (simulated models, free) or `live` (real providers). `live` returns **409** when no live model is enabled, with a message saying why.

The response contains `completed`, `final_answer`, `abstained_at`, total cost and latency, the budget used, and `traces`. Each trace has the agent, task type, risk, final action, reason, output, and:

- `routing`: chosen model, mode (`proven`, `cheap_first`, `direct_upgrade`, `strongest`), written reason, quality floor, fallback list, whether an independent verifier was required, **every candidate** (tier, expected cost, mean, lower bound, sample size, qualified) and the models rejected by hard constraints with reasons
- `attempts`: per attempt model, tokens, latency, generation and verification cost, action, error, and each check's name, status, detail and cost

Other endpoints: `GET /health`, `GET /models` (including disabled models), `GET /benchmark` (404 with an instruction if no run exists). Interactive documentation is generated at `/docs`. CORS allows the local dev origins `http://localhost:5173` and `http://127.0.0.1:5173`.

The TypeScript types in `frontend/src/types.ts` mirror `backend/app/api/schemas.py`.

## 6. Evaluation methodology

The central principle is *measure first, claim second*. The benchmark is designed so that it can show the router losing.

### 6.1 Tasks

1,000 tasks generated deterministically from a seed, spread evenly across five categories and three difficulties, each with an exact expected answer, so no judge is needed to grade:

| Category | Task | Difficulty control |
|---|---|---|
| Extraction | Read the total due from an invoice | Number of distractor amounts |
| Reasoning | Multi-step arithmetic story | Number of steps (2, 3, 5) |
| Coding | Evaluate a Python expression | Expression complexity |
| Summarization | Find the most frequent word | Margin between top counts |
| Tool use | Emit a correct JSON tool call | Number and type of arguments |

Each task also carries a risk level (70% low, 20% medium, 10% high) and a list of *plausible wrong answers*, which a simulated model can return to model confident mistakes. Prompts are unique, and grading is exact (numeric tolerance 0.005, case-insensitive word, structural JSON equality).

### 6.2 Simulated world

Each model answers correctly with a probability that depends on category and difficulty (for example, the small model is 98% on easy extraction and 15% on hard coding, the frontier model never below 90%). When wrong, the answer is either malformed (cheap checks catch it) or well-formed but incorrect (only a judge could catch it), in model-specific proportions. Randomness is a hash of seed, model, task and conversation length, so:

- runs are exactly reproducible,
- an escalated retry (longer conversation) gets a fresh draw,
- two strategies that call the same model on the same task see the same draw, which makes comparisons properly paired.

The judge is also simulated: it catches a wrong answer with probability 0.85 and wrongly rejects a correct one with probability 0.03, at the price of a mid-tier call. These are stated assumptions, and the CLI sweeps them (section 6.5).

Latency is computed from a simple per-model table (time to first token plus tokens per second). It is illustrative and not measured.

### 6.3 Strategies compared

| Strategy | Definition |
|---|---|
| `fixed_cheap` | Every call to the cheapest model, no verification |
| `fixed_strongest` | Every call to the strongest model, no verification |
| `complexity_only` | Route by prompt length (terciles learned on train) to small, mid or strongest, no verification. A proxy for complexity-based routers |
| `router_cold_start` | The full router with an empty registry: cautious cascade with verification |
| `router_learned` | The full router with profiles learned on the train split |

The router strategies run through the real `RoutedExecutor`, with the verification pipeline applied to every call (stricter than the live workflow, which uses deterministic checks only at low risk).

### 6.4 Protocol and statistics

1. **Split.** 40% train, 20% validation, 40% test, stratified by category and difficulty.
2. **Profile.** Every model answers each train task once; ground-truth correctness (without any judge) feeds the registry. Profiling uses a different random seed from evaluation.
3. **Tune.** The low-risk floor is chosen from `{0.6, 0.7, 0.8, 0.9}` on the **validation** split only: the cheapest cost per success among floors that reach a 0.93 success target. The result was 0.6.
4. **Evaluate.** Every strategy runs once on the **held-out test** split (415 tasks), 5 repeats with different random draws.

Reported metrics: success rate (accepted and correct) with a Wilson 95% interval, cost per successful task, mean latency, escalation rate, abstain rate, and **false-acceptance rate** (accepted answers that were wrong).

Significance, always paired on the same tasks:

- **Exact McNemar test** on the discordant pairs, using repeat 0 only so each task is one independent pair.
- **Paired bootstrap** (2,000 resamples) over *tasks*, keeping all repeats of a task together, for confidence intervals on the success difference and the cost-per-success difference.

### 6.5 Sensitivity analysis

Because the judge dominates the cost of verification, the CLI reruns the benchmark under four judge settings (mid-tier judge catching 85% and 60%, cheap judge catching 70% and 50%) and reports success, false acceptance and saving against the strongest model for each. The saving ranged from −3.2% to +48.7%, which is why the judge is treated as a measured component rather than an assumed one.

### 6.6 Reproducing

```
cd backend
python -m app.benchmark.cli                 # about 45 seconds
python -m app.benchmark.cli --tasks 400 --repeats 3 --no-sensitivity   # quick run
```

Outputs are written to `data/benchmark_results.json` and `data/capability_profiles.json` at the repository root. The API loads them at start-up: the simulated profiles route simulated runs, and `GET /benchmark` serves the results to the dashboard.

### 6.6 Measuring real models

The `app/profiling` package replaces assumptions with measurements for real models.

- **Profiler** (`python -m app.profiling.cli`). Generates the benchmark tasks, sends each to every chosen model, normalises the reply (reasoning tags, code fences and whitespace only; extra words still count as wrong), and grades it against the known answer. Results become worker-role Beta profiles in `data/live_capability_profiles.json`, plus a report with per-category and per-difficulty rates and Wilson intervals. Each call is appended to a JSON-lines log, so an interrupted run resumes and the same task is never measured twice.
- **Safety on free tiers.** A per-provider sliding window limits requests and tokens per minute, with defaults below published limits. A guard caps notional spend and call count. Transient errors (rate limits, overload) are retried with backoff and are **not** recorded as model failures; a rejected key or unknown model id stops that model at once with the provider's message. A dry run prints the estimated cost, calls and time and checks which keys are missing.
- **Trace** (`python -m app.profiling.trace_cli`). Runs the real router, verifier and a dedicated judge on tasks the profiler did not use. A recording client keeps every reply, so each attempt, including ones the judge rejected, is graded against ground truth. The summary reports success, escalation rate, wrong-but-accepted answers, and the judge's measured catch rate and false-reject rate with intervals. This is the evidence for how good the judge actually is.
- **Live registry.** Live-mode runs on the API use `live_capability_profiles.json` only. Simulated skill numbers never route real calls.

- **Judge comparison** (`python -m app.profiling.judge_cli`). Gives each judge the same items (every task with its correct answer and one plausible wrong answer) and reports catch rate and false-reject rate with Wilson intervals, per task type. No judge grades its own answer.
- **Trace options.** `--categories` focuses a run on some task types, `--label` keeps output files apart, `--max-tokens` sets the reply budget (reasoning models can use it all and return nothing; truncated replies are flagged per attempt), and the output prints each worker's no-judge cost per success from the profiling run as a baseline.

Limits of the measurement: one task generator with exact-answer tasks, profiles per category and worker role only, results valid for the date and model versions measured.

## 7. Testing

More than 160 automated tests run in a few seconds with no network access.

| Area | What is checked |
|---|---|
| Statistics | Beta CDF and quantiles against reference values; Wilson interval; McNemar exact values; bootstrap detects a clear winner |
| Registry | Counting, lower bound ordering (a lucky small sample does not beat solid evidence), persistence round trip |
| Constraints and router | Each hard constraint and its message; each routing mode; escalation path; cascade skipped when it would lose; determinism |
| Cascade maths | Expected cost formula and break-even |
| LLM client | Token and cost computation, cached-token clamping, timeout, error wrapping, provider dispatch |
| Verification | Schema, rule checks, judge pass and fail with cost, fenced JSON, fail-closed on unparseable or failed judge, pipeline short-circuit |
| Executor | Accept, escalate with evidence, bounded loop, provider failure fallback, budget exhaustion, unverifiable output abstains, no eligible model |
| Workflow | Happy path with cost accounting, planner failure then recovery, abstention stops the run, budget stops runaway calls, risk-dependent judge, context compression |
| API | Health, models, run, validation errors, live-mode 409, benchmark endpoint with and without data |
| Profiling | Sliding-window limiter, answer normalisation, resume without new calls, transient errors retried and not counted as failures, fatal errors told apart from rate limits, call and cost caps, truncation flagged |
| Live trace | Judge sends a wrong cheap answer back and the stronger model fixes it, free check catches malformed answers, a blind judge lets errors through, judge confusion counts, resume, held-out tasks only |
| Benchmark | Deterministic tasks, expected answers grade correct and wrong ones do not, disjoint complete splits, reproducible runs, sensible ordering of strategies, profiles learned from train only, JSON-safe output |

Run with `pytest -q` from `backend/`. Static checks: `ruff check .` and, for the frontend, `npm run typecheck`.

## 8. Configuration, security and privacy

- Settings are read from the environment or `backend/.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `JUDGE_MODEL`, `CORS_ORIGINS`, `LOG_LEVEL`, `ENVIRONMENT`. Keys are `SecretStr` and never logged or returned by the API (`/health` lists only provider *names*).
- Sensitive data is a first-class routing input. When a call is flagged, only models marked `approved_for_sensitive_data` are eligible, enforced in Stage A before any cost reasoning.
- The `.env` file and key material must stay out of version control.
- The API currently has no authentication and is intended to run locally or behind a gateway. CORS is limited to the local development origins.
- The `/run` endpoint bounds work per request (cost, calls, steps, input length) to prevent runaway spend.

## 9. Failure behaviour

| Situation | Behaviour |
|---|---|
| No model satisfies the hard constraints | The call abstains with every rejection reason; no model is invoked |
| Provider error or timeout | Falls back to the next model in the path; abstains if none remains |
| Output fails verification | Escalates with the failure evidence; abstains when nothing stronger remains |
| Judge unavailable or unparseable | Treated as unverified and the call abstains; never counted as a pass |
| Budget reached | The next call is not made and the workflow abstains with that reason |
| An agent abstains | The workflow stops at that agent and reports it |
| Live mode with no enabled real model | `409` with instructions |
| Live mode with an enabled model whose provider key is missing | `409` naming the variable to set; nothing is called |
| Requested judge unknown or not usable in that mode | `422` listing the available models |
| Provider rate limit or overload during profiling | Retried with backoff; not counted as a model failure; the model stops after repeated consecutive errors and the run can resume |
| Benchmark results missing | `404` with the command to create them |
| New or rarely seen task type | Counts as unproven; low risk starts cheap with verification, higher risk uses the strongest model |

## 10. Known limitations

- **Simulated models and judge.** Skill tables and judge accuracy are assumptions. Absolute numbers should not be read as predictions for real models.
- **Profiles keyed by category, not difficulty.** Averaging within a category blurs evidence; a difficulty signal would sharpen routing.
- **Only the worker role is profiled in the benchmark.** Planner and synthesizer calls therefore start cheap at low risk (verified by schema or rules) or go to the strongest model at higher risk. The verifier role is chosen by `pick_judge` and is not yet routed through the registry.
- **Judge errors are independent in the simulation.** Real judges share blind spots with the models they grade, so real performance of the verification layer may be worse.
- **Cache effects are not modelled in expected-cost estimates.** The router assumes a cache miss, which is conservative.
- **Workers run sequentially.** Independent steps could run in parallel to cut latency.
- **The static task generator is one distribution.** Results across other distributions are not claimed.
- **No authentication on the API.**
- **Real-model entries are disabled by default** and their prices come from public price listings dated in `pricing_verified_on`; confirm them before relying on cost numbers. Prices that change with prompt size (a higher rate above a token threshold) are not modelled.
- **Only the worker role is measured live.** Planner and synthesizer calls on real models therefore remain unproven and are judged.
- **Free-tier limits are not published precisely** for every provider, so default throttles are conservative and may be adjusted with `--rpm` and `--tpm`.

## 11. Extending the system

**Add a real model.** Add an entry to `models.json` with verified prices, context window, capability flags and privacy approval; set `enabled` to true; add the provider key; run profiling on your own labelled tasks so the registry has evidence for it.

**Add a task category.** Add a value to `TaskCategory`, a task generator and a grading rule in the benchmark, and (for the simulated world) a skill row per model. No router change is needed: the registry key space is open.

**Measure real models.** Use `app.profiling.cli` to profile them and `app.profiling.trace_cli` to trace the router and judge on held-out tasks (section 6.6). To run the full strategy comparison on real models, `SimulatedWorld` is a responder with the signature `(model, messages) -> str`: replace it with a backend that calls real models and replace `NoisyJudge` with `JudgeCheck`. The strategies, split, statistics and reporting are unchanged.

**Change the policy.** Quality floors, the escalation limit, expected output fraction and verification cost are fields of `RoutingPolicy`.

**Use online learning.** The registry already holds Beta posteriors. Thompson sampling over them, restricted to low-risk traffic, is the natural next step, with the existing lower-bound rule kept for medium and high risk.
