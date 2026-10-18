# Logic: the scenario, the approach, and why this one

This document explains the thinking behind the system. It starts from a concrete situation, shows what goes wrong with the obvious solutions, builds the approach from that, and then compares it honestly with the alternatives that were considered.

Technical detail (modules, algorithms, API, tests) is in [TECHNICAL.md](TECHNICAL.md).

---

## 1. The scenario

Take a support-ticket assistant built from cooperating agents. A customer writes in, and the application must produce a correct reply.

| Agent | Job | What a mistake looks like |
|---|---|---|
| Planner | Break the ticket into steps and return them as JSON | Not valid JSON, so the whole run stops, or a poor plan that misleads every later agent |
| Worker 1 | Pull the amount and invoice number out of the message | A wrong number that flows into the reply |
| Worker 2 | Summarise the customer's history | A summary that drops the important complaint |
| Worker 3 | Check the request against refund policy | A confident wrong policy conclusion |
| Synthesizer | Write the final reply from the workers' results | Fluent text built on a bad input |

Each ticket triggers five or more LLM calls. The application handles thousands of tickets a day, and a small share involve money, so not every ticket carries the same risk.

The team needs to decide which model each agent calls. There are three obvious answers, and each one fails in a way that teaches something.

### Option 1: everything on the strongest model

Quality is high and nothing needs deciding. In the benchmark this project ships, the strongest model succeeds 97.5% of the time at $0.000771 per successful task. That is the quality ceiling and the cost ceiling at once.

The waste is easy to see. Pulling an amount out of a short invoice is a task a small model handles almost perfectly (98% on easy extraction in the simulated world, and in the benchmark the router answers most extraction and summarization calls with the small model). Paying the frontier price for it is paying for capability that is not used. Across thousands of tickets, most calls are of this kind.

### Option 2: everything on the cheapest model

Cost per successful task is about 60 times lower. But the cheap model succeeds only 68% of the time on the mixed workload, and the 32% it gets wrong are returned as confident answers. Nothing flags them. "Cost per successful task" looks excellent while the real cost, wrong replies reaching customers, is invisible to the metric.

### Option 3: a person picks a model per agent

A reasonable engineer says: small model for extraction, mid for the policy check, strongest for the final reply. This is better than either extreme, and it is where most teams stop. It has four weaknesses:

1. It is a guess. Nobody measured whether the small model is good enough *on these tickets*.
2. It cannot adapt to the instance. An easy extraction and a nightmare extraction get the same model.
3. It goes stale. A new model appears, a price drops, an existing model is updated, and the table is wrong without anyone noticing.
4. It ignores risk. A refund of $5,000 and a question about opening hours are treated the same.

### What the scenario asks for

Put the three failures together and the requirement becomes clear. The system must pick a model **per call**, from **evidence about this workload**, take **risk** into account, **check** its own output, and **never spend without a limit**. That is the approach.

---

## 2. The approach, built up step by step

### Step 1: state the goal precisely

"Best output at least cost" is two goals. The way to make it solvable is to make one a constraint:

```
minimise    expected cost of the workflow
subject to  P(output is correct)  ≥  floor(risk)
            plus hard rules (privacy, context size, tools, provider list, budget)
```

Quality is a floor that depends on risk, and cost is minimised above it. A refund question gets a higher floor than a question about opening hours.

### Step 2: remove models that are not allowed, before any optimising

Some conditions are not trade-offs. If the ticket contains personal data and a model is not approved for it, that model is out, however cheap. If the prompt does not fit the context window, the model is out. The same goes for missing tool support, a provider outside the allow-list, or a worst-case cost above the call's cap. This is Stage A. Optimisation never gets to override it, and every rejection is recorded with its reason.

### Step 3: measure which model is good enough, and be careful about what "measured" means

For every combination of model, agent role and task category the system keeps a count of successes and failures and treats the true success rate as a Beta distribution.

The routing decision does not use the raw rate. It uses the **5th percentile** (the lower bound). The reason is the difference between these two models:

| Model | Result | Raw rate | Lower bound |
|---|---|---|---|
| A | 9 correct out of 10 | 90% | about 64% |
| B | 170 correct out of 200 | 85% | about 80% |

A looks better on the raw rate and is actually the weaker bet, because 10 samples prove very little. Using the lower bound makes the router prefer evidence over luck, and it means a new model must *earn* its way into routing by accumulating a track record. Until a model has at least 10 samples in a cell, the system treats it as unproven.

### Step 4: choose the cheapest model that clears the floor

Among the allowed models, take those whose lower bound clears the floor for this call's risk (0.80 low, 0.90 medium, 0.95 high), and pick the cheapest. This is the **proven** path: the choice is backed by data.

### Step 5: decide what to do when nothing is proven

At the start of a deployment, or for a new kind of task, there is no evidence. This is where risk changes the strategy:

- **Low risk:** start with the cheapest model and rely on verification. A failure is caught and escalated, and the failure itself becomes evidence.
- **Medium or high risk:** go straight to the strongest model and add an independent judge. A wrong answer here costs more than the saving, so the router does not experiment on it.

Cheap-first is not applied blindly either. Trying the cheap model first costs

```
E[cost] = c_cheap + c_verify + P(escalate) · (c_strong + c_verify)
```

whereas going directly costs `c_strong + c_verify`. With illustrative prices of $0.000027 for the small model, $0.0027 for the strongest and $0.0005 for a judge call, trying the small model first pays off unless it fails more than about 83% of the time. For a mid-tier first attempt priced at $0.00054 the break-even is about 68%. The router computes this with the measured failure rate and skips the cheap step when it would lose.

### Step 6: verify the output, cheapest check first

An answer is not trusted because a model produced it. It goes through a pipeline ordered by cost:

1. **Deterministic checks (free).** Is it non-empty, valid JSON, matching the expected schema, within length, containing required items? These cannot be fooled by fluent nonsense and cost nothing.
2. **An independent judge (paid).** Only runs if the free checks passed, so it is never paid for on obviously broken output. The judge is picked from a different provider than the worker where possible, because models from the same family tend to share blind spots.

The pipeline stops at the first failure. A judge that cannot run, or answers in an unparseable way, counts as **unverified**, never as a pass. The system fails closed.

### Step 7: act on the verdict

| Verification result | Action |
|---|---|
| All checks pass | Accept |
| A check fails and a stronger model remains | Escalate, passing the failure evidence to the stronger model |
| A check could not run | Abstain (the output is unverified, a more expensive model would not fix that) |
| Failed and nothing stronger remains | Abstain and hand to a human |

Abstaining is a feature. A system that says "I could not verify this" is more useful than one that returns a confident guess.

### Step 8: make the whole thing bounded

A workflow carries a **budget** (cost and call count) that is checked before every model call. The escalation path is finite (at most two escalations after the first model). If a provider call fails, the executor falls back to the next model rather than crashing. If an agent abstains, the workflow stops instead of building on an unverified result. Between agents, earlier results are compressed so context does not grow with every hop.

### Step 9: close the loop with measurement

The registry is only as good as its data, so the project includes the harness that produces it. Every model answers a training set, successes and failures are recorded against ground truth, and the router is then compared with simpler strategies on tasks it has never seen, with statistical tests. The benchmark is what turns "we think this routing is good" into a number with an interval around it.

---

## 3. Why this approach, compared with the alternatives

Eight other designs were considered. Each is reasonable in some setting, and each fails the scenario above in a specific way.

| # | Alternative | What it does well | Why it was not chosen |
|---|---|---|---|
| 1 | **One strong model for everything** | Simplest. Highest quality with no decisions | Pays frontier prices for easy calls. No cost control. Nothing to defend as "least cost" |
| 2 | **One cheap model for everything** | Lowest bill | 32% of answers wrong and unflagged in the benchmark. Fails the quality half of the question |
| 3 | **Hand-written per-agent assignment** | Quick, transparent, no infrastructure | Based on intuition, never re-checked, blind to instance difficulty and to risk. Becomes wrong silently |
| 4 | **Prompt-complexity or learned preference router** (the family behind RouteLLM-style routers) | Very cheap at runtime. Works without running the model | Trained on general chat preferences, not on agent tasks with exact correctness. Recent work reports these routers are brittle under small input changes. In the benchmark, a prompt-length router reached 84.9% success against 97.1% for this design, because difficulty depended on the task type more than on length. (That baseline is a simple length heuristic, a proxy for the idea rather than a trained router.) |
| 5 | **Ask an LLM to choose the model** | Flexible, can read the task | Adds a model call, latency and cost to *every* call, is non-deterministic, and its choices are no better measured than a person's |
| 6 | **Pure cascade** (always start with the cheapest, as in FrugalGPT) | Strong theory, good savings on easy workloads | Starting point is fixed, so it wastes money whenever the cheap model usually fails, and it has no notion of risk. This design keeps the cascade as a *tool* but chooses the starting point from evidence and the risk level, and tests whether the cascade pays off |
| 7 | **Gateway or proxy fallbacks** (LiteLLM, OpenRouter and similar) | Excellent for availability, unified API, cost tracking | They decide what to do when a provider is down, not which model is *good enough for this task*. They are used here as the transport layer under the router, not as a replacement for it |
| 8 | **Online bandit that learns on live traffic** | Adapts continuously | Exploration means real users sometimes get an experimental model, which is unacceptable for high-risk calls. It also needs a reliable live quality signal. The Beta statistics in this design are exactly what Thompson sampling would use, so it remains a natural extension for low-risk traffic |

### The reasons in short

The chosen design is the one that satisfies all of the following at once, which none of the alternatives does:

- **Measured.** Decisions come from success rates on this workload, with the uncertainty included.
- **Conservative.** Lower bounds, minimum sample sizes, and a model must earn trust.
- **Risk-aware.** High-stakes calls never experiment.
- **Cost-aware in a calculated way.** Cheap-first is used only where its expected cost is lower.
- **Self-checking.** Cheap models are only safe because their output is verified.
- **Bounded.** Cost, calls and escalations all have hard limits.
- **Explainable.** Every decision carries a written reason, the candidates considered and their evidence, so it can be audited and shown in the dashboard.
- **Testable.** It is built from pure decision functions and has a benchmark with significance tests, so claims about it can be checked.

---

## 4. What the evidence shows

On 415 held-out tasks repeated 5 times, the router with learned profiles matched the strongest model on success (97.1% against 97.5%, difference not significant, p = 0.58), cut cost per success by about 8%, and reduced wrong-but-accepted answers from 32% (cheap model) to about 1%. The same run showed the weakness plainly: verification adds a cost to every call, so a pricey judge shrinks or even removes the saving. With a cheap judge that catches 70% of errors the saving rose to 49%, at a cost of about two points of success.

The benchmark uses simulated models, so these results demonstrate that the method behaves as designed. They are not a claim about any specific commercial model. The full table, method and limitations are in the [README](../README.md#5-results) and [TECHNICAL.md](TECHNICAL.md#6-evaluation-methodology).

---

## 5. Where this approach is weak, and what would improve it

- **It needs data to shine.** With no profiles the router is a cautious cascade (still useful, but roughly 10% costlier than with learned profiles in the benchmark). Value grows as evidence accumulates.
- **Task category is a coarse signal.** Difficulty inside a category varies a lot. Adding a cheap difficulty estimate to the profile key is the most promising refinement.
- **Verification quality caps the result.** The router can only be as safe as its checks. Strong deterministic checks matter more than a clever judge wherever the output has structure.
- **Ground truth is the hard part in production.** The benchmark has exact answers. A deployment needs labelled samples, human feedback or a trusted evaluator to keep the profiles honest and to detect when a model changes behaviour.
- **Static prices.** Prices are data in the catalog and are marked unverified until checked. A live price feed would remove a manual step.
