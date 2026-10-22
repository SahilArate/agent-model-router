"""Converts internal traces and profiles into the API's response models."""

from app.api.schemas import (
    AttemptOut,
    BudgetOut,
    CallTraceOut,
    CandidateOut,
    CheckOut,
    EscalationOut,
    EstimateOut,
    ModelOut,
    RoutingOut,
    RunResponse,
)
from app.api.services import Mode
from app.registry.models import ModelProfile
from app.router.decision import RoutingDecision
from app.verification.outcome import Action
from app.verification.types import CheckStatus
from app.workflow.budget import Budget
from app.workflow.trace import AttemptTrace, CallTrace, WorkflowResult


def model_out(model: ModelProfile) -> ModelOut:
    return ModelOut(
        id=model.id,
        provider=model.provider.value,
        tier=model.tier.value,
        enabled=model.enabled,
        input_price_per_mtok=model.input_price_per_mtok,
        output_price_per_mtok=model.output_price_per_mtok,
        context_window=model.context_window,
        supports_tools=model.supports_tools,
        supports_structured_output=model.supports_structured_output,
        approved_for_sensitive_data=model.approved_for_sensitive_data,
        notes=model.notes,
    )


def _routing_out(decision: RoutingDecision) -> RoutingOut:
    return RoutingOut(
        model_id=decision.model.id,
        mode=decision.mode.value,
        reason=decision.reason,
        quality_floor=decision.quality_floor,
        fallbacks=[m.id for m in decision.fallbacks],
        requires_independent_verifier=decision.requires_independent_verifier,
        candidates=[
            CandidateOut(
                model_id=c.model.id,
                tier=c.model.tier.value,
                expected_cost_usd=c.expected_cost_usd,
                qualified=c.qualified,
                estimate=EstimateOut(
                    mean=c.estimate.mean,
                    lower_bound=c.estimate.lower_bound,
                    sample_size=c.estimate.sample_size,
                    sufficient_evidence=c.estimate.sufficient_evidence,
                ),
            )
            for c in decision.candidates
        ],
        rejected={model_id: list(reasons) for model_id, reasons in decision.rejected.items()},
    )


def _attempt_out(attempt: AttemptTrace) -> AttemptOut:
    return AttemptOut(
        model_id=attempt.model_id,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        latency_s=attempt.latency_s,
        generation_cost_usd=attempt.generation_cost_usd,
        verification_cost_usd=attempt.verification_cost_usd,
        action=attempt.action.value if attempt.action else None,
        error=attempt.error,
        checks=[
            CheckOut(
                name=c.name,
                status=c.status.value,
                detail=c.detail,
                cost_usd=c.cost_usd,
                judge_model=c.judge_model,
                independent=c.independent,
            )
            for c in attempt.checks
        ],
    )


def escalation_steps(trace: CallTrace) -> list[EscalationOut]:
    """One entry per hand-over to a stronger model, with the evidence that caused it."""
    steps = []
    for attempt, following in zip(trace.attempts, trace.attempts[1:], strict=False):
        if attempt.action is not Action.ESCALATE and not attempt.error:
            continue
        failed = next((c for c in attempt.checks if c.status is not CheckStatus.PASS), None)
        if failed is not None:
            check, reason = failed.name, failed.detail
            judge_model, independent = failed.judge_model, failed.independent
        else:
            check, reason = "provider", attempt.error or "the call failed"
            judge_model = independent = None
        steps.append(
            EscalationOut(
                from_model=attempt.model_id,
                to_model=following.model_id,
                check=check,
                reason=reason,
                judge_model=judge_model,
                independent=independent,
                wasted_cost_usd=attempt.cost_usd,
            )
        )
    return steps


def _call_trace_out(trace: CallTrace) -> CallTraceOut:
    return CallTraceOut(
        agent=trace.agent.value,
        category=trace.category.value,
        risk=trace.risk.value,
        final_action=trace.final_action.value,
        reason=trace.reason,
        output=trace.output,
        cost_usd=trace.cost_usd,
        latency_s=trace.latency_s,
        escalations=trace.escalations,
        escalation_steps=escalation_steps(trace),
        routing=_routing_out(trace.decision) if trace.decision else None,
        attempts=[_attempt_out(a) for a in trace.attempts],
    )


def run_response(result: WorkflowResult, budget: Budget, mode: Mode) -> RunResponse:
    return RunResponse(
        mode=mode,
        task=result.task,
        completed=result.completed,
        final_answer=result.final_answer,
        abstained_at=result.abstained_at.value if result.abstained_at else None,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        budget=BudgetOut(
            spent_usd=budget.spent_usd,
            calls=budget.calls,
            max_cost_usd=budget.max_cost_usd,
            max_calls=budget.max_calls,
        ),
        traces=[_call_trace_out(t) for t in result.traces],
    )
