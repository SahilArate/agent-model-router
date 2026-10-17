"""Runs one LLM call end to end: route, generate, verify, then accept, escalate or abstain."""

import time

from app.llm.backends import estimate_tokens
from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.types import ChatMessage
from app.router.constraints import NoEligibleModelError
from app.router.decision import RoutingDecision
from app.router.engine import ModelRouter
from app.router.request import RoutingRequest
from app.verification.outcome import Action, decide
from app.verification.types import VerificationContext
from app.workflow.budget import Budget
from app.workflow.spec import CallSpec
from app.workflow.trace import AttemptTrace, CallTrace

_EMPTY_PLACEHOLDER = "(empty response)"


class RoutedExecutor:
    """Every model call in the system goes through here.

    The loop is bounded: each pass uses up one model from the router's escalation path,
    and the workflow budget is checked before every call.
    """

    def __init__(self, router: ModelRouter, client: LLMClient) -> None:
        self._router = router
        self._client = client

    async def run(self, spec: CallSpec, budget: Budget) -> CallTrace:
        request = RoutingRequest(
            agent=spec.agent,
            category=spec.category,
            risk=spec.risk,
            input_tokens=sum(estimate_tokens(m.content) for m in spec.messages),
            max_output_tokens=spec.max_output_tokens,
            needs_structured_output=spec.needs_structured_output,
            contains_sensitive_data=spec.contains_sensitive_data,
            allowed_providers=spec.allowed_providers,
        )

        try:
            decision = self._router.route(request)
        except NoEligibleModelError as exc:
            return self._finish(spec, None, [], None, Action.ABSTAIN, str(exc))

        model = decision.model
        fallbacks = decision.fallbacks
        messages = list(spec.messages)
        attempts: list[AttemptTrace] = []

        while True:
            if budget.exhausted:
                reason = "workflow budget exhausted before the next attempt"
                return self._finish(spec, decision, attempts, None, Action.ABSTAIN, reason)

            started = time.perf_counter()
            try:
                result = await self._client.complete(
                    model, messages, max_tokens=spec.max_output_tokens
                )
            except LLMError as exc:
                budget.charge_call(0.0)
                attempts.append(
                    AttemptTrace(
                        model_id=model.id,
                        latency_s=time.perf_counter() - started,
                        error=str(exc),
                    )
                )
                if not fallbacks:
                    reason = f"provider call failed and no fallback remains ({exc})"
                    return self._finish(spec, decision, attempts, None, Action.ABSTAIN, reason)
                model, fallbacks = fallbacks[0], fallbacks[1:]
                continue

            budget.charge_call(result.cost_usd)
            report = await spec.pipeline.verify(
                VerificationContext(spec.task_prompt, result.text, model)
            )
            budget.charge(report.cost_usd)
            outcome = decide(report, fallbacks)

            attempts.append(
                AttemptTrace(
                    model_id=model.id,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_s=result.latency_s,
                    generation_cost_usd=result.cost_usd,
                    verification_cost_usd=report.cost_usd,
                    checks=report.results,
                    action=outcome.action,
                )
            )

            if outcome.action is Action.ACCEPT:
                return self._finish(
                    spec, decision, attempts, result.text, Action.ACCEPT, outcome.reason
                )
            if outcome.action is Action.ABSTAIN or outcome.next_model is None:
                return self._finish(
                    spec, decision, attempts, None, Action.ABSTAIN, outcome.reason
                )

            model, fallbacks = outcome.next_model, outcome.remaining_fallbacks
            messages = [
                *spec.messages,
                ChatMessage(role="assistant", content=result.text or _EMPTY_PLACEHOLDER),
                ChatMessage(
                    role="user",
                    content=(
                        f"Your previous answer failed verification ({outcome.failure_evidence}). "
                        "Fix these problems and answer again."
                    ),
                ),
            ]

    @staticmethod
    def _finish(
        spec: CallSpec,
        decision: RoutingDecision | None,
        attempts: list[AttemptTrace],
        output: str | None,
        action: Action,
        reason: str,
    ) -> CallTrace:
        return CallTrace(
            agent=spec.agent,
            category=spec.category,
            risk=spec.risk,
            decision=decision,
            attempts=tuple(attempts),
            output=output,
            final_action=action,
            reason=reason,
        )