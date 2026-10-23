"""A client wrapper for live runs: throttled, capped, retried, and recording what models said."""

import asyncio
from dataclasses import dataclass

from app.llm.backends import estimate_tokens
from app.llm.client import LLMClient
from app.llm.errors import LLMError
from app.llm.types import ChatMessage, LLMResult
from app.profiling.limiter import Limits, RateLimiter, limits_for
from app.profiling.runner import RETRY_DELAYS_S, Guard, Sleep, is_daily_quota, is_fatal
from app.registry.models import ModelProfile


@dataclass(frozen=True)
class RecordedCall:
    model_id: str
    text: str | None  # None when the call failed
    output_tokens: int = 0


class GuardedClient:
    """Drop-in for LLMClient.complete that adds provider throttling, spend and call caps,
    retries for transient errors, and a record of every reply (failed calls included)."""

    def __init__(
        self,
        inner: LLMClient,
        guard: Guard,
        *,
        limits: dict[str, Limits] | None = None,
        retry_delays_s: tuple[float, ...] = RETRY_DELAYS_S,
        sleep: Sleep = asyncio.sleep,
        limiter_sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._limits = limits or {}
        self._retry_delays_s = retry_delays_s
        self._sleep = sleep
        self._limiter_sleep = limiter_sleep
        self._limiters: dict[str, RateLimiter] = {}
        self.calls: list[RecordedCall] = []

    def _limiter(self, model: ModelProfile) -> RateLimiter:
        # Providers apply their limits per model, so each model gets its own window.
        if model.id not in self._limiters:
            provider = model.provider.value
            self._limiters[model.id] = RateLimiter(
                self._limits.get(provider) or limits_for(provider), sleep=self._limiter_sleep
            )
        return self._limiters[model.id]

    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult:
        tokens_in = sum(estimate_tokens(m.content) for m in messages)
        worst = model.cost(tokens_in, max_tokens)
        last_error: LLMError | None = None
        for attempt in range(len(self._retry_delays_s) + 1):
            if not self._guard.begin_call(worst):
                self.calls.append(RecordedCall(model.id, None))
                raise LLMError(f"run stopped: {self._guard.reason}")
            reservation = await self._limiter(model).acquire(tokens_in + max_tokens)
            try:
                result = await self._inner.complete(
                    model, messages, max_tokens=max_tokens, temperature=temperature
                )
            except LLMError as exc:
                last_error = exc
                if (
                    is_fatal(str(exc))
                    or is_daily_quota(str(exc))
                    or attempt == len(self._retry_delays_s)
                ):
                    break
                await self._sleep(self._retry_delays_s[attempt])
                continue
            RateLimiter.settle(reservation, result.input_tokens + result.output_tokens)
            self._guard.charge(result.cost_usd)
            self.calls.append(RecordedCall(model.id, result.text, result.output_tokens))
            return result
        self.calls.append(RecordedCall(model.id, None))
        raise last_error or LLMError(f"{model.id} call failed")
