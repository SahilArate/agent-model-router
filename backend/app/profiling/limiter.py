"""Per-provider rate limiting for live profiling (requests and tokens per minute)."""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

WINDOW_S = 60.0


@dataclass(frozen=True)
class Limits:
    rpm: int
    tpm: int


# Deliberately below the published free-tier limits so a run never trips a 429 on purpose.
# Groq free tier (gpt-oss-120b): 30 RPM, 8,000 TPM. Google AI Studio publishes no fixed numbers.
DEFAULT_LIMITS: dict[str, Limits] = {
    "groq": Limits(rpm=20, tpm=6_000),
    "gemini": Limits(rpm=8, tpm=150_000),
    "openai": Limits(rpm=60, tpm=200_000),
    "anthropic": Limits(rpm=30, tpm=100_000),
}
FALLBACK_LIMITS = Limits(rpm=10, tpm=20_000)


def limits_for(provider: str) -> Limits:
    return DEFAULT_LIMITS.get(provider, FALLBACK_LIMITS)


class RateLimiter:
    """Sliding one-minute window over request count and token count.

    acquire() reserves an estimated token amount and waits until both limits allow the call;
    settle() replaces the estimate with the real usage so later calls are paced accurately.
    """

    def __init__(
        self,
        limits: Limits,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if limits.rpm < 1 or limits.tpm < 1:
            raise ValueError("rpm and tpm must be positive")
        self._limits = limits
        self._clock = clock
        self._sleep = sleep
        self._events: deque[list[float]] = deque()  # [timestamp, tokens]
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_S:
            self._events.popleft()

    async def acquire(self, tokens: int) -> list[float]:
        while True:
            async with self._lock:
                now = self._clock()
                self._prune(now)
                used = sum(event[1] for event in self._events)
                # A single call larger than the whole budget is allowed through on an empty
                # window, otherwise it would wait forever.
                fits = not self._events or used + tokens <= self._limits.tpm
                if len(self._events) < self._limits.rpm and fits:
                    reservation = [now, float(tokens)]
                    self._events.append(reservation)
                    return reservation
                wait = self._events[0][0] + WINDOW_S - now
            await self._sleep(max(wait, 0.0) + 0.05)

    @staticmethod
    def settle(reservation: list[float], actual_tokens: int) -> None:
        reservation[1] = float(actual_tokens)
