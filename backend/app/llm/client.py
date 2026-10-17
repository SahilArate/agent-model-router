"""The single entry point for model calls. Adds timing, cost and uniform error handling."""

import asyncio
import time

from app.core.config import Settings
from app.llm.backends import DispatchBackend, LiteLLMBackend, LLMBackend, MockBackend
from app.llm.errors import LLMError
from app.llm.types import ChatMessage, LLMResult
from app.registry.models import ModelProfile

DEFAULT_TIMEOUT_S = 60.0


class LLMClient:
    """Calls a model through a backend and returns text plus measured usage, latency and cost.

    Retries and fallbacks are deliberately not handled here: they are routing decisions
    and belong to the router's cascade, where they can be bounded and recorded.
    """

    def __init__(self, backend: LLMBackend, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._backend = backend
        self._timeout_s = timeout_s

    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._backend.complete(
                    model, messages, max_tokens=max_tokens, temperature=temperature
                ),
                timeout=self._timeout_s,
            )
        except TimeoutError as exc:
            raise LLMError(f"{model.id} timed out after {self._timeout_s}s") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{model.id} call failed: {exc}") from exc
        latency_s = time.perf_counter() - started

        cached_tokens = min(raw.cached_tokens, raw.input_tokens)
        return LLMResult(
            model_id=model.id,
            text=raw.text,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cached_tokens=cached_tokens,
            latency_s=latency_s,
            cost_usd=model.cost(raw.input_tokens, raw.output_tokens, cached_tokens),
        )


def build_default_client(
    settings: Settings | None = None, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> LLMClient:
    """Mock models run offline; every other provider goes through LiteLLM."""
    backend = DispatchBackend(mock=MockBackend(), live=LiteLLMBackend(settings))
    return LLMClient(backend, timeout_s=timeout_s)