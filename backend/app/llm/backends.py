"""Backends that actually produce completions: a mock for offline runs and LiteLLM for real ones."""

import asyncio
import math
from collections.abc import Callable
from typing import Protocol

from app.core.config import Settings, get_settings
from app.llm.errors import LLMError
from app.llm.types import ChatMessage, RawCompletion
from app.registry.models import ModelProfile, Provider

_CHARS_PER_TOKEN = 4

Responder = Callable[[ModelProfile, list[ChatMessage]], str]


class LLMBackend(Protocol):
    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion: ...


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token). Only used by the mock."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _echo_responder(model: ModelProfile, messages: list[ChatMessage]) -> str:
    last = messages[-1].content if messages else ""
    return f"[{model.id}] {last}"


class MockBackend:
    """Deterministic, offline backend. Pass a responder to script what each model says."""

    def __init__(self, responder: Responder | None = None, latency_s: float = 0.0) -> None:
        self._responder = responder or _echo_responder
        self._latency_s = latency_s

    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion:
        if self._latency_s:
            await asyncio.sleep(self._latency_s)

        text = self._responder(model, messages)[: max_tokens * _CHARS_PER_TOKEN]
        return RawCompletion(
            text=text,
            input_tokens=sum(estimate_tokens(m.content) for m in messages),
            output_tokens=estimate_tokens(text),
        )


class LiteLLMBackend:
    """Calls real providers through LiteLLM, using keys from our settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _api_key(self, provider: Provider) -> str | None:
        keys = {
            Provider.OPENAI: self._settings.openai_api_key,
            Provider.ANTHROPIC: self._settings.anthropic_api_key,
            Provider.GEMINI: self._settings.gemini_api_key,
        }
        secret = keys.get(provider)
        return secret.get_secret_value() if secret else None

    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion:
        api_key = self._api_key(model.provider)
        if api_key is None:
            raise LLMError(f"no API key configured for provider '{model.provider}'")

        import litellm  # imported lazily: it is slow to import and not needed for mock runs

        response = await litellm.acompletion(
            model=model.litellm_model,
            messages=[m.model_dump() for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) or getattr(
            usage, "cache_read_input_tokens", 0
        )
        return RawCompletion(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=int(cached or 0),
        )


class DispatchBackend:
    """Sends mock-provider models to the mock backend and everything else to the live one."""

    def __init__(self, mock: LLMBackend, live: LLMBackend) -> None:
        self._mock = mock
        self._live = live

    async def complete(
        self,
        model: ModelProfile,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion:
        backend = self._mock if model.provider is Provider.MOCK else self._live
        return await backend.complete(
            model, messages, max_tokens=max_tokens, temperature=temperature
        )