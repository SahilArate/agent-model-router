import asyncio

import pytest

from app.core.config import Settings
from app.core.types import Tier
from app.llm.backends import DispatchBackend, LiteLLMBackend, MockBackend
from app.llm.client import LLMClient, build_default_client
from app.llm.errors import LLMError
from app.llm.types import ChatMessage, RawCompletion
from app.registry.models import ModelProfile, Provider


def make_model(provider: Provider = Provider.MOCK, **overrides) -> ModelProfile:
    fields = {
        "id": "test-model",
        "provider": provider,
        "litellm_model": "mock/test-model",
        "tier": Tier.SMALL,
        "input_price_per_mtok": 1.0,
        "output_price_per_mtok": 4.0,
        "context_window": 8000,
    }
    return ModelProfile(**{**fields, **overrides})


USER_HELLO = [ChatMessage(role="user", content="hello world")]


class FakeBackend:
    """Returns a fixed completion, or raises/sleeps, so client behaviour can be tested."""

    def __init__(self, completion=None, error=None, delay_s=0.0):
        self.completion = completion
        self.error = error
        self.delay_s = delay_s
        self.calls = 0

    async def complete(self, model, messages, *, max_tokens, temperature):
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error:
            raise self.error
        return self.completion


async def test_result_carries_usage_and_cost():
    client = LLMClient(MockBackend())
    model = make_model()

    result = await client.complete(model, USER_HELLO)

    assert result.model_id == "test-model"
    assert result.text == "[test-model] hello world"
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.cost_usd == pytest.approx(model.cost(result.input_tokens, result.output_tokens))


async def test_latency_is_measured():
    client = LLMClient(MockBackend(latency_s=0.05))

    result = await client.complete(make_model(), USER_HELLO)

    assert result.latency_s >= 0.04


async def test_mock_output_respects_max_tokens():
    backend = MockBackend(responder=lambda model, messages: "x" * 1000)
    client = LLMClient(backend)

    result = await client.complete(make_model(), USER_HELLO, max_tokens=10)

    assert result.output_tokens <= 10


async def test_backend_failure_is_wrapped():
    boom = RuntimeError("provider exploded")
    client = LLMClient(FakeBackend(error=boom))

    with pytest.raises(LLMError, match="provider exploded") as info:
        await client.complete(make_model(), USER_HELLO)

    assert info.value.__cause__ is boom


async def test_timeout_is_reported():
    client = LLMClient(FakeBackend(delay_s=1.0), timeout_s=0.05)

    with pytest.raises(LLMError, match="timed out"):
        await client.complete(make_model(), USER_HELLO)


async def test_cached_tokens_are_clamped_to_input_tokens():
    completion = RawCompletion(text="ok", input_tokens=10, output_tokens=5, cached_tokens=999)
    client = LLMClient(FakeBackend(completion=completion))

    result = await client.complete(make_model(), USER_HELLO)

    assert result.cached_tokens == 10


async def test_invalid_arguments_are_rejected():
    client = LLMClient(MockBackend())

    with pytest.raises(ValueError):
        await client.complete(make_model(), [])
    with pytest.raises(ValueError):
        await client.complete(make_model(), USER_HELLO, max_tokens=0)


async def test_dispatch_routes_by_provider():
    mock = FakeBackend(completion=RawCompletion(text="m", input_tokens=1, output_tokens=1))
    live = FakeBackend(completion=RawCompletion(text="l", input_tokens=1, output_tokens=1))
    client = LLMClient(DispatchBackend(mock=mock, live=live))

    await client.complete(make_model(Provider.MOCK), USER_HELLO)
    await client.complete(make_model(Provider.ANTHROPIC), USER_HELLO)

    assert (mock.calls, live.calls) == (1, 1)


async def test_live_backend_requires_an_api_key(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    backend = LiteLLMBackend(Settings(_env_file=None))
    client = LLMClient(backend)

    with pytest.raises(LLMError, match="no API key"):
        await client.complete(make_model(Provider.ANTHROPIC), USER_HELLO)


async def test_default_client_runs_mock_models_offline():
    client = build_default_client(Settings(_env_file=None))

    result = await client.complete(make_model(Provider.MOCK), USER_HELLO)

    assert result.text.startswith("[test-model]")