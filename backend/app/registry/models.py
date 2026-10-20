from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.types import Tier

_TOKENS_PER_MILLION = 1_000_000


class Provider(StrEnum):
    MOCK = "mock"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    provider: Provider
    litellm_model: str
    tier: Tier
    input_price_per_mtok: float = Field(ge=0)
    output_price_per_mtok: float = Field(ge=0)
    cached_input_price_per_mtok: float | None = Field(default=None, ge=0)
    context_window: int = Field(gt=0)
    supports_tools: bool = False
    supports_structured_output: bool = False
    approved_for_sensitive_data: bool = False
    enabled: bool = True
    pricing_verified_on: date | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _cached_price_not_above_input(self) -> "ModelProfile":
        cached = self.cached_input_price_per_mtok
        if cached is not None and cached > self.input_price_per_mtok:
            raise ValueError("cached input price cannot exceed the regular input price")
        return self

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        if min(input_tokens, output_tokens, cached_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if cached_tokens > input_tokens:
            raise ValueError("cached_tokens cannot exceed input_tokens")
        cached_price = self.cached_input_price_per_mtok
        if cached_price is None:
            cached_price = self.input_price_per_mtok
        total = (
            (input_tokens - cached_tokens) * self.input_price_per_mtok
            + cached_tokens * cached_price
            + output_tokens * self.output_price_per_mtok
        )
        return total / _TOKENS_PER_MILLION
