"""Data passed into and out of the LLM layer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class RawCompletion(BaseModel):
    """What a backend returns: text and token usage, before cost and latency are added."""

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)


class LLMResult(BaseModel):
    """One completed model call, with everything the router needs to learn from it."""

    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_s: float
    cost_usd: float