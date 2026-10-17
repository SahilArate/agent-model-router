"""Errors raised by the LLM layer."""


class LLMError(Exception):
    """A model call failed (provider error, timeout or missing credentials)."""