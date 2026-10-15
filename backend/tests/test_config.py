from app.core.config import Settings


def test_defaults_without_env_file(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.configured_providers == []


def test_provider_detected_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")

    settings = Settings(_env_file=None)

    assert settings.configured_providers == ["openai"]


def test_secret_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value")

    settings = Settings(_env_file=None)

    assert "super-secret-value" not in repr(settings)
    assert settings.anthropic_api_key.get_secret_value() == "super-secret-value"