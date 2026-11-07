import pytest

from src.ai_provider import (
    AIProviderError,
    AnthropicProvider,
    FakeProvider,
    OpenAICompatibleProvider,
    _call_with_retry,
    build_provider,
    load_ai_config,
)
from src.ai_provider import AIConfig


def test_fake_provider_returns_fixed_response_and_records_prompt():
    provider = FakeProvider("ola")

    result = provider.generate("prompt de teste")

    assert result == "ola"
    assert provider.calls == ["prompt de teste"]


def test_fake_provider_supports_callable_response():
    provider = FakeProvider(lambda prompt: f"echo:{prompt}")

    assert provider.generate("abc") == "echo:abc"


def test_fake_provider_can_simulate_failure():
    provider = FakeProvider(fail=True)

    with pytest.raises(AIProviderError):
        provider.generate("qualquer")


def test_call_with_retry_succeeds_after_one_failure():
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("falha temporaria")
        return "ok"

    assert _call_with_retry(flaky, max_retries=1) == "ok"
    assert attempts["count"] == 2


def test_call_with_retry_raises_ai_provider_error_after_exhausting_retries():
    def always_fails() -> str:
        raise RuntimeError("falha permanente")

    with pytest.raises(AIProviderError):
        _call_with_retry(always_fails, max_retries=1)


def test_load_ai_config_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("AI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = load_ai_config()

    assert config.provider == "openai"
    assert config.model == "gpt-test"
    assert config.openai_api_key == "sk-test"
    assert config.openai_base_url == "https://example.test/v1"


def test_load_ai_config_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)

    config = load_ai_config()

    assert config.provider == "anthropic"
    assert config.model


def test_build_provider_returns_anthropic_provider():
    config = AIConfig(
        provider="anthropic",
        model="claude-test",
        anthropic_api_key="key",
        openai_api_key=None,
        openai_base_url=None,
    )

    provider = build_provider(config)

    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-test"


def test_build_provider_returns_openai_provider():
    config = AIConfig(
        provider="openai",
        model="gpt-test",
        anthropic_api_key=None,
        openai_api_key="key",
        openai_base_url="https://example.test/v1",
    )

    provider = build_provider(config)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://example.test/v1"


def test_build_provider_raises_when_anthropic_key_missing():
    config = AIConfig(
        provider="anthropic", model="claude-test", anthropic_api_key=None, openai_api_key=None, openai_base_url=None
    )

    with pytest.raises(AIProviderError):
        build_provider(config)


def test_build_provider_raises_when_openai_key_missing():
    config = AIConfig(
        provider="openai", model="gpt-test", anthropic_api_key=None, openai_api_key=None, openai_base_url=None
    )

    with pytest.raises(AIProviderError):
        build_provider(config)


def test_build_provider_raises_for_unknown_provider():
    config = AIConfig(
        provider="azure", model="m", anthropic_api_key="k", openai_api_key="k", openai_base_url=None
    )

    with pytest.raises(AIProviderError):
        build_provider(config)
