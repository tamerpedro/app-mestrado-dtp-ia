"""Provedor de IA online para geracao de riscos (nao ha LLM local no ambiente).

Uma unica interface (``AIProvider.generate``) com duas implementacoes reais
(Anthropic e compativel com OpenAI) e um ``FakeProvider`` para testes, que
nunca acessam a rede. Provedor e modelo vem de ``AI_PROVIDER``/``AI_MODEL``,
por variavel de ambiente ou ``.streamlit/secrets.toml``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit sempre disponivel em producao
    st = None  # type: ignore[assignment]


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1  # 1 nova tentativa = 2 chamadas no total
TEMPERATURE = 0.0

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o-mini",
}


class AIProviderError(Exception):
    """Falha ao configurar ou chamar o provedor de IA."""


class AIProvider(ABC):
    """Interface unica para geracao de texto por IA online."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Retorna o texto bruto gerado pelo modelo para o ``prompt``."""


def _call_with_retry(call: Callable[[], str], max_retries: int) -> str:
    last_exc: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - qualquer falha de rede/SDK vira AIProviderError
            last_exc = exc
    raise AIProviderError(str(last_exc)) from last_exc


class AnthropicProvider(AIProvider):
    """Implementacao via API da Anthropic (``ANTHROPIC_API_KEY``)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        import anthropic  # import tardio: so exigido em uso real, nao nos testes

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

        def call() -> str:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        return _call_with_retry(call, self.max_retries)


class OpenAICompatibleProvider(AIProvider):
    """Implementacao via API compativel com OpenAI (``OPENAI_API_KEY``/``OPENAI_BASE_URL``)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        import openai  # import tardio: so exigido em uso real, nao nos testes

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        def call() -> str:
            response = client.chat.completions.create(
                model=self.model,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        return _call_with_retry(call, self.max_retries)


class FakeProvider(AIProvider):
    """Provedor sem rede para testes: devolve uma resposta fixa ou computada."""

    def __init__(self, response: str | Callable[[str], str] = "{}", *, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.fail:
            raise AIProviderError("Falha simulada do FakeProvider.")
        return self.response(prompt) if callable(self.response) else self.response


@dataclass(frozen=True)
class AIConfig:
    provider: str
    model: str
    anthropic_api_key: str | None
    openai_api_key: str | None
    openai_base_url: str | None


def _secret(key: str) -> str | None:
    if st is None:
        return None
    try:
        value = st.secrets.get(key)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - secrets.toml ausente ou invalido
        return None
    return str(value) if value is not None else None


def _config_value(key: str) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    return _secret(key)


def load_ai_config() -> AIConfig:
    """Le ``AI_PROVIDER``/``AI_MODEL`` e as chaves de API de env ou ``secrets.toml``."""
    provider = (_config_value("AI_PROVIDER") or "anthropic").strip().lower()
    model = _config_value("AI_MODEL") or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    return AIConfig(
        provider=provider,
        model=model,
        anthropic_api_key=_config_value("ANTHROPIC_API_KEY"),
        openai_api_key=_config_value("OPENAI_API_KEY"),
        openai_base_url=_config_value("OPENAI_BASE_URL"),
    )


def build_provider(
    config: AIConfig,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AIProvider:
    """Constroi o ``AIProvider`` configurado; lanca ``AIProviderError`` se a chave faltar."""
    if config.provider == "openai":
        if not config.openai_api_key:
            raise AIProviderError("OPENAI_API_KEY nao configurada.")
        return OpenAICompatibleProvider(
            api_key=config.openai_api_key,
            model=config.model,
            base_url=config.openai_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    if config.provider == "anthropic":
        if not config.anthropic_api_key:
            raise AIProviderError("ANTHROPIC_API_KEY nao configurada.")
        return AnthropicProvider(
            api_key=config.anthropic_api_key,
            model=config.model,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise AIProviderError(f"AI_PROVIDER desconhecido: '{config.provider}'. Use 'anthropic' ou 'openai'.")


def get_provider(**kwargs: Any) -> AIProvider:
    """Atalho: carrega a configuracao do ambiente e constroi o provedor."""
    return build_provider(load_ai_config(), **kwargs)
