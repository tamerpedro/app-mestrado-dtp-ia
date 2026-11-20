"""Isola os testes da rede e de segredos locais do desenvolvedor.

Mesmo que o desenvolvedor tenha um ``.streamlit/secrets.toml`` real (por
exemplo para rodar o app localmente com uma chave de IA), os testes nunca
devem ler esse arquivo nem variaveis de ambiente de chave de IA ja
presentes no sistema: eles sempre caem no FakeProvider (sem rede), nunca
na API real, mesmo via ``tests/test_app_language.py`` (que executa
``app.py`` inteiro com ``streamlit.testing.v1.AppTest``).
"""

from __future__ import annotations

import pytest

from src import ai_provider, eval_mode

_AI_ENV_VARS = ["AI_PROVIDER", "AI_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"]


@pytest.fixture(autouse=True)
def _no_real_ai_secrets(monkeypatch):
    for key in [*_AI_ENV_VARS, "MODO_AVALIACAO"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(ai_provider, "_secret", lambda key: None)
    monkeypatch.setattr(eval_mode, "_secret", lambda key: None)
