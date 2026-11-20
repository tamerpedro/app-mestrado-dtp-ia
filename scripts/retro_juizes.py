"""Avaliacao retrospectiva: juizes do painel economico (um por fornecedor).

Cada juiz e o modelo de texto de menor custo utilizavel do fornecedor na data
da selecao (registrada em ``<corpus>/retrospectiva/painel/selecao_modelos.md``).
Todos recebem so o texto do prompt e devolvem o texto da resposta e os tokens
de entrada e de saida da chamada. Temperatura 0 onde o modelo aceita; onde so
ha nivel de raciocinio, vale o padrao do fornecedor (nada e enviado).

As chaves nunca sao impressas nem gravadas: a da Anthropic e a que o projeto ja
usa (``ANTHROPIC_API_KEY``, variavel de ambiente ou ``.streamlit/secrets.toml``);
as da OpenAI e do Google vem do arquivo indicado em ``--chaves``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_SAIDA = 16000  # o raciocinio conta no limite de saida; com 2.048 o JSON do Opus chegava cortado
TIMEOUT_PADRAO = 180.0
TENTATIVAS_DE_REDE = 2  # das bibliotecas (rede, limite de taxa); a resposta invalida e tratada em retro_julgar


class JuizError(Exception):
    """Falha ao configurar ou chamar um juiz do painel."""


@dataclass(frozen=True)
class Resposta:
    texto: str
    tokens_entrada: int
    tokens_saida: int


class Juiz:
    """Um juiz do painel. As subclasses definem o modelo e fazem a chamada."""

    nome = ""  # usado em --painel e no nome do arquivo (julgamento_<nome>.csv)
    modelo = ""
    variavel = ""  # nome da variavel da chave
    parametros = ""  # o que foi enviado alem do prompt; gravado em cada linha
    preco_entrada = 0.0  # US$ por milhao de tokens, na data da selecao
    preco_saida = 0.0

    def __init__(self, chave: str | None, *, timeout: float = TIMEOUT_PADRAO) -> None:
        self._chave = (chave or "").strip()
        self.timeout = timeout
        self._cliente = None

    def perguntar(self, prompt: str) -> Resposta:
        if not self._chave:
            raise JuizError(f"{self.variavel} nao configurada.")
        try:
            return self._chamar(prompt)
        except Exception as exc:  # noqa: BLE001 - qualquer falha de rede ou da biblioteca
            mensagem = f"{type(exc).__name__}: {exc}".replace(self._chave, "<chave>")
            raise JuizError(mensagem) from None

    def _chamar(self, prompt: str) -> Resposta:
        raise NotImplementedError


class JuizAnthropic(Juiz):
    nome = "anthropic"
    modelo = "claude-haiku-5-5"
    variavel = "ANTHROPIC_API_KEY"
    parametros = f"acesso=Anthropic Messages API; temperatura=nao aceita; effort=padrao do fornecedor; max_tokens={MAX_SAIDA}"
    preco_entrada = 0.10
    preco_saida = 0.50

    def _chamar(self, prompt: str) -> Resposta:
        if self._cliente is None:
            import anthropic  # import tardio: so exigido em uso real, nao nos testes

            self._cliente = anthropic.Anthropic(api_key=self._chave, timeout=self.timeout, max_retries=TENTATIVAS_DE_REDE)
        resposta = self._cliente.messages.create(
            model=self.modelo, max_tokens=MAX_SAIDA, messages=[{"role": "user", "content": prompt}]
        )
        texto = "".join(bloco.text for bloco in resposta.content if bloco.type == "text")
        return Resposta(texto, resposta.usage.input_tokens, resposta.usage.output_tokens)


class JuizOpenAI(Juiz):
    nome = "openai"
    modelo = "gpt-6-luna"
    variavel = "OPENAI_API_KEY"
    parametros = (
        f"acesso=OpenAI Responses API; temperatura=nao aceita; reasoning.effort=padrao do fornecedor; "
        f"max_output_tokens={MAX_SAIDA}"
    )
    preco_entrada = 0.10
    preco_saida = 0.50

    def _chamar(self, prompt: str) -> Resposta:
        if self._cliente is None:
            import openai

            self._cliente = openai.OpenAI(api_key=self._chave, timeout=self.timeout, max_retries=TENTATIVAS_DE_REDE)
        resposta = self._cliente.responses.create(model=self.modelo, input=prompt, max_output_tokens=MAX_SAIDA)
        uso = resposta.usage
        return Resposta(resposta.output_text or "", uso.input_tokens, uso.output_tokens)


class JuizGoogle(Juiz):
    nome = "google"
    modelo = "gemini-3.1-flash-lite"
    variavel = "GEMINI_API_KEY"
    parametros = (
        f"acesso=Gemini API (google-genai, api_key); temperature=0; thinking=padrao do fornecedor; "
        f"max_output_tokens={MAX_SAIDA}"
    )
    preco_entrada = 0.25
    preco_saida = 1.50

    def _chamar(self, prompt: str) -> Resposta:
        from google import genai
        from google.genai import types

        if self._cliente is None:
            # Gemini API (so api_key, sem vertexai=True); o cliente fica guardado: ele fecha a conexao ao ser descartado
            self._cliente = genai.Client(
                api_key=self._chave, http_options=types.HttpOptions(timeout=int(self.timeout * 1000))
            )
        resposta = self._cliente.models.generate_content(
            model=self.modelo,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=MAX_SAIDA),
        )
        uso = resposta.usage_metadata
        # os tokens de raciocinio sao cobrados como saida
        saida = (uso.candidates_token_count or 0) + (uso.thoughts_token_count or 0)
        return Resposta(resposta.text or "", uso.prompt_token_count or 0, saida)


JUIZES: dict[str, type[Juiz]] = {classe.nome: classe for classe in (JuizAnthropic, JuizOpenAI, JuizGoogle)}

_LINHA_ENV_RE = re.compile(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)$")


def ler_chaves(caminho: Path) -> dict[str, str]:
    """``{NOME: valor}`` de um arquivo ``NOME=valor`` (aspas e ``export`` opcionais)."""
    if not caminho.is_file():
        raise JuizError(f"Arquivo de chaves nao encontrado: {caminho}")
    chaves = {}
    for linha in caminho.read_text(encoding="utf-8-sig").splitlines():
        achado = _LINHA_ENV_RE.match(linha)
        if achado and not linha.lstrip().startswith("#"):
            chaves[achado.group(1)] = achado.group(2).strip().strip("\"'")
    return chaves


def construir(nome: str, chaves: Path | None, *, timeout: float = TIMEOUT_PADRAO) -> Juiz:
    """O juiz ``nome`` com a chave dele; sem a chave, so falha quando for chamado."""
    if nome not in JUIZES:
        raise JuizError(f"Juiz desconhecido: '{nome}'. Use um de: {', '.join(JUIZES)}.")
    classe = JUIZES[nome]
    if nome == JuizAnthropic.nome:
        from src import ai_provider

        chave = ai_provider._config_value(classe.variavel)
    else:
        chave = ler_chaves(chaves).get(classe.variavel) if chaves else None
    return classe(chave, timeout=timeout)
