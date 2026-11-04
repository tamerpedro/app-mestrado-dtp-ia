"""Motor LLM restrito ao repositorio.

Desligado por padrao (``config.LLMConfig.enabled=False``); so pode ser usado
com endpoint local, a menos que ``allow_external=True`` seja definido
explicitamente (ver validacao em ``evaluation/config.py``). Em nenhum caso o
app ou os outros motores enviam dados a servicos externos: isso e exclusivo
deste motor, e so quando explicitamente habilitado.

O modelo so pode "escolher" entre os IDs de risco realmente existentes na
biblioteca carregada -- qualquer ID fora dessa lista e descartado do
ranking e contado na taxa de IDs invalidos (ver ``metrics/validity.py``).
Isso vale tanto para um LLM real quanto para o client falso usado nos
testes.
"""

from __future__ import annotations

import json
from typing import Protocol

from src.models import ContractContext, RiskItem

from ..config import LLMConfig
from ..text_utils import context_query_text
from .base import RankResult

name = "llm"


class LLMNotEnabledError(RuntimeError):
    pass


class LLMClient(Protocol):
    """Qualquer cliente que receba um prompt e devolva o texto da resposta."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class LocalHTTPLLMClient:
    """Cliente padrao: chama um endpoint HTTP local compativel com a API de
    chat da OpenAI (ex.: Ollama, llama.cpp server, vLLM). A dependencia
    ``requests`` so e importada aqui, na hora da chamada -- nao e necessaria
    para carregar o modulo nem para os testes (que usam um client falso).
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            self.config.endpoint,
            json={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


SYSTEM_PROMPT = (
    "Voce ajuda a selecionar riscos de uma biblioteca fixa para uma contratacao de TIC. "
    "Responda SOMENTE com um JSON de uma lista de IDs de risco (ex.: [\"R001\", \"R006\"]), "
    "ordenados do mais para o menos relevante, usando apenas IDs que aparecam na lista de "
    "candidatos fornecida. Nao invente IDs. Nao inclua texto fora do JSON."
)


def _build_user_prompt(context: ContractContext, risks: list[RiskItem], max_candidates: int) -> str:
    candidates = risks[:max_candidates]
    candidate_lines = [
        f"- {risk.id}: {risk.titulo} (categoria={risk.categoria}, "
        f"palavras_chave={';'.join(risk.palavras_chave)})"
        for risk in candidates
    ]
    return (
        f"Contratacao:\n{context_query_text(context)}\n\n"
        f"Candidatos (use somente estes IDs):\n" + "\n".join(candidate_lines)
    )


def _parse_ids(raw_response: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        start, end = raw_response.find("["), raw_response.rfind("]")
        if start == -1 or end == -1 or end < start:
            return ()
        try:
            parsed = json.loads(raw_response[start : end + 1])
        except json.JSONDecodeError:
            return ()

    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def rank(
    context: ContractContext,
    risks: list[RiskItem],
    config: LLMConfig,
    client: LLMClient | None = None,
) -> RankResult:
    if not config.enabled:
        raise LLMNotEnabledError(
            "O motor LLM esta desligado por padrao (config.LLMConfig.enabled=False). "
            "Habilite explicitamente para usa-lo."
        )

    client = client or LocalHTTPLLMClient(config)
    user_prompt = _build_user_prompt(context, risks, config.max_candidates)
    raw_response = client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    raw_ids = _parse_ids(raw_response)

    known_ids = {risk.id for risk in risks}
    seen: set[str] = set()
    valid_ids: list[str] = []
    for risk_id in raw_ids:
        if risk_id in known_ids and risk_id not in seen:
            valid_ids.append(risk_id)
            seen.add(risk_id)

    return RankResult(ranked_ids=tuple(valid_ids), raw_ids_before_validation=raw_ids)
