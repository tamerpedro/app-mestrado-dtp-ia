"""Geracao de 1 risco de IA por categoria, a partir de exemplos da biblioteca.

O prompt fica versionado em ``prompts/gerar_risco_v1.txt`` (``PROMPT_VERSION``
vai para o log por processo). A saida da IA e JSON, validado nos campos do
``RiskItem`` antes de virar sugestao na avaliacao cega.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .ai_provider import AIProvider, AIProviderError
from .models import ContractContext, RiskItem

PROMPT_VERSION = "v1"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "gerar_risco_v1.txt"

PROMPT_VERSION_V2 = "v2"
PROMPT_PATH_V2 = Path(__file__).resolve().parent.parent / "prompts" / "gerar_risco_v2.txt"

MAX_EXAMPLE_FIELD_CHARS = 160
MAX_EXAMPLES = 3
MIN_EXAMPLES = 2

REQUIRED_TEXT_FIELDS = [
    "titulo",
    "causa",
    "consequencia",
    "acao_preventiva",
    "acao_contingencia",
    "responsavel_sugerido",
]
REQUIRED_SCALE_FIELDS = ["probabilidade_padrao", "impacto_padrao"]

LANGUAGE_NAMES = {"pt": "português do Brasil", "en": "English"}


class RiskGenerationError(Exception):
    """Falha ao gerar ou validar o risco de IA (prompt, chamada ou JSON invalido)."""


@dataclass(frozen=True)
class GeneratedRisk:
    titulo: str
    causa: str
    consequencia: str
    probabilidade_padrao: int
    impacto_padrao: int
    acao_preventiva: str
    acao_contingencia: str
    responsavel_sugerido: str


def _truncate(value: str, limit: int = MAX_EXAMPLE_FIELD_CHARS) -> str:
    value = (value or "").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _example_block(risk: RiskItem, lang: str) -> str:
    return (
        f"- titulo: {_truncate(risk.text('titulo', lang))}\n"
        f"  causa: {_truncate(risk.text('causa', lang))}\n"
        f"  consequencia: {_truncate(risk.text('consequencia', lang))}\n"
        f"  acao_preventiva: {_truncate(risk.text('acao_preventiva', lang))}\n"
        f"  acao_contingencia: {_truncate(risk.text('acao_contingencia', lang))}"
    )


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _reference_section(library_risk: RiskItem | None, lang: str) -> str:
    if library_risk is None:
        return (
            "## Risco de referência da biblioteca\n"
            "Nenhum risco da biblioteca atingiu a pontuação mínima nesta categoria; "
            "proponha um risco plausível por conta própria."
        )
    return (
        "## Risco já identificado pela biblioteca nesta categoria\n"
        "Proponha um risco DIFERENTE deste (outra causa, consequência ou ângulo); "
        "nunca uma reformulação do mesmo risco:\n"
        f"- titulo: {_truncate(library_risk.text('titulo', lang))}\n"
        f"- causa: {_truncate(library_risk.text('causa', lang))}\n"
        f"- consequencia: {_truncate(library_risk.text('consequencia', lang))}"
    )


def build_prompt(
    context: ContractContext,
    category_label_text: str,
    library_risk: RiskItem | None,
    example_risks: list[RiskItem],
    lang: str = "pt",
) -> str:
    """Monta o prompt a partir do template versionado (``PROMPT_PATH``)."""
    examples = example_risks[:MAX_EXAMPLES]
    examples_text = "\n".join(_example_block(risk, lang) for risk in examples) or "(nenhum exemplo disponível)"
    template = load_prompt_template()
    return template.format(
        idioma=LANGUAGE_NAMES.get(lang, LANGUAGE_NAMES["pt"]),
        objeto=context.objeto,
        tipo_contratacao=context.tipo_contratacao,
        criticidade=context.criticidade,
        prazo=context.prazo,
        modalidade=context.modalidade,
        contexto=context.contexto,
        categoria=category_label_text,
        secao_risco_referencia=_reference_section(library_risk, lang),
        exemplos=examples_text,
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    match = _JSON_BLOCK_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RiskGenerationError(f"Resposta da IA não é um JSON válido: {exc}") from exc
    if not isinstance(data, dict):
        raise RiskGenerationError("Resposta da IA não é um objeto JSON.")
    return data


def validate_generated_risk(data: dict) -> GeneratedRisk:
    """Valida o JSON da IA nos campos do ``RiskItem``; lanca ``RiskGenerationError`` caso contrario."""
    missing_text = [field for field in REQUIRED_TEXT_FIELDS if not str(data.get(field, "")).strip()]
    missing_scale = [field for field in REQUIRED_SCALE_FIELDS if data.get(field) is None]
    if missing_text or missing_scale:
        missing = [*missing_text, *missing_scale]
        raise RiskGenerationError(f"Campos ausentes na resposta da IA: {', '.join(missing)}")

    try:
        probabilidade = int(data["probabilidade_padrao"])
        impacto = int(data["impacto_padrao"])
    except (TypeError, ValueError) as exc:
        raise RiskGenerationError("probabilidade_padrao/impacto_padrao devem ser inteiros.") from exc
    if probabilidade not in range(1, 6) or impacto not in range(1, 6):
        raise RiskGenerationError("probabilidade_padrao/impacto_padrao devem estar entre 1 e 5.")

    return GeneratedRisk(
        titulo=str(data["titulo"]).strip(),
        causa=str(data["causa"]).strip(),
        consequencia=str(data["consequencia"]).strip(),
        probabilidade_padrao=probabilidade,
        impacto_padrao=impacto,
        acao_preventiva=str(data["acao_preventiva"]).strip(),
        acao_contingencia=str(data["acao_contingencia"]).strip(),
        responsavel_sugerido=str(data["responsavel_sugerido"]).strip(),
    )


def generate_risk_for_category(
    provider: AIProvider,
    context: ContractContext,
    category_label_text: str,
    library_risk: RiskItem | None,
    example_risks: list[RiskItem],
    lang: str = "pt",
) -> GeneratedRisk:
    """Gera e valida 1 risco de IA para a categoria; falhas viram ``RiskGenerationError``."""
    prompt = build_prompt(context, category_label_text, library_risk, example_risks, lang=lang)
    try:
        raw = provider.generate(prompt)
    except AIProviderError as exc:
        raise RiskGenerationError(str(exc)) from exc
    return validate_generated_risk(_extract_json(raw))


# --------------------------------------------------------------------------
# v2: lista de N riscos genericos por categoria, cientes de todo o catalogo
# (catalogo + IA misturados na interface, sem pareamento 1-a-1)
# --------------------------------------------------------------------------


def load_prompt_template_v2() -> str:
    return PROMPT_PATH_V2.read_text(encoding="utf-8")


def build_prompt_v2(
    context: ContractContext,
    category_label_text: str,
    count: int,
    catalog_titles: list[str],
    example_risks: list[RiskItem],
    lang: str = "pt",
) -> str:
    """Monta o prompt v2: pede ``count`` riscos genericos, cientes de todos os
    titulos ja cadastrados (``catalog_titles``), a partir do template
    versionado (``PROMPT_PATH_V2``)."""
    examples = example_risks[:MAX_EXAMPLES]
    examples_text = "\n".join(_example_block(risk, lang) for risk in examples) or "(nenhum exemplo disponível)"
    titles_text = "\n".join(f"- {title}" for title in catalog_titles if title) or "(nenhum risco cadastrado ainda)"
    template = load_prompt_template_v2()
    return template.format(
        idioma=LANGUAGE_NAMES.get(lang, LANGUAGE_NAMES["pt"]),
        objeto=context.objeto,
        tipo_contratacao=context.tipo_contratacao,
        criticidade=context.criticidade,
        prazo=context.prazo,
        modalidade=context.modalidade,
        contexto=context.contexto,
        categoria=category_label_text,
        quantidade=count,
        titulos_catalogo=titles_text,
        exemplos=examples_text,
    )


_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_list(raw_text: str) -> list:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    match = _JSON_LIST_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RiskGenerationError(f"Resposta da IA não é um JSON válido: {exc}") from exc
    if not isinstance(data, list):
        raise RiskGenerationError("Resposta da IA não é uma lista JSON.")
    return data


def validate_generated_risks_list(data: list) -> list[GeneratedRisk]:
    """Valida cada item da lista como um ``GeneratedRisk`` (mesmas regras do v1)."""
    if not data:
        raise RiskGenerationError("A resposta da IA veio com uma lista vazia.")
    return [validate_generated_risk(item) for item in data]


def generate_risks_for_category(
    provider: AIProvider,
    context: ContractContext,
    category_label_text: str,
    count: int,
    catalog_titles: list[str],
    example_risks: list[RiskItem],
    lang: str = "pt",
) -> list[GeneratedRisk]:
    """Gera e valida ``count`` riscos de IA para a categoria (chamada unica);
    falhas viram ``RiskGenerationError``."""
    prompt = build_prompt_v2(context, category_label_text, count, catalog_titles, example_risks, lang=lang)
    try:
        raw = provider.generate(prompt)
    except AIProviderError as exc:
        raise RiskGenerationError(str(exc)) from exc
    return validate_generated_risks_list(_extract_json_list(raw))
