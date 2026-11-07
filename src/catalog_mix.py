"""Lista cega catalogo+IA por categoria, para a avaliacao (sem pareamento).

Em cada categoria, mostra todos os riscos do catalogo cujo
``tipo_contratacao`` inclui o tipo informado (so um filtro, sem pontuacao
nem ranqueamento) misturados a ``min(3, n)`` riscos gerados por IA (``n`` =
total de riscos do catalogo na categoria, contado antes do filtro por
tipo). A ordem dentro da categoria e sorteada com uma semente; cada item
recebe um ID neutro (S01, S02, ...) sem nenhuma pista de origem.

A geracao de IA (``generate_ai_risks_for_all_categories``) e separada da
montagem da lista (``build_category_mix``) para permitir cache por caso
compartilhado (``src.case_store``): quando ja existe um cache, o chamador
passa os riscos ja gerados direto para ``build_category_mix``, sem chamar
a IA de novo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ai_provider import AIProvider
from .domain import CATEGORIES, DEFAULT_STRATEGY, category_label
from .models import ActionItem, ContractContext, MatrixRow, RiskItem
from .risk_generator import GeneratedRisk, RiskGenerationError, generate_risks_for_category
from .scoring import risk_level

ORIGIN_CATALOG = "catalogo"
ORIGIN_AI = "ia"

MAX_AI_PER_CATEGORY = 3
MAX_STYLE_EXAMPLES = 3


@dataclass(frozen=True)
class BlindItem:
    """Um item exibido na lista cega; ``origin`` nunca aparece na UI."""

    slot_id: str
    origin: str
    categoria: str
    posicao: int
    row: MatrixRow


@dataclass(frozen=True)
class CategoryItems:
    categoria: str
    items: list[BlindItem]


@dataclass(frozen=True)
class MixResult:
    categories: list[CategoryItems]
    seed: int


def ai_count_for_category(total_in_category: int, max_count: int = MAX_AI_PER_CATEGORY) -> int:
    """``min(3, n)``: quantos riscos de IA gerar, ``n`` = total do catalogo
    na categoria (antes do filtro por tipo de contratacao)."""
    return min(max_count, max(0, total_in_category))


def filter_catalog_by_contract_type(risks: list[RiskItem], tipo_contratacao: str) -> list[RiskItem]:
    tipo = (tipo_contratacao or "").strip().lower()
    return [risk for risk in risks if tipo in risk.tipo_contratacao]


def _catalog_risk_to_row(risk: RiskItem, lang: str) -> MatrixRow:
    return MatrixRow(
        id=risk.id,
        risco=risk.text("titulo", lang),
        categoria=risk.categoria,
        causa=risk.text("causa", lang),
        consequencias=[risk.text("consequencia", lang)],
        probabilidade=risk.probabilidade_padrao,
        impacto=risk.impacto_padrao,
        nivel=risk_level(risk.probabilidade_padrao, risk.impacto_padrao),
        estrategia=DEFAULT_STRATEGY,
        acoes_preventivas=[ActionItem(risk.text("acao_preventiva", lang))],
        acoes_contingencia=[ActionItem(risk.text("acao_contingencia", lang))],
        tags=risk.tipo_contratacao,
    )


def _generated_risk_to_row(generated: GeneratedRisk, categoria: str) -> MatrixRow:
    return MatrixRow(
        id="",
        risco=generated.titulo,
        categoria=categoria,
        causa=generated.causa,
        consequencias=[generated.consequencia],
        probabilidade=generated.probabilidade_padrao,
        impacto=generated.impacto_padrao,
        nivel=risk_level(generated.probabilidade_padrao, generated.impacto_padrao),
        estrategia=DEFAULT_STRATEGY,
        acoes_preventivas=[ActionItem(generated.acao_preventiva)],
        acoes_contingencia=[ActionItem(generated.acao_contingencia)],
        tags=["ia"],
    )


def generate_ai_risks_for_all_categories(
    risks: list[RiskItem],
    context: ContractContext,
    provider: AIProvider,
    *,
    lang: str = "pt",
    on_ai_failure=None,
) -> dict[str, list[GeneratedRisk]]:
    """1 chamada de IA por categoria (``min(3, n)`` riscos); falha numa
    categoria so afeta essa categoria (segue so com o catalogo dela)."""
    catalog_titles_all = [risk.text("titulo", lang) for risk in risks]
    result: dict[str, list[GeneratedRisk]] = {}
    for categoria in CATEGORIES:
        risks_in_category = [risk for risk in risks if risk.categoria == categoria]
        count = ai_count_for_category(len(risks_in_category))
        if count == 0:
            result[categoria] = []
            continue
        examples = risks_in_category[:MAX_STYLE_EXAMPLES]
        try:
            result[categoria] = generate_risks_for_category(
                provider,
                context,
                category_label(categoria, lang),
                count,
                catalog_titles_all,
                examples,
                lang=lang,
            )
        except RiskGenerationError as exc:
            if on_ai_failure is not None:
                on_ai_failure(categoria, str(exc))
            result[categoria] = []
    return result


def build_category_mix(
    risks: list[RiskItem],
    context: ContractContext,
    ai_risks_by_category: dict[str, list[GeneratedRisk]],
    *,
    lang: str = "pt",
    seed: int | None = None,
) -> MixResult:
    """Monta a lista cega por categoria: catalogo filtrado + riscos de IA ja
    gerados (ou vindos do cache de um caso), misturados e sorteados."""
    rng_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(rng_seed)

    slot_counter = 1
    categories: list[CategoryItems] = []

    for categoria in CATEGORIES:
        risks_in_category = [risk for risk in risks if risk.categoria == categoria]
        filtered = filter_catalog_by_contract_type(risks_in_category, context.tipo_contratacao)
        generated_list = ai_risks_by_category.get(categoria, [])

        entries: list[tuple[str, MatrixRow]] = [(ORIGIN_CATALOG, _catalog_risk_to_row(risk, lang)) for risk in filtered]
        entries += [(ORIGIN_AI, _generated_risk_to_row(generated, categoria)) for generated in generated_list]

        if not entries:
            continue

        rng.shuffle(entries)
        items = []
        for offset, (origin, row) in enumerate(entries):
            items.append(
                BlindItem(
                    slot_id=f"S{slot_counter + offset:02d}",
                    origin=origin,
                    categoria=categoria,
                    posicao=offset + 1,
                    row=row,
                )
            )
        slot_counter += len(entries)
        categories.append(CategoryItems(categoria=categoria, items=items))

    return MixResult(categories=categories, seed=rng_seed)
