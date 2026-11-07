"""Pareamento 1 biblioteca + 1 IA por categoria, para a avaliacao cega.

Para cada categoria, a biblioteca entra com o risco de maior pontuacao
(``suggestion_score`` com o ``minimum_score`` atual, sem alterar a logica de
pontuacao); a IA gera 1 risco "diferente" a partir desse risco e de 2-3
exemplos de estilo. Categorias sem risco de biblioteca acima do corte
mostram so o risco de IA e sao marcadas ``sem_par``. A posicao de cada
sugestao dentro do par e sorteada com uma semente registrada no log, e cada
sugestao recebe um ID neutro (S01, S02, ...) sem nenhuma pista de origem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ai_provider import AIProvider
from .domain import CATEGORIES, DEFAULT_STRATEGY, category_label
from .models import ActionItem, ContractContext, MatrixRow, RiskItem
from .risk_generator import GeneratedRisk, RiskGenerationError, generate_risk_for_category
from .scoring import risk_level
from .suggestions import suggestion_score

ORIGIN_LIBRARY = "biblioteca"
ORIGIN_AI = "ia"

MAX_STYLE_EXAMPLES = 3


class AIFailureCallback:
    """Tipagem informal: ``callback(categoria: str, erro: str) -> None``."""


@dataclass(frozen=True)
class BlindSuggestion:
    """Uma sugestao exibida na avaliacao cega; ``origin`` nunca aparece na UI."""

    slot_id: str
    origin: str
    categoria: str
    row: MatrixRow


@dataclass(frozen=True)
class CategoryPair:
    categoria: str
    suggestions: list[BlindSuggestion]
    sem_par: bool


@dataclass(frozen=True)
class PairingResult:
    pairs: list[CategoryPair]
    seed: int


def top_library_risk_per_category(
    risks: list[RiskItem],
    context: ContractContext,
    minimum_score: int,
) -> dict[str, RiskItem]:
    """O risco de maior ``suggestion_score`` por categoria, acima do corte.

    Usa a mesma pontuacao de ``src.suggestions`` (nao reimplementada aqui);
    serve para montar o par e o prompt, nao para a lista de sugestoes da
    interface nao cega.
    """
    ranked = sorted(
        ((suggestion_score(risk, context), risk) for risk in risks),
        key=lambda item: (item[0], item[1].categoria, item[1].id),
        reverse=True,
    )
    chosen: dict[str, RiskItem] = {}
    for score, risk in ranked:
        if score < minimum_score:
            continue
        chosen.setdefault(risk.categoria, risk)
    return chosen


def _style_examples(risks: list[RiskItem], category: str, exclude_id: str, count: int = MAX_STYLE_EXAMPLES) -> list[RiskItem]:
    same_category = [risk for risk in risks if risk.categoria == category and risk.id != exclude_id]
    others = [risk for risk in risks if risk.categoria != category and risk.id != exclude_id]
    return [*same_category, *others][:count]


def _library_risk_to_row(risk: RiskItem, lang: str) -> MatrixRow:
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


def _generate_ai_row(
    provider: AIProvider,
    context: ContractContext,
    category: str,
    library_risk: RiskItem | None,
    risks: list[RiskItem],
    lang: str,
    on_ai_failure,
) -> MatrixRow | None:
    exclude_id = library_risk.id if library_risk is not None else ""
    examples = _style_examples(risks, category, exclude_id)
    if library_risk is None and not examples:
        return None
    try:
        generated = generate_risk_for_category(
            provider,
            context,
            category_label(category, lang),
            library_risk,
            examples,
            lang=lang,
        )
    except RiskGenerationError as exc:
        if on_ai_failure is not None:
            on_ai_failure(category, str(exc))
        return None
    return _generated_risk_to_row(generated, category)


def build_category_pairs(
    risks: list[RiskItem],
    context: ContractContext,
    provider: AIProvider,
    *,
    minimum_score: int = 2,
    lang: str = "pt",
    seed: int | None = None,
    on_ai_failure=None,
) -> PairingResult:
    """Monta, para cada categoria, o par biblioteca+IA (ou so IA, caso ``sem_par``)."""
    rng_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(rng_seed)

    library_by_category = top_library_risk_per_category(risks, context, minimum_score)
    slot_counter = 1
    pairs: list[CategoryPair] = []

    for category in CATEGORIES:
        library_risk = library_by_category.get(category)
        ai_row = _generate_ai_row(provider, context, category, library_risk, risks, lang, on_ai_failure)

        rows_with_origin: list[tuple[str, MatrixRow]] = []
        if library_risk is not None:
            rows_with_origin.append((ORIGIN_LIBRARY, _library_risk_to_row(library_risk, lang)))
        if ai_row is not None:
            rows_with_origin.append((ORIGIN_AI, ai_row))

        if not rows_with_origin:
            continue

        rng.shuffle(rows_with_origin)
        suggestions = []
        for origin, row in rows_with_origin:
            suggestions.append(BlindSuggestion(slot_id=f"S{slot_counter:02d}", origin=origin, categoria=category, row=row))
            slot_counter += 1

        pairs.append(CategoryPair(categoria=category, suggestions=suggestions, sem_par=library_risk is None))

    return PairingResult(pairs=pairs, seed=rng_seed)
