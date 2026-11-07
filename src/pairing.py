"""Pareamento 1 biblioteca + 1 IA por categoria, para a avaliacao cega.

Para cada categoria com ao menos 1 risco na biblioteca, entra o de maior
``suggestion_score`` (sem alterar a logica de pontuacao) - mesmo abaixo do
``minimum_score`` atual, inclusive com pontuacao 0: toda categoria com
risco na biblioteca forma um par. ``sem_par`` so ocorre quando a categoria
nao tem nenhum risco na biblioteca. A IA gera 1 risco "diferente" a partir
do risco de biblioteca escolhido e de 2-3 exemplos de estilo. A posicao de
cada sugestao dentro do par e sorteada com uma semente registrada no log, e
cada sugestao recebe um ID neutro (S01, S02, ...) sem nenhuma pista de
origem - a pontuacao e o corte (``abaixo_corte``) ficam so no log, nunca na
interface.
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
    """Uma sugestao exibida na avaliacao cega; nada disso aparece na UI alem

    do ``row`` (texto) e do ``slot_id`` (neutro). ``pontuacao``/``abaixo_corte``
    so existem para sugestoes de biblioteca (``origin == ORIGIN_LIBRARY``);
    ficam ``None`` para sugestoes de IA.
    """

    slot_id: str
    origin: str
    categoria: str
    row: MatrixRow
    pontuacao: int | None = None
    abaixo_corte: bool | None = None


@dataclass(frozen=True)
class CategoryPair:
    categoria: str
    suggestions: list[BlindSuggestion]
    sem_par: bool


@dataclass(frozen=True)
class PairingResult:
    pairs: list[CategoryPair]
    seed: int


@dataclass(frozen=True)
class LibraryPick:
    risk: RiskItem
    pontuacao: int
    abaixo_corte: bool


def top_library_risk_per_category(
    risks: list[RiskItem],
    context: ContractContext,
    minimum_score: int,
) -> dict[str, LibraryPick]:
    """O risco de maior ``suggestion_score`` por categoria, mesmo abaixo do corte.

    Toda categoria com ao menos 1 risco na biblioteca entra aqui (mesmo com
    pontuacao 0); ``abaixo_corte`` indica se ficou abaixo de ``minimum_score``
    (usado so para o log e para a consolidacao, nunca exibido na interface).
    Usa a mesma pontuacao de ``src.suggestions`` (nao reimplementada aqui).
    """
    ranked = sorted(
        ((suggestion_score(risk, context), risk) for risk in risks),
        key=lambda item: (item[0], item[1].categoria, item[1].id),
        reverse=True,
    )
    chosen: dict[str, LibraryPick] = {}
    for score, risk in ranked:
        if risk.categoria in chosen:
            continue
        chosen[risk.categoria] = LibraryPick(risk=risk, pontuacao=score, abaixo_corte=score < minimum_score)
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
    """Monta, para cada categoria, o par biblioteca+IA.

    ``sem_par`` so acontece quando a categoria nao tem nenhum risco na
    biblioteca; nesse caso so a sugestao de IA (se gerada) aparece.
    """
    rng_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(rng_seed)

    library_by_category = top_library_risk_per_category(risks, context, minimum_score)
    slot_counter = 1
    pairs: list[CategoryPair] = []

    for category in CATEGORIES:
        pick = library_by_category.get(category)
        library_risk = pick.risk if pick is not None else None
        ai_row = _generate_ai_row(provider, context, category, library_risk, risks, lang, on_ai_failure)

        rows_with_origin: list[tuple[str, MatrixRow, int | None, bool | None]] = []
        if pick is not None:
            rows_with_origin.append(
                (ORIGIN_LIBRARY, _library_risk_to_row(pick.risk, lang), pick.pontuacao, pick.abaixo_corte)
            )
        if ai_row is not None:
            rows_with_origin.append((ORIGIN_AI, ai_row, None, None))

        if not rows_with_origin:
            continue

        rng.shuffle(rows_with_origin)
        suggestions = []
        for origin, row, pontuacao, abaixo_corte in rows_with_origin:
            suggestions.append(
                BlindSuggestion(
                    slot_id=f"S{slot_counter:02d}",
                    origin=origin,
                    categoria=category,
                    row=row,
                    pontuacao=pontuacao,
                    abaixo_corte=abaixo_corte,
                )
            )
            slot_counter += 1

        pairs.append(CategoryPair(categoria=category, suggestions=suggestions, sem_par=pick is None))

    return PairingResult(pairs=pairs, seed=rng_seed)
