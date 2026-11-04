"""Motor atual de regras, avaliado como somente-leitura.

Nao reimplementa nem modifica ``src/suggestions.py``. Para obter um ranking
completo e comparavel aos demais motores, usa ``suggestion_score`` (a mesma
funcao de pontuacao do app) sobre todos os riscos e ordena com o mesmo
critério de ``suggest_risks`` -- (score, categoria, id), decrescente --
sem aplicar ``minimum_score`` nem o limite por categoria, que no app sao
politicas de apresentacao (quantas sugestoes mostrar), nao parte da
qualidade do ranking que esta sendo medida aqui.

``evaluation/README.md`` documenta essa escolha e como obter, em paralelo,
os numeros "como o usuario realmente ve" (com os cortes de producao).
"""

from __future__ import annotations

from src.models import ContractContext, RiskItem
from src.suggestions import MAX_SUGGESTIONS_PER_CATEGORY, suggest_risks, suggestion_score

from .base import RankResult

name = "rules"


def rank(context: ContractContext, risks: list[RiskItem]) -> RankResult:
    ranked = sorted(
        ((suggestion_score(risk, context), risk) for risk in risks),
        key=lambda item: (item[0], item[1].categoria, item[1].id),
        reverse=True,
    )
    ranked_ids = tuple(risk.id for _score, risk in ranked)
    return RankResult(ranked_ids=ranked_ids)


def rank_as_shown_in_production(
    context: ContractContext,
    risks: list[RiskItem],
    minimum_score: int = 2,
    max_per_category: int | None = MAX_SUGGESTIONS_PER_CATEGORY,
) -> RankResult:
    """Ranking identico ao que ``app.py`` mostra ao usuario (com os cortes)."""
    rows = suggest_risks(risks, context, minimum_score=minimum_score, max_per_category=max_per_category)
    return RankResult(ranked_ids=tuple(row.id for row in rows))
