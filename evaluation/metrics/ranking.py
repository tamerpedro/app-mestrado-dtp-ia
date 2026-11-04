"""Metricas de qualidade de ranking: precisao@k, revocacao@k, nDCG@k, MRR.

Relevancia binaria: um risco e relevante se o seu ID esta no gabarito do
caso. Todas as funcoes aceitam ``predicted_ids`` ja ordenados por
relevancia decrescente (como devolvido por ``engines.*.rank``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(predicted_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Divide sempre por ``k`` (definicao padrao), nao pelo tamanho da lista
    devolvida. Isso importa para motores que podem devolver menos de ``k``
    itens (ex.: `rules_production`, com corte por `minimum_score`): devolver
    2 itens relevantes entre 2 e contar como precisao@10 = 1.0 inflaria
    artificialmente a metrica de um motor que decidiu sugerir pouco."""
    hits = sum(1 for risk_id in predicted_ids[:k] if risk_id in relevant_ids)
    return hits / k


def recall_at_k(predicted_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    top_k = predicted_ids[:k]
    hits = sum(1 for risk_id in top_k if risk_id in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(predicted_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    total = 0.0
    for position, risk_id in enumerate(predicted_ids[:k], start=1):
        if risk_id in relevant_ids:
            total += 1.0 / math.log2(position + 1)
    return total


def ndcg_at_k(predicted_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return None
    return dcg_at_k(predicted_ids, relevant_ids, k) / ideal_dcg


def mean_reciprocal_rank(predicted_ids: Sequence[str], relevant_ids: frozenset[str]) -> float:
    for position, risk_id in enumerate(predicted_ids, start=1):
        if risk_id in relevant_ids:
            return 1.0 / position
    return 0.0
