"""Interface comum aos motores de sugestao avaliados.

Cada motor recebe apenas os campos de ``ContractContext`` e a lista de
riscos da biblioteca, e devolve um ranking completo (todos os IDs recebidos,
ordenados por relevancia decrescente, sem corte por ``minimum_score`` ou por
categoria). O corte em top-k e feito depois, nas metricas -- assim
precisao@5, @10 etc. sao comparaveis entre motores que tem politicas de
corte internas diferentes (ou nenhuma).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.models import ContractContext, RiskItem


@dataclass(frozen=True)
class RankResult:
    ranked_ids: tuple[str, ...]
    raw_ids_before_validation: tuple[str, ...] = field(default=())
    """IDs devolvidos pelo motor antes de filtrar os invalidos (so difere de
    ``ranked_ids`` para o motor LLM; usado para calcular a taxa de IDs
    invalidos). Para os demais motores, igual a ``ranked_ids``."""

    def __post_init__(self) -> None:
        if not self.raw_ids_before_validation:
            object.__setattr__(self, "raw_ids_before_validation", self.ranked_ids)


class SuggestionEngine(Protocol):
    name: str

    def rank(self, context: ContractContext, risks: list[RiskItem]) -> RankResult:
        ...
