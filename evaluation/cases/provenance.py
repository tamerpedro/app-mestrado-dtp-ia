"""Validacao de proveniencia dos casos.

Regra do protocolo: um caso que alimentou a construcao de ``data/riscos_base.csv``
(por exemplo, porque um de seus riscos foi copiado para a biblioteca) nao pode
entrar na avaliacao principal -- isso seria vazamento de dados (o motor de
regras e o BM25/embeddings "aprenderiam", via a biblioteca, o proprio
gabarito do caso). Esses casos ficam disponiveis separadamente para
inspecao, mas sao sempre excluidos do calculo de metricas principal.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import RiskItem

from .schema import EvalCase


@dataclass(frozen=True)
class ProvenanceReport:
    main_eval_cases: tuple[EvalCase, ...]
    excluded_leaked_cases: tuple[EvalCase, ...]
    cases_with_unknown_gabarito_ids: dict[str, tuple[str, ...]]

    @property
    def total_cases(self) -> int:
        return len(self.main_eval_cases) + len(self.excluded_leaked_cases)


def split_by_provenance(cases: list[EvalCase], risks: list[RiskItem]) -> ProvenanceReport:
    """Separa casos usados na construcao da base (excluidos) dos demais (avaliacao principal).

    Tambem reporta, por caso, quais IDs do gabarito nao existem na biblioteca
    carregada -- esses riscos nunca poderao ser recuperados por nenhum motor
    (o teto de recall do caso e < 1), o que e informativo e nao um erro.
    """
    known_ids = {risk.id for risk in risks}

    main_eval: list[EvalCase] = []
    excluded: list[EvalCase] = []
    unknown_gabarito: dict[str, tuple[str, ...]] = {}

    for case in cases:
        if case.used_to_build_base:
            excluded.append(case)
        else:
            main_eval.append(case)

        unknown = tuple(risk_id for risk_id in case.gabarito_ids if risk_id not in known_ids)
        if unknown:
            unknown_gabarito[case.case_id] = unknown

    return ProvenanceReport(
        main_eval_cases=tuple(main_eval),
        excluded_leaked_cases=tuple(excluded),
        cases_with_unknown_gabarito_ids=unknown_gabarito,
    )
