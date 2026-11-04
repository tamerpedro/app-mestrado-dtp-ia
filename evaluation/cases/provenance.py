"""Validacao de proveniencia dos casos.

Regra do protocolo: um caso que alimentou a construcao OU a revisao/ajuste
de ``data/riscos_base.csv`` nao pode entrar na avaliacao principal. Isso vai
alem de "um risco foi copiado literalmente para a biblioteca": tambem conta
um caso usado para revisar o texto de um risco, calibrar suas
palavras-chave, calibrar seus pesos (``probabilidade_padrao`` /
``impacto_padrao``) ou calibrar os limiares do motor de regras
(``minimum_score``, limite por categoria) -- ver a docstring completa de
``EvalCase.used_to_build_base`` em ``cases/schema.py``. Qualquer uma dessas
formas de influencia vazaria o gabarito do caso para dentro dos motores, por
isso o caso e sempre excluido da avaliacao principal. Esses casos ficam
disponiveis separadamente para inspecao, mas nunca entram no calculo de
metricas principal.

Alem do controle de vazamento, este modulo tambem verifica, de forma
informativa (sem bloquear a avaliacao), se cada caso segue as boas praticas
de anotacao do protocolo (ver ``evaluation/README.md``):

- o ``context`` deveria vir de documentos anteriores ao mapa de riscos
  final (``context_predates_risk_map``);
- o gabarito deveria ter sido anotado por pelo menos duas pessoas,
  independentemente e sem acesso a saida de nenhum motor
  (``annotator_count``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import RiskItem

from .schema import EvalCase

RECOMMENDED_MIN_ANNOTATORS = 2


@dataclass(frozen=True)
class ProvenanceReport:
    main_eval_cases: tuple[EvalCase, ...]
    excluded_leaked_cases: tuple[EvalCase, ...]
    cases_with_unknown_gabarito_ids: dict[str, tuple[str, ...]]
    methodology_warnings: dict[str, tuple[str, ...]]
    """Avisos informativos (nao bloqueiam a avaliacao) sobre boas praticas de
    anotacao nao seguidas ou nao informadas para um caso -- ver o docstring
    do modulo."""

    @property
    def total_cases(self) -> int:
        return len(self.main_eval_cases) + len(self.excluded_leaked_cases)


def _methodology_warnings_for_case(case: EvalCase) -> tuple[str, ...]:
    warnings: list[str] = []
    if case.context_predates_risk_map is None:
        warnings.append(
            "context_predates_risk_map nao informado: confirme que o contexto vem de "
            "documentos anteriores ao mapa de riscos final."
        )
    elif case.context_predates_risk_map is False:
        warnings.append(
            "context_predates_risk_map=false: o contexto NAO vem de documentos anteriores "
            "ao mapa de riscos -- risco de vazamento por retrospectiva (hindsight bias)."
        )

    if case.annotator_count is None:
        warnings.append(
            "annotator_count nao informado: confirme que o gabarito foi anotado sem acesso "
            "as saidas dos motores, preferencialmente por 2 ou mais anotadores independentes."
        )
    elif case.annotator_count < RECOMMENDED_MIN_ANNOTATORS:
        warnings.append(
            f"annotator_count={case.annotator_count}: recomendado >= {RECOMMENDED_MIN_ANNOTATORS} "
            "anotadores independentes para permitir checar concordancia."
        )
    return tuple(warnings)


def split_by_provenance(cases: list[EvalCase], risks: list[RiskItem]) -> ProvenanceReport:
    """Separa casos usados na construcao/revisao da base (excluidos) dos
    demais (avaliacao principal); reporta IDs de gabarito desconhecidos e
    avisos de metodologia de anotacao (ver docstring do modulo).
    """
    known_ids = {risk.id for risk in risks}

    main_eval: list[EvalCase] = []
    excluded: list[EvalCase] = []
    unknown_gabarito: dict[str, tuple[str, ...]] = {}
    methodology_warnings: dict[str, tuple[str, ...]] = {}

    for case in cases:
        if case.used_to_build_base:
            excluded.append(case)
        else:
            main_eval.append(case)

        unknown = tuple(risk_id for risk_id in case.gabarito_ids if risk_id not in known_ids)
        if unknown:
            unknown_gabarito[case.case_id] = unknown

        warnings = _methodology_warnings_for_case(case)
        if warnings:
            methodology_warnings[case.case_id] = warnings

    return ProvenanceReport(
        main_eval_cases=tuple(main_eval),
        excluded_leaked_cases=tuple(excluded),
        cases_with_unknown_gabarito_ids=unknown_gabarito,
        methodology_warnings=methodology_warnings,
    )
