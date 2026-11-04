"""Formato de um caso de avaliacao.

Um caso representa uma contratacao real de TIC cuja matriz de risco final
(o "mapa real") e conhecida. O gabarito e a lista de IDs de ``data/riscos_base.csv``
presentes nesse mapa real. Ver ``evaluation/README.md`` para o formato
completo em JSON e ``evaluation/cases/synthetic/case_example_001.json`` para
um exemplo.

Protocolo de proveniencia (``used_to_build_base``) e de anotacao
(``context_predates_risk_map`` / ``annotator_count``): ver docstrings dos
campos abaixo e ``evaluation/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import ContractContext

REQUIRED_CONTEXT_FIELDS = (
    "objeto",
    "tipo_contratacao",
    "valor_estimado",
    "criticidade",
    "prazo",
    "modalidade",
    "contexto",
)


class CaseFormatError(ValueError):
    """Caso de avaliacao malformado (JSON ausente de campos obrigatorios etc.)."""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    context: ContractContext
    gabarito_ids: tuple[str, ...]
    used_to_build_base: bool
    """``True`` se este caso alimentou a construcao OU a revisao/ajuste da
    biblioteca de riscos de QUALQUER forma -- nao so "um risco foi copiado
    literalmente para ``data/riscos_base.csv``". Em particular, marque
    ``True`` tambem quando o caso foi usado para:

    - revisar ou ajustar o *conteudo* de um risco existente (titulo, causa,
      consequencia, acoes) na biblioteca;
    - calibrar ou revisar as *palavras-chave* (``palavras_chave``) de um
      ou mais riscos;
    - calibrar os *pesos* (``probabilidade_padrao`` / ``impacto_padrao``)
      de um ou mais riscos;
    - calibrar os *limiares* usados pelo motor de regras (``minimum_score``,
      ``MAX_SUGGESTIONS_PER_CATEGORY`` em ``src/suggestions.py``) ou
      quaisquer parametros equivalentes de outro motor (ex.: ``k1``/``b``
      do BM25, se algum dia forem ajustados a partir de casos reais).

    Qualquer uma dessas formas de influencia faz o caso vazar informacao do
    seu proprio gabarito para dentro dos motores -- por isso ele e sempre
    excluido da avaliacao principal (``cases/provenance.py``), mesmo que
    nenhum risco tenha sido copiado literalmente.
    """
    source_path: str
    provenance_notes: str = ""
    context_predates_risk_map: bool | None = None
    """``True`` se o ``context`` foi extraido de documentos anteriores ao
    mapa de riscos final (ex.: termo de referencia, estudo tecnico
    preliminar, edital) -- nunca do proprio mapa de riscos ou de material
    produzido depois dele. ``None`` quando nao informado (casos antigos);
    ver aviso em ``cases/provenance.py``."""
    annotator_count: int | None = None
    """Quantas pessoas anotaram o gabarito de forma independente, sem acesso
    as sugestoes de nenhum motor (anotacao as cegas). Preferencialmente
    ``>= 2``, para permitir checar concordancia entre anotadores. ``None``
    quando nao informado."""

    @property
    def gabarito_set(self) -> frozenset[str]:
        return frozenset(self.gabarito_ids)


def case_from_dict(payload: dict, *, source_path: str = "<memoria>") -> EvalCase:
    missing_top = [field for field in ("case_id", "context", "gabarito_ids", "used_to_build_base") if field not in payload]
    if missing_top:
        raise CaseFormatError(f"{source_path}: campos obrigatorios ausentes: {', '.join(missing_top)}")

    context_payload = payload["context"]
    missing_context = [field for field in REQUIRED_CONTEXT_FIELDS if field not in context_payload]
    if missing_context:
        raise CaseFormatError(
            f"{source_path}: campos de ContractContext ausentes em 'context': {', '.join(missing_context)}"
        )

    gabarito_ids = payload["gabarito_ids"]
    if not isinstance(gabarito_ids, list) or not gabarito_ids:
        raise CaseFormatError(f"{source_path}: 'gabarito_ids' deve ser uma lista nao vazia de IDs de risco.")

    context = ContractContext(
        objeto=str(context_payload["objeto"]),
        tipo_contratacao=str(context_payload["tipo_contratacao"]),
        valor_estimado=float(context_payload["valor_estimado"]),
        criticidade=str(context_payload["criticidade"]),
        prazo=str(context_payload["prazo"]),
        modalidade=str(context_payload["modalidade"]),
        contexto=str(context_payload["contexto"]),
    )

    context_predates_risk_map = payload.get("context_predates_risk_map")
    annotator_count = payload.get("annotator_count")

    return EvalCase(
        case_id=str(payload["case_id"]),
        context=context,
        gabarito_ids=tuple(str(item) for item in gabarito_ids),
        used_to_build_base=bool(payload["used_to_build_base"]),
        source_path=source_path,
        provenance_notes=str(payload.get("provenance_notes", "")),
        context_predates_risk_map=None if context_predates_risk_map is None else bool(context_predates_risk_map),
        annotator_count=None if annotator_count is None else int(annotator_count),
    )
