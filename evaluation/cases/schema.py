"""Formato de um caso de avaliacao.

Um caso representa uma contratacao real de TIC cuja matriz de risco final
(o "mapa real") e conhecida. O gabarito e a lista de IDs de ``data/riscos_base.csv``
presentes nesse mapa real. Ver ``evaluation/README.md`` para o formato
completo em JSON e ``evaluation/cases/synthetic/case_example_001.json`` para
um exemplo.
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
    source_path: str
    provenance_notes: str = ""

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

    return EvalCase(
        case_id=str(payload["case_id"]),
        context=context,
        gabarito_ids=tuple(str(item) for item in gabarito_ids),
        used_to_build_base=bool(payload["used_to_build_base"]),
        source_path=source_path,
        provenance_notes=str(payload.get("provenance_notes", "")),
    )
