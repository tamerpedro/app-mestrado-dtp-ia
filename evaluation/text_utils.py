"""Normalizacao e tokenizacao de texto usadas pelo BM25 e pelos embeddings.

Deliberadamente independente de ``src/suggestions.py``: a avaliacao nao deve
depender de funcoes internas do app (que podem mudar sem aviso e nao fazem
parte da interface publica). A duplicacao da normalizacao (minusculas, sem
acento, tokens alfanumericos) e pequena e intencional -- mantem cada motor
auditavel isoladamente.
"""

from __future__ import annotations

import re
import unicodedata

from src.models import ContractContext, RiskItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_accents.lower()


def tokenize(value: str | None, min_length: int = 2) -> list[str]:
    return [token for token in _TOKEN_RE.findall(normalize_text(value)) if len(token) >= min_length]


def risk_document_text(risk: RiskItem) -> str:
    """Texto do risco usado como 'documento' pelo BM25 e pelos embeddings."""
    return " ".join(
        [
            risk.titulo,
            risk.causa,
            risk.consequencia,
            " ".join(risk.palavras_chave),
            " ".join(risk.tipo_contratacao),
            risk.categoria,
        ]
    )


def context_query_text(context: ContractContext) -> str:
    """Texto da contratacao usado como 'consulta' pelo BM25 e pelos embeddings.

    Usa apenas campos de ``ContractContext`` (nunca o gabarito).
    """
    return " ".join(
        [
            context.objeto,
            context.contexto,
            context.tipo_contratacao,
            context.criticidade,
            context.modalidade,
            context.prazo,
        ]
    )
