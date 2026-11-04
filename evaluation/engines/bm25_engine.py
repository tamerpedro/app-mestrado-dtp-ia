"""Motor BM25 (Okapi), determinístico e auditável.

Implementacao propria em Python puro (sem dependencia externa) para manter
o motor facil de inspecionar linha a linha. Formula padrao do BM25 Okapi
(Robertson & Zaragoza), com ``k1`` e ``b`` vindos de ``config.BM25Config``
e registrados no relatorio da execucao.
"""

from __future__ import annotations

import math

from src.models import ContractContext, RiskItem

from ..config import BM25Config
from ..text_utils import context_query_text, risk_document_text, tokenize
from .base import RankResult

name = "bm25"


class BM25Index:
    """Indice BM25 sobre um corpus fixo de documentos (um por risco)."""

    def __init__(self, documents: list[list[str]], config: BM25Config) -> None:
        self.config = config
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avg_doc_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for doc in documents:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.term_frequencies.append(counts)
            for term in counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query_terms: list[str], doc_index: int) -> float:
        k1, b = self.config.k1, self.config.b
        doc_length = self.doc_lengths[doc_index]
        counts = self.term_frequencies[doc_index]
        total = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = term_frequency + k1 * (1 - b + b * doc_length / (self.avg_doc_length or 1.0))
            total += idf * (term_frequency * (k1 + 1)) / denominator
        return total


def rank(context: ContractContext, risks: list[RiskItem], config: BM25Config | None = None) -> RankResult:
    config = config or BM25Config()
    documents = [tokenize(risk_document_text(risk)) for risk in risks]
    index = BM25Index(documents, config)
    query_terms = tokenize(context_query_text(context))

    scored = [(index.score(query_terms, position), risk) for position, risk in enumerate(risks)]
    scored.sort(key=lambda item: (item[0], item[1].categoria, item[1].id), reverse=True)
    return RankResult(ranked_ids=tuple(risk.id for _score, risk in scored))
