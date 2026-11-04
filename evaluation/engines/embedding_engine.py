"""Motor de embeddings, com modelo fixo e execucao local.

O backend "de producao" (``SentenceTransformerBackend``) usa um modelo
fixo (nome + revision pinados em ``config.EmbeddingConfig``) carregado e
executado localmente via ``sentence-transformers`` -- nenhum texto sai da
maquina. E uma dependencia opcional (``evaluation/requirements-embeddings.txt``):
so e importada quando o backend e de fato instanciado.

``HashingEmbeddingBackend`` e um backend puramente local e determinístico
(hashing de tokens, sem download de modelo) usado pelos testes automatizados
e para desenvolvimento offline. Ele NAO produz embeddings semanticos reais
e nao deve ser usado para gerar os numeros finais da avaliacao -- isso fica
documentado tambem em ``evaluation/README.md``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from src.models import ContractContext, RiskItem

from ..config import EmbeddingConfig
from ..text_utils import context_query_text, risk_document_text, tokenize
from .base import RankResult

name = "embedding"

Vector = tuple[float, ...]


class EmbeddingBackend(Protocol):
    label: str

    def encode(self, texts: list[str]) -> list[Vector]:
        ...


class HashingEmbeddingBackend:
    """Backend local, deterministico, sem dependencias -- so para testes/offline.

    Projeta tokens em um espaco de dimensao fixa via hash estavel (SHA-256),
    com sinal determinado pelo bit menos significativo do hash (hashing trick
    classico). Nao captura semantica; serve apenas para exercitar a logica de
    ranking por similaridade de cosseno nos testes, sem baixar nenhum modelo.
    """

    label = "hashing-fallback-local-v1"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector_for_text(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return tuple(vector)

    def encode(self, texts: list[str]) -> list[Vector]:
        return [self._vector_for_text(text) for text in texts]


class SentenceTransformerBackend:
    """Backend de producao: modelo fixo, execucao local, via sentence-transformers."""

    def __init__(self, model_name: str, model_revision: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "sentence-transformers nao esta instalado. Instale com "
                "'pip install -r evaluation/requirements-embeddings.txt' para rodar o motor "
                "de embeddings de producao (ou use HashingEmbeddingBackend para testes)."
            ) from exc

        self.model_name = model_name
        self.model_revision = model_revision
        self.label = f"{model_name}@{model_revision}"
        self._model = SentenceTransformer(model_name, revision=model_revision)

    def encode(self, texts: list[str]) -> list[Vector]:
        embeddings = self._model.encode(texts, normalize_embeddings=False)
        return [tuple(float(value) for value in row) for row in embeddings]


def _cosine_similarity(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_default_backend(config: EmbeddingConfig | None = None) -> EmbeddingBackend:
    config = config or EmbeddingConfig()
    if config.backend == "hashing":
        return HashingEmbeddingBackend()
    return SentenceTransformerBackend(config.model_name, config.model_revision)


def rank(
    context: ContractContext,
    risks: list[RiskItem],
    backend: EmbeddingBackend,
) -> RankResult:
    document_texts = [risk_document_text(risk) for risk in risks]
    query_text = context_query_text(context)

    document_vectors = backend.encode(document_texts)
    query_vector = backend.encode([query_text])[0]

    scored = [
        (_cosine_similarity(query_vector, doc_vector), risk)
        for doc_vector, risk in zip(document_vectors, risks)
    ]
    scored.sort(key=lambda item: (item[0], item[1].categoria, item[1].id), reverse=True)
    return RankResult(ranked_ids=tuple(risk.id for _score, risk in scored))
