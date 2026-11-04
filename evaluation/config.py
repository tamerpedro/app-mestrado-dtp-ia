"""Configuracao de uma execucao de avaliacao.

Tudo que afeta o resultado (sementes, parametros do BM25, modelo e endpoint
do motor de embeddings/LLM) fica aqui para que ``runner.py`` possa registrar
a configuracao completa junto do relatorio (reprodutibilidade do *protocolo*,
nao so do motor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_local_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    hostname = hostname.lower()
    return hostname in LOCAL_HOSTS or hostname.endswith(".localhost")


class LLMConfigError(ValueError):
    """Configuracao do motor LLM invalida ou insegura."""


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


# Modelo fixo e versao (revision) do motor de embeddings "de producao".
# Pin explicito a um hash de commit (nunca a uma branch como "main", que muda
# de conteudo sob o mesmo nome) para reprodutibilidade: trocar o modelo exige
# editar aqui -- e registrar a mudanca no relatorio, nunca silenciosamente.
# Hash obtido em https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# (commit em 2026-01-28); atualizar junto com a data e o motivo se o modelo for trocado.
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    model_revision: str = DEFAULT_EMBEDDING_MODEL_REVISION
    backend: str = "sentence-transformers"  # ou "hashing" (fallback local p/ testes, ver engines/embedding_engine.py)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    endpoint: str = "http://localhost:11434/v1/chat/completions"
    model: str = "llama3.1:8b-instruct"
    allow_external: bool = False
    timeout_s: float = 30.0
    temperature: float = 0.0
    max_candidates: int = 62

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.allow_external:
            return
        hostname = urlparse(self.endpoint).hostname
        if not _is_local_host(hostname):
            raise LLMConfigError(
                f"Endpoint LLM '{self.endpoint}' nao e local (host={hostname!r}). "
                "Defina allow_external=True explicitamente para permitir envio de dados "
                "a um servico fora da maquina local -- isso nunca e feito por padrao."
            )


# Metrica principal, definida a priori (antes de olhar os dados reais) para
# decidir se um motor candidato substitui o motor atual. Fixada em
# Revocacao@10: numa biblioteca curada e pequena (62 riscos), nao achar um
# risco relevante (falso negativo) e o erro mais caro -- o excesso de
# sugestoes e filtrado por revisao humana de qualquer forma. As demais
# metricas (precisao@5/@10, revocacao@5, nDCG@10, MRR) sao exploratorias:
# ajudam a entender o comportamento dos motores, mas nao decidem a troca.
PRIMARY_METRIC = "recall_at_10"
EXPLORATORY_METRICS = ("precision_at_5", "precision_at_10", "recall_at_5", "ndcg_at_10", "mrr")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20260101
    k_values: tuple[int, ...] = (5, 10)
    ndcg_k: int = 10
    reproducibility_repeats: int = 3
    bootstrap_resamples: int = 10_000
    bootstrap_confidence: float = 0.95
    primary_metric: str = PRIMARY_METRIC
    bm25: BM25Config = field(default_factory=BM25Config)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    engines: tuple[str, ...] = ("rules", "bm25", "embedding", "llm")

    def enabled_engines(self) -> tuple[str, ...]:
        if not self.llm.enabled:
            return tuple(name for name in self.engines if name != "llm")
        return self.engines
