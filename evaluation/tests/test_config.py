import re

from evaluation.config import (
    DEFAULT_EMBEDDING_MODEL_REVISION,
    EXPLORATORY_METRICS,
    PRIMARY_METRIC,
    LLMConfig,
    RunConfig,
)

_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def test_embedding_revision_is_pinned_to_a_commit_hash_not_a_branch():
    """Uma branch (ex. "main") pode mudar de conteudo sob o mesmo nome -- o
    motor de embeddings de producao precisa de um hash de commit fixo para
    ser reprodutivel (ver evaluation/config.py)."""
    assert _COMMIT_HASH_RE.match(DEFAULT_EMBEDDING_MODEL_REVISION)
    assert DEFAULT_EMBEDDING_MODEL_REVISION not in {"main", "master", "latest"}


def test_recall_at_10_is_the_pre_registered_primary_metric():
    assert PRIMARY_METRIC == "recall_at_10"
    assert RunConfig().primary_metric == "recall_at_10"
    assert PRIMARY_METRIC not in EXPLORATORY_METRICS


def test_llm_default_model_and_timeout():
    """llama3.1:8b (nao -instruct) e 300s (nao 30s): modelos locais em CPU
    podem demorar mais que um timeout HTTP "normal" para responder -- um
    timeout curto inflava a contagem de falhas de resposta sem o modelo ter
    de fato falhado (ver evaluation/README.md, secao `llm`)."""
    config = LLMConfig()

    assert config.model == "llama3.1:8b"
    assert config.timeout_s == 300.0
