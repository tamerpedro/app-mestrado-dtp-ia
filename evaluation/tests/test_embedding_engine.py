from evaluation.engines import embedding_engine
from evaluation.engines.embedding_engine import HashingEmbeddingBackend


def test_hashing_backend_ranks_matching_document_first(synthetic_risks, security_context):
    backend = HashingEmbeddingBackend()
    result = embedding_engine.rank(security_context, synthetic_risks, backend)

    assert set(result.ranked_ids) == {risk.id for risk in synthetic_risks}
    assert result.ranked_ids[0] == "S001"


def test_hashing_backend_is_deterministic(synthetic_risks, security_context):
    backend = HashingEmbeddingBackend()
    first = embedding_engine.rank(security_context, synthetic_risks, backend)
    second = embedding_engine.rank(security_context, synthetic_risks, backend)

    assert first.ranked_ids == second.ranked_ids


def test_build_default_backend_requires_optional_dependency_for_production_backend():
    from evaluation.config import EmbeddingConfig

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        import pytest

        with pytest.raises(ImportError):
            embedding_engine.build_default_backend(EmbeddingConfig(backend="sentence-transformers"))
    else:
        import pytest

        pytest.skip("sentence-transformers instalado neste ambiente; sem caso de erro a testar aqui.")


def test_build_default_backend_hashing_never_requires_optional_dependency():
    from evaluation.config import EmbeddingConfig

    backend = embedding_engine.build_default_backend(EmbeddingConfig(backend="hashing"))

    assert isinstance(backend, HashingEmbeddingBackend)
