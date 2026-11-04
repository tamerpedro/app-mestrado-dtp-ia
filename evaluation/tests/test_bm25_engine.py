from evaluation.config import BM25Config
from evaluation.engines import bm25_engine


def test_bm25_ranks_matching_document_first(synthetic_risks, security_context):
    result = bm25_engine.rank(security_context, synthetic_risks)

    assert set(result.ranked_ids) == {risk.id for risk in synthetic_risks}
    assert result.ranked_ids[0] == "S001"


def test_bm25_is_deterministic_across_runs(synthetic_risks, security_context):
    first = bm25_engine.rank(security_context, synthetic_risks)
    second = bm25_engine.rank(security_context, synthetic_risks)

    assert first.ranked_ids == second.ranked_ids


def test_bm25_config_parameters_affect_score_but_not_the_api(synthetic_risks, security_context):
    lenient = bm25_engine.rank(security_context, synthetic_risks, config=BM25Config(k1=0.1, b=0.0))
    default = bm25_engine.rank(security_context, synthetic_risks, config=BM25Config())

    assert set(lenient.ranked_ids) == set(default.ranked_ids)
