from evaluation.engines import rules_engine


def test_rank_returns_all_risks_ordered_by_score(synthetic_risks, security_context):
    result = rules_engine.rank(security_context, synthetic_risks)

    assert set(result.ranked_ids) == {risk.id for risk in synthetic_risks}
    assert result.ranked_ids[0] == "S001"  # risco de seguranca, mais aderente ao contexto


def test_rank_is_deterministic(synthetic_risks, security_context):
    first = rules_engine.rank(security_context, synthetic_risks)
    second = rules_engine.rank(security_context, synthetic_risks)

    assert first.ranked_ids == second.ranked_ids


def test_production_ranking_respects_minimum_score_and_category_cap(synthetic_risks, security_context):
    production = rules_engine.rank_as_shown_in_production(security_context, synthetic_risks)
    full_ranking = rules_engine.rank(security_context, synthetic_risks)

    assert len(production.ranked_ids) <= len(full_ranking.ranked_ids)
    assert set(production.ranked_ids) <= set(full_ranking.ranked_ids)
