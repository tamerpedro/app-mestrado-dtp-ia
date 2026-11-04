from evaluation.metrics.ranking import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.metrics.validity import invalid_id_rate


def test_precision_at_k_counts_hits_in_top_k():
    predicted = ["A", "B", "C", "D", "E"]
    relevant = frozenset({"B", "D", "Z"})

    assert precision_at_k(predicted, relevant, 5) == 2 / 5
    assert precision_at_k(predicted, relevant, 2) == 1 / 2
    assert precision_at_k([], relevant, 5) == 0.0


def test_recall_at_k_divides_by_relevant_set_size():
    predicted = ["A", "B", "C"]
    relevant = frozenset({"B", "D"})

    assert recall_at_k(predicted, relevant, 3) == 1 / 2
    assert recall_at_k(predicted, frozenset(), 3) is None


def test_perfect_ranking_has_ndcg_of_one():
    predicted = ["A", "B", "C"]
    relevant = frozenset({"A", "B"})

    assert ndcg_at_k(predicted, relevant, 10) == 1.0


def test_worse_ranking_has_lower_ndcg_than_perfect():
    perfect = ndcg_at_k(["A", "B", "C"], frozenset({"A", "B"}), 10)
    worse = ndcg_at_k(["C", "A", "B"], frozenset({"A", "B"}), 10)

    assert worse < perfect


def test_ndcg_is_none_when_no_relevant_items():
    assert ndcg_at_k(["A", "B"], frozenset(), 10) is None


def test_dcg_ignores_items_outside_k():
    predicted = ["X", "X", "X", "A"]
    relevant = frozenset({"A"})

    assert dcg_at_k(predicted, relevant, 3) == 0.0
    assert dcg_at_k(predicted, relevant, 4) > 0.0


def test_mean_reciprocal_rank_rewards_earlier_hits():
    relevant = frozenset({"B"})

    assert mean_reciprocal_rank(["A", "B", "C"], relevant) == 1 / 2
    assert mean_reciprocal_rank(["B", "A", "C"], relevant) == 1.0
    assert mean_reciprocal_rank(["A", "C"], relevant) == 0.0


def test_invalid_id_rate_counts_ids_outside_the_library():
    valid = frozenset({"R001", "R002"})

    assert invalid_id_rate(["R001", "R999", "R002", "FAKE"], valid) == 0.5
    assert invalid_id_rate([], valid) is None
    assert invalid_id_rate(["R001"], valid) == 0.0
