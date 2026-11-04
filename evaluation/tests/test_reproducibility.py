import pytest

from evaluation.metrics.reproducibility import measure_reproducibility


def test_deterministic_function_has_perfect_reproducibility():
    result = measure_reproducibility(lambda: ["A", "B", "C"], repeats=3)

    assert result.exact_match_rate == 1.0
    assert result.mean_top_k_jaccard == 1.0


def test_varying_function_has_imperfect_reproducibility():
    responses = iter([["A", "B"], ["A", "C"], ["B", "A"]])

    result = measure_reproducibility(lambda: next(responses), repeats=3)

    assert result.exact_match_rate == 0.0
    assert 0.0 < result.mean_top_k_jaccard < 1.0


def test_requires_at_least_two_repeats():
    with pytest.raises(ValueError):
        measure_reproducibility(lambda: ["A"], repeats=1)
