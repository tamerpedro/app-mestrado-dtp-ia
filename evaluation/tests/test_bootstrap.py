import pytest

from evaluation.metrics.bootstrap import paired_bootstrap_ci


def test_ci_is_centered_around_the_observed_mean_difference():
    diffs = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    result = paired_bootstrap_ci(diffs, confidence=0.95, n_resamples=2000, seed=42)

    assert result.observed_mean_diff == pytest.approx(0.2)
    assert result.ci_lower <= result.observed_mean_diff <= result.ci_upper


def test_ci_excludes_zero_when_difference_is_clearly_positive():
    diffs = [0.3] * 30

    result = paired_bootstrap_ci(diffs, confidence=0.95, n_resamples=2000, seed=1)

    assert result.ci_lower > 0
    assert result.p_value_two_sided < 0.05


def test_ci_includes_zero_when_difference_is_noisy_and_small():
    diffs = [0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02, 0.01, -0.01, 0.0]

    result = paired_bootstrap_ci(diffs, confidence=0.95, n_resamples=2000, seed=7)

    assert result.ci_lower <= 0 <= result.ci_upper


def test_result_is_reproducible_given_the_same_seed():
    diffs = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0, -0.05, 0.2, -0.3, 0.1]

    first = paired_bootstrap_ci(diffs, n_resamples=500, seed=123)
    second = paired_bootstrap_ci(diffs, n_resamples=500, seed=123)

    assert first == second


def test_requires_at_least_two_cases():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1], n_resamples=100, seed=1)
