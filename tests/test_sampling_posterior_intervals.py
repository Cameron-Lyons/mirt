"""Posterior parameter interval and summary performance contracts."""

import numpy as np
import pytest

from mirt import ParameterSamples, posterior_summary


def test_equal_tailed_summary_uses_one_quantile_pass_per_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    samples = ParameterSamples(
        discrimination=rng.lognormal(size=(101, 3, 2)),
        difficulty=rng.normal(size=(101, 3)),
        guessing=rng.uniform(0.0, 0.3, size=(101, 3)),
    )
    arrays = {
        "discrimination": samples.discrimination,
        "difficulty": samples.difficulty,
        "guessing": samples.guessing,
    }
    expected = {
        name: {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "median": np.median(values, axis=0),
            "ci_lower": np.percentile(values, 10.0, axis=0),
            "ci_upper": np.percentile(values, 90.0, axis=0),
        }
        for name, values in arrays.items()
    }
    percentile = np.percentile
    quantile_requests = []

    def counted_percentile(values, q, axis=None):
        quantile_requests.append(np.asarray(q, dtype=np.float64))
        return percentile(values, q, axis=axis)

    monkeypatch.setattr(np, "percentile", counted_percentile)
    actual = posterior_summary(samples, credible_level=0.8)

    assert len(quantile_requests) == len(arrays)
    for request in quantile_requests:
        np.testing.assert_allclose(request, [10.0, 50.0, 90.0])
    for name, statistics in expected.items():
        for statistic, values in statistics.items():
            np.testing.assert_allclose(actual[name][statistic], values)


def test_highest_density_summary_selects_the_shortest_empirical_window() -> None:
    draws = np.array([0.0, 0.1, 0.2, 0.3, 10.0, 11.0, 12.0, 13.0])[:, None]
    samples = ParameterSamples(
        discrimination=draws + 1.0,
        difficulty=draws,
    )

    summary = posterior_summary(
        samples,
        credible_level=0.5,
        interval_method="highest_density",
    )

    np.testing.assert_allclose(summary["difficulty"]["ci_lower"], [0.0])
    np.testing.assert_allclose(summary["difficulty"]["ci_upper"], [0.3])
    np.testing.assert_allclose(summary["difficulty"]["median"], [5.15])
    np.testing.assert_allclose(summary["discrimination"]["ci_lower"], [1.0])
    np.testing.assert_allclose(summary["discrimination"]["ci_upper"], [1.3])


def test_highest_density_summary_preserves_multidimensional_shapes() -> None:
    discrimination = np.array(
        [
            [[0.8, 1.2], [1.0, 0.7]],
            [[0.9, 1.1], [1.2, 0.8]],
            [[1.0, 1.0], [1.1, 0.9]],
            [[1.1, 0.9], [1.3, 1.0]],
        ]
    )
    samples = ParameterSamples(
        discrimination=discrimination,
        difficulty=np.arange(8, dtype=np.float64).reshape(4, 2),
    )

    summary = posterior_summary(
        samples,
        credible_level=0.75,
        interval_method="highest_density",
    )

    assert summary["discrimination"]["ci_lower"].shape == (2, 2)
    assert summary["discrimination"]["ci_upper"].shape == (2, 2)
    assert summary["difficulty"]["median"].shape == (2,)


def test_highest_density_summary_handles_one_draw() -> None:
    samples = ParameterSamples(
        discrimination=np.array([[1.2, 0.8]]),
        difficulty=np.array([[0.1, -0.2]]),
    )

    summary = posterior_summary(
        samples,
        interval_method="highest_density",
    )

    for statistics in summary.values():
        np.testing.assert_array_equal(statistics["ci_lower"], statistics["median"])
        np.testing.assert_array_equal(statistics["ci_upper"], statistics["median"])


@pytest.mark.parametrize("interval_method", ["central", "hdi", "", None, []])
def test_posterior_summary_rejects_unknown_interval_methods(interval_method) -> None:
    samples = ParameterSamples(np.ones((2, 1)), np.zeros((2, 1)))

    with pytest.raises(ValueError, match="interval_method"):
        posterior_summary(samples, interval_method=interval_method)
