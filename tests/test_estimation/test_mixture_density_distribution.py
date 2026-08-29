"""Distribution-level tests for Gaussian latent mixtures."""

import numpy as np
import pytest
from scipy import stats

from mirt.estimation.latent_density import MixtureDensity


@pytest.fixture
def mixture() -> MixtureDensity:
    return MixtureDensity(
        n_components=3,
        means=np.array([-2.0, 0.5, 2.5]),
        variances=np.array([0.25, 1.0, 0.49]),
        weights=np.array([0.2, 0.5, 0.3]),
    )


def test_component_probabilities_match_weighted_normal_densities(
    mixture: MixtureDensity,
) -> None:
    theta = np.array([-2.0, -0.25, 1.0, 3.0])
    weighted = (
        stats.norm.pdf(
            theta[:, None],
            loc=mixture.means[None, :],
            scale=np.sqrt(mixture.variances)[None, :],
        )
        * mixture.weights[None, :]
    )
    expected = weighted / weighted.sum(axis=1, keepdims=True)

    actual = mixture.component_probabilities(theta)

    assert actual.shape == (4, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        mixture.component_probabilities(theta[:, None]),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )


def test_component_probabilities_are_stable_in_remote_tails() -> None:
    mixture = MixtureDensity(
        n_components=3,
        means=np.array([-10.0, 0.0, 10.0]),
        variances=np.array([0.1, 2.0, 0.1]),
        weights=np.array([0.0, 0.25, 0.75]),
    )

    probabilities = mixture.component_probabilities(np.array([-1e6, 1e6]))

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(probabilities[:, 0], 0.0)


def test_classification_selects_maximum_probability_component(
    mixture: MixtureDensity,
) -> None:
    theta = np.array([-3.0, 0.5, 3.0])

    classes = mixture.classify(theta)

    np.testing.assert_array_equal(classes, [0, 1, 2])
    np.testing.assert_array_equal(
        classes,
        np.argmax(mixture.component_probabilities(theta), axis=1),
    )


def test_cdf_matches_weighted_component_cdfs(mixture: MixtureDensity) -> None:
    theta = np.linspace(-5.0, 5.0, 51)
    expected = (
        stats.norm.cdf(
            theta[:, None],
            loc=mixture.means[None, :],
            scale=np.sqrt(mixture.variances)[None, :],
        )
        @ mixture.weights
    )

    actual = mixture.cdf(theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert np.all(np.diff(actual) >= 0.0)
    assert 0.0 <= actual[0] < actual[-1] <= 1.0
    assert mixture.cdf(np.array(0.0)).shape == (1,)


def test_analytic_moments_match_component_formula(mixture: MixtureDensity) -> None:
    expected_mean = float(mixture.weights @ mixture.means)
    expected_variance = float(
        mixture.weights @ (mixture.variances + mixture.means**2) - expected_mean**2
    )

    assert mixture.mean == pytest.approx(expected_mean)
    assert mixture.variance == pytest.approx(expected_variance)
    assert mixture.standard_deviation == pytest.approx(np.sqrt(expected_variance))


def test_sampling_is_reproducible_and_returns_generating_components(
    mixture: MixtureDensity,
) -> None:
    samples, components = mixture.sample(
        200_000,
        random_state=2026,
        return_components=True,
    )
    repeated = mixture.sample(200_000, random_state=2026)

    np.testing.assert_array_equal(samples, repeated)
    assert samples.shape == components.shape == (200_000,)
    assert np.all((components >= 0) & (components < mixture.n_components))
    assert np.mean(samples) == pytest.approx(mixture.mean, abs=0.015)
    assert np.var(samples) == pytest.approx(mixture.variance, abs=0.03)
    np.testing.assert_allclose(
        np.bincount(components, minlength=3) / len(components),
        mixture.weights,
        atol=0.003,
    )


def test_sampling_accepts_existing_generator_and_zero_draws(
    mixture: MixtureDensity,
) -> None:
    generator = np.random.default_rng(42)
    first = mixture.sample(4, random_state=generator)
    second = mixture.sample(4, random_state=generator)
    empty, components = mixture.sample(0, return_components=True)

    assert not np.array_equal(first, second)
    assert empty.shape == components.shape == (0,)


@pytest.mark.parametrize(
    ("method", "argument", "error", "message"),
    [
        ("component_probabilities", np.ones((2, 2)), ValueError, "univariate"),
        ("component_probabilities", np.array([np.nan]), ValueError, "finite"),
        ("cdf", np.ones((2, 2)), ValueError, "univariate"),
        ("classify", np.array([np.inf]), ValueError, "finite"),
    ],
)
def test_distribution_operations_validate_points(
    mixture: MixtureDensity,
    method: str,
    argument: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        getattr(mixture, method)(argument)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_samples": -1}, ValueError, "non-negative integer"),
        ({"n_samples": 1.5}, ValueError, "non-negative integer"),
        ({"n_samples": True}, ValueError, "non-negative integer"),
        ({"n_samples": 1, "random_state": "seed"}, TypeError, "random_state"),
        ({"n_samples": 1, "return_components": 1}, TypeError, "boolean"),
    ],
)
def test_sampling_validates_controls(
    mixture: MixtureDensity,
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        mixture.sample(**kwargs)
