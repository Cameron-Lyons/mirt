"""Strict numerical contracts for latent-density estimation."""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.estimation.latent_density import (
    CustomDensity,
    DavidianCurve,
    EmpiricalHistogram,
    EmpiricalHistogramWoods,
    GaussianDensity,
    MixtureDensity,
    create_density,
)


def test_gaussian_infers_dimensions_and_accepts_convenient_point_shapes() -> None:
    density = GaussianDensity(cov=np.eye(2))

    assert density.n_dimensions == 2
    assert density.log_density(np.array([0.0, 0.0])).shape == (1,)
    assert GaussianDensity().log_density(np.array([-1.0, 0.0, 1.0])).shape == (3,)


def test_gaussian_owns_constructor_arrays() -> None:
    mean = np.array([0.0, 1.0])
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]])
    density = GaussianDensity(mean=mean, cov=covariance)
    expected = density.log_density(np.zeros((1, 2)))

    mean[:] = 10.0
    covariance[:] = 10.0

    assert_allclose(density.log_density(np.zeros((1, 2))), expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_dimensions": 0}, "positive integer"),
        ({"mean": np.zeros((1, 1))}, "one-dimensional"),
        ({"mean": np.array([np.nan])}, "finite"),
        ({"mean": np.zeros(2), "cov": np.eye(3)}, "cov shape"),
        ({"cov": np.ones((2, 3))}, "square"),
        ({"cov": np.array([[1.0, 0.5], [0.1, 1.0]])}, "symmetric"),
        ({"cov": np.array([[1.0, 2.0], [2.0, 1.0]])}, "positive definite"),
        ({"n_dimensions": 3, "mean": np.zeros(2)}, "must match"),
    ],
)
def test_gaussian_rejects_invalid_distributions(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GaussianDensity(**kwargs)


@pytest.mark.parametrize("name", ["estimate_mean", "estimate_cov"])
def test_gaussian_requires_boolean_estimation_flags(name: str) -> None:
    with pytest.raises(TypeError, match="boolean"):
        GaussianDensity(**{name: 1})


def test_gaussian_update_validates_points_and_weights() -> None:
    density = GaussianDensity(n_dimensions=2, estimate_mean=True)

    with pytest.raises(ValueError, match="shape"):
        density.update(np.zeros((3, 1)), np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        density.update(np.zeros((3, 2)), np.ones(2))
    with pytest.raises(ValueError, match="positive total mass"):
        density.update(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        density.update(np.zeros((3, 2)), np.array([1.0, -1.0, 1.0]))


@pytest.mark.parametrize("density_type", [EmpiricalHistogram, EmpiricalHistogramWoods])
def test_histograms_validate_and_own_initial_probabilities(
    density_type: type[EmpiricalHistogram] | type[EmpiricalHistogramWoods],
) -> None:
    probabilities = np.array([1.0, 2.0, 1.0])
    density = density_type(n_bins=3, initial_probs=probabilities)
    probabilities[:] = 0.0

    assert_allclose(density.probabilities, [0.25, 0.5, 0.25])
    returned = density.probabilities
    assert returned is not None
    returned[:] = 0.0
    assert_allclose(density.probabilities, [0.25, 0.5, 0.25])


@pytest.mark.parametrize("density_type", [EmpiricalHistogram, EmpiricalHistogramWoods])
@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([]),
        np.array([[0.5, 0.5]]),
        np.array([0.5, np.nan]),
        np.array([0.5, -0.5]),
        np.zeros(2),
    ],
)
def test_histograms_reject_invalid_initial_probabilities(
    density_type: type[EmpiricalHistogram] | type[EmpiricalHistogramWoods],
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        density_type(initial_probs=probabilities)


@pytest.mark.parametrize("density_type", [EmpiricalHistogram, EmpiricalHistogramWoods])
def test_histograms_reject_inconsistent_point_counts_and_zero_mass(
    density_type: type[EmpiricalHistogram] | type[EmpiricalHistogramWoods],
) -> None:
    density = density_type(n_bins=3)

    with pytest.raises(ValueError, match="3 bins"):
        density.update(np.arange(4.0), np.ones(4))
    with pytest.raises(ValueError, match="positive total mass"):
        density.update(np.arange(3.0), np.zeros(3))

    density.update(np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError, match="expected 3"):
        density.log_density(np.arange(2.0))


@pytest.mark.parametrize("density_type", [EmpiricalHistogram, EmpiricalHistogramWoods])
def test_histograms_normalize_large_finite_mass_without_overflow(
    density_type: type[EmpiricalHistogram] | type[EmpiricalHistogramWoods],
) -> None:
    density = density_type()

    density.update(np.arange(2.0), np.array([1e308, 1e308]))

    assert_allclose(density.probabilities, [0.5, 0.5])


@pytest.mark.parametrize("density_type", [EmpiricalHistogram, EmpiricalHistogramWoods])
def test_histogram_quadrature_mass_uses_owned_discrete_probabilities(
    density_type: type[EmpiricalHistogram] | type[EmpiricalHistogramWoods],
) -> None:
    density = density_type(initial_probs=np.array([1.0, 2.0, 1.0]))
    theta = np.arange(3.0)[:, None]

    log_mass = density.log_quadrature_mass(theta, np.ones(3))

    assert_allclose(np.exp(log_mass), [0.25, 0.5, 0.25])
    with pytest.raises(ValueError, match="shape"):
        density.log_quadrature_mass(theta, np.ones((3, 1)))


@pytest.mark.parametrize("factor", [-1.0, np.nan, np.inf])
def test_woods_histogram_validates_extrapolation_factor(factor: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        EmpiricalHistogramWoods(extrapolation_factor=factor)


def test_woods_histogram_requires_numeric_extrapolation_factor() -> None:
    with pytest.raises(TypeError, match="real number"):
        EmpiricalHistogramWoods(extrapolation_factor="1.0")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": -1},
        {"degree": 1.5},
        {"degree": True},
        {"degree": 2, "coefficients": np.ones(2)},
        {"degree": 2, "coefficients": np.array([1.0, np.nan, 0.0])},
        {"degree": 2, "coefficients": np.zeros(3)},
    ],
)
def test_davidian_curve_validates_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        DavidianCurve(**kwargs)


def test_davidian_update_preserves_density_normalization_and_ownership() -> None:
    theta = np.linspace(-4.0, 4.0, 401)
    weights = np.exp(-0.5 * ((theta - 0.75) / 1.2) ** 2)
    density = DavidianCurve(degree=6)

    density.update(theta, weights)

    coefficients = density.coefficients
    factorials = np.array([factorial(k) for k in range(7)])
    assert_allclose(np.sum(coefficients**2 * factorials), 1.0, atol=1e-12)
    coefficients[:] = 0.0
    assert not np.all(density.coefficients == 0.0)


def test_davidian_normalizes_large_finite_coefficients_without_overflow() -> None:
    density = DavidianCurve(
        degree=2,
        coefficients=np.array([1e308, 1e308, 1e308]),
    )
    factorials = np.array([factorial(k) for k in range(3)])

    assert_allclose(np.sum(density.coefficients**2 * factorials), 1.0)


def test_davidian_update_rejects_invalid_mass() -> None:
    density = DavidianCurve()

    with pytest.raises(ValueError, match="positive total mass"):
        density.update(np.arange(5.0), np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        density.update(np.arange(5.0), np.ones(4))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_components": 0}, "positive integer"),
        ({"n_components": 2, "means": np.zeros(3)}, "means must have shape"),
        (
            {"n_components": 2, "variances": np.array([1.0, -1.0])},
            "finite positive",
        ),
        (
            {"n_components": 2, "weights": np.array([0.0, 0.0])},
            "positive total mass",
        ),
        (
            {"n_components": 2, "weights": np.array([0.5, np.nan])},
            "finite non-negative",
        ),
    ],
)
def test_mixture_rejects_invalid_components(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MixtureDensity(**kwargs)


def test_mixture_owns_inputs_and_stays_stable_in_extreme_tails() -> None:
    means = np.array([-2.0, 2.0])
    variances = np.array([0.25, 0.25])
    weights = np.array([0.4, 0.6])
    density = MixtureDensity(
        means=means,
        variances=variances,
        weights=weights,
    )
    expected = density.log_density(np.array([-100.0, 100.0]))

    means[:] = 0.0
    variances[:] = 10.0
    weights[:] = 0.5

    assert np.all(np.isfinite(expected))
    assert_allclose(density.log_density(np.array([-100.0, 100.0])), expected)


def test_mixture_update_is_normalized_and_finite_for_remote_points() -> None:
    density = MixtureDensity(
        means=np.array([-1_000.0, 1_000.0]),
        variances=np.ones(2),
    )
    theta = np.array([-0.1, 0.0, 0.1])

    density.update(theta, np.array([1.0, 2.0, 1.0]))

    assert_allclose(density.weights.sum(), 1.0)
    assert np.all(np.isfinite(density.means))
    assert np.all(np.isfinite(density.variances))
    assert np.all(density.variances > 0.0)


def test_mixture_update_validates_weights() -> None:
    density = MixtureDensity()

    with pytest.raises(ValueError, match="positive total mass"):
        density.update(np.arange(3.0), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        density.update(np.arange(3.0), np.ones(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_density_func": None},
        {"log_density_func": lambda theta: theta, "update_func": 1},
        {"log_density_func": lambda theta: theta, "n_params": -1},
    ],
)
def test_custom_density_validates_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CustomDensity(**kwargs)


def test_custom_density_validates_callback_output() -> None:
    wrong_shape = CustomDensity(log_density_func=lambda theta: np.zeros(2))
    nan_output = CustomDensity(
        log_density_func=lambda theta: np.full(len(theta), np.nan)
    )

    with pytest.raises(ValueError, match="one value per"):
        wrong_shape.log_density(np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        nan_output.log_density(np.zeros(3))


def test_density_factory_accepts_surrounding_whitespace() -> None:
    assert isinstance(create_density("  mixture  "), MixtureDensity)


def test_density_factory_requires_string_name() -> None:
    with pytest.raises(TypeError, match="string"):
        create_density(None)
