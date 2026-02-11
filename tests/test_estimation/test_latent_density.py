"""Tests for latent density module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from mirt.estimation.latent_density import (
    CustomDensity,
    DavidianCurve,
    EmpiricalHistogram,
    EmpiricalHistogramWoods,
    GaussianDensity,
    MixtureDensity,
    create_density,
)


class TestGaussianDensity:
    def test_default_1d(self):
        d = GaussianDensity()
        assert d.n_dimensions == 1
        assert_allclose(d.mean, [0.0])
        assert_allclose(d.cov, [[1.0]])

    def test_custom_mean_cov(self):
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        d = GaussianDensity(mean=mean, cov=cov)
        assert d.n_dimensions == 2
        assert_allclose(d.mean, mean)
        assert_allclose(d.cov, cov)

    def test_log_density_shape_1d(self):
        d = GaussianDensity()
        theta = np.array([[0.0], [-1.0], [1.0]])
        result = d.log_density(theta)
        assert result.shape == (3,)

    def test_log_density_values_standard_normal(self):
        d = GaussianDensity()
        theta = np.array([[0.0]])
        result = d.log_density(theta)
        expected = stats.norm.logpdf(0.0)
        assert_allclose(result[0], expected, atol=1e-10)

    def test_log_density_max_at_mean(self):
        d = GaussianDensity(mean=np.array([2.0]))
        theta = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        result = d.log_density(theta)
        assert np.argmax(result) == 2

    def test_density_equals_exp_log_density(self):
        d = GaussianDensity()
        theta = np.array([[0.0], [-1.0], [1.0]])
        assert_allclose(d.density(theta), np.exp(d.log_density(theta)))

    def test_n_parameters_no_estimation(self):
        d = GaussianDensity()
        assert d.n_parameters == 0

    def test_n_parameters_estimate_mean(self):
        d = GaussianDensity(n_dimensions=2, estimate_mean=True)
        assert d.n_parameters == 2

    def test_n_parameters_estimate_cov(self):
        d = GaussianDensity(n_dimensions=2, estimate_cov=True)
        assert d.n_parameters == 3

    def test_n_parameters_both(self):
        d = GaussianDensity(n_dimensions=3, estimate_mean=True, estimate_cov=True)
        assert d.n_parameters == 3 + 6

    def test_update_no_estimation(self):
        d = GaussianDensity()
        original_mean = d.mean.copy()
        theta = np.array([[0.0], [1.0], [2.0]])
        weights = np.array([0.3, 0.4, 0.3])
        d.update(theta, weights)
        assert_allclose(d.mean, original_mean)

    def test_update_mean(self):
        d = GaussianDensity(estimate_mean=True)
        theta = np.array([[0.0], [1.0], [2.0]])
        weights = np.array([0.1, 0.3, 0.6])
        d.update(theta, weights)
        assert d.mean[0] > 0

    def test_update_covariance(self):
        d = GaussianDensity(n_dimensions=1, estimate_cov=True)
        theta = np.array([[-2.0], [0.0], [2.0]])
        weights = np.array([0.33, 0.34, 0.33])
        d.update(theta, weights)
        assert d.cov[0, 0] > 0

    def test_multidimensional(self):
        d = GaussianDensity(n_dimensions=3)
        theta = np.zeros((5, 3))
        result = d.log_density(theta)
        assert result.shape == (5,)


class TestEmpiricalHistogram:
    def test_uninitialized_raises(self):
        d = EmpiricalHistogram()
        with pytest.raises(ValueError, match="not initialized"):
            d.log_density(np.array([0.0]))

    def test_initial_probs(self):
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        d = EmpiricalHistogram(initial_probs=probs)
        assert d.n_bins == 5
        assert_allclose(d._probs.sum(), 1.0)

    def test_update_sets_probs(self):
        d = EmpiricalHistogram()
        weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        theta = np.linspace(-2, 2, 5)
        d.update(theta, weights)
        assert d._probs is not None
        assert_allclose(d._probs.sum(), 1.0, atol=1e-10)

    def test_log_density_shape(self):
        d = EmpiricalHistogram(initial_probs=np.ones(5))
        result = d.log_density(np.zeros(5))
        assert result.shape == (5,)

    def test_density_positive(self):
        d = EmpiricalHistogram(initial_probs=np.array([0.2, 0.3, 0.3, 0.2]))
        dens = d.density(np.zeros(4))
        assert np.all(dens > 0)

    def test_n_parameters_uninitialized(self):
        d = EmpiricalHistogram()
        assert d.n_parameters == 0

    def test_n_parameters_after_update(self):
        d = EmpiricalHistogram()
        d.update(np.zeros(10), np.ones(10))
        assert d.n_parameters == 9

    def test_update_reflects_weights(self):
        d = EmpiricalHistogram()
        weights = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        d.update(np.linspace(-2, 2, 5), weights)
        assert d._probs[2] > d._probs[0]


class TestEmpiricalHistogramWoods:
    def test_uninitialized_raises(self):
        d = EmpiricalHistogramWoods()
        with pytest.raises(ValueError, match="not initialized"):
            d.log_density(np.array([0.0]))

    def test_initial_probs(self):
        probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        d = EmpiricalHistogramWoods(initial_probs=probs)
        assert d.n_bins == 5

    def test_update_and_log_density(self):
        d = EmpiricalHistogramWoods()
        n_bins = 20
        weights = np.exp(-0.5 * np.linspace(-3, 3, n_bins) ** 2)
        theta = np.linspace(-3, 3, n_bins)
        d.update(theta, weights)
        result = d.log_density(theta)
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))

    def test_extrapolation_factor(self):
        d0 = EmpiricalHistogramWoods(extrapolation_factor=0.0)
        d1 = EmpiricalHistogramWoods(extrapolation_factor=2.0)

        n_bins = 20
        theta = np.linspace(-3, 3, n_bins)
        weights = np.exp(-0.5 * (theta + 1) ** 2) + 0.3 * np.exp(
            -0.5 * (theta - 2) ** 2
        )
        weights[0] = 0.001
        weights[-1] = 0.001

        d0.update(theta, weights)
        d1.update(theta, weights)

        ld0 = d0.log_density(theta)
        ld1 = d1.log_density(theta)
        assert not np.allclose(ld0, ld1)

    def test_n_parameters(self):
        d = EmpiricalHistogramWoods()
        d.update(np.zeros(10), np.ones(10))
        assert d.n_parameters == 9


class TestDavidianCurve:
    def test_default_initialization(self):
        d = DavidianCurve()
        assert d.degree == 4
        assert d.n_parameters == 4

    def test_custom_degree(self):
        d = DavidianCurve(degree=6)
        assert d.degree == 6
        assert d.n_parameters == 6

    def test_log_density_shape(self):
        d = DavidianCurve(degree=4)
        theta = np.linspace(-3, 3, 20)
        result = d.log_density(theta)
        assert result.shape == (20,)

    def test_log_density_finite(self):
        d = DavidianCurve(degree=4)
        theta = np.linspace(-3, 3, 20)
        result = d.log_density(theta)
        assert np.all(np.isfinite(result))

    def test_density_nonnegative(self):
        d = DavidianCurve(degree=4)
        theta = np.linspace(-4, 4, 50)
        dens = d.density(theta)
        assert np.all(dens >= 0)

    def test_2d_input(self):
        d = DavidianCurve(degree=4)
        theta = np.linspace(-2, 2, 10).reshape(-1, 1)
        result = d.log_density(theta)
        assert result.shape == (10,)

    def test_update(self):
        d = DavidianCurve(degree=4)
        coeffs_before = d._coeffs.copy()
        theta = np.linspace(-3, 3, 30)
        weights = np.exp(-0.5 * theta**2)
        weights /= weights.sum()
        d.update(theta, weights)
        assert not np.allclose(d._coeffs, coeffs_before)

    def test_custom_coefficients(self):
        coeffs = np.array([1.0, 0.0, 0.0])
        d = DavidianCurve(degree=2, coefficients=coeffs)
        theta = np.array([0.0])
        ld = d.log_density(theta)
        assert np.isfinite(ld[0])


class TestMixtureDensity:
    def test_default_2_components(self):
        d = MixtureDensity()
        assert d.n_components == 2
        assert_allclose(d.weights.sum(), 1.0)

    def test_custom_parameters(self):
        d = MixtureDensity(
            n_components=3,
            means=np.array([-2, 0, 2]),
            variances=np.array([0.5, 0.5, 0.5]),
            weights=np.array([0.3, 0.4, 0.3]),
        )
        assert d.n_components == 3
        assert_allclose(d.weights.sum(), 1.0)

    def test_log_density_shape(self):
        d = MixtureDensity(n_components=2)
        theta = np.linspace(-3, 3, 20)
        result = d.log_density(theta)
        assert result.shape == (20,)

    def test_log_density_finite(self):
        d = MixtureDensity(n_components=2)
        theta = np.linspace(-4, 4, 30)
        result = d.log_density(theta)
        assert np.all(np.isfinite(result))

    def test_density_nonnegative(self):
        d = MixtureDensity()
        theta = np.linspace(-4, 4, 50)
        dens = d.density(theta)
        assert np.all(dens >= 0)

    def test_2d_input(self):
        d = MixtureDensity()
        theta = np.linspace(-2, 2, 10).reshape(-1, 1)
        result = d.log_density(theta)
        assert result.shape == (10,)

    def test_update_changes_params(self):
        d = MixtureDensity(n_components=2)
        means_before = d.means.copy()
        theta = np.concatenate([np.full(20, -2), np.full(20, 2)])
        weights = np.ones(40) / 40
        d.update(theta, weights)
        assert not np.allclose(d.means, means_before, atol=0.01)

    def test_n_parameters(self):
        d = MixtureDensity(n_components=3)
        assert d.n_parameters == 8


class TestCustomDensity:
    def test_log_density(self):
        d = CustomDensity(log_density_func=lambda theta: -0.5 * theta**2)
        theta = np.array([0.0, 1.0, -1.0])
        result = d.log_density(theta)
        assert_allclose(result, np.array([0.0, -0.5, -0.5]))

    def test_update_with_func(self):
        state = {"called": False}

        def update_fn(theta, weights):
            state["called"] = True

        d = CustomDensity(
            log_density_func=lambda t: np.zeros_like(t),
            update_func=update_fn,
        )
        d.update(np.zeros(5), np.ones(5))
        assert state["called"]

    def test_update_without_func(self):
        d = CustomDensity(log_density_func=lambda t: np.zeros_like(t))
        d.update(np.zeros(5), np.ones(5))

    def test_n_parameters(self):
        d = CustomDensity(log_density_func=lambda t: np.zeros_like(t), n_params=5)
        assert d.n_parameters == 5


class TestCreateDensity:
    def test_gaussian(self):
        d = create_density("gaussian")
        assert isinstance(d, GaussianDensity)

    def test_normal_alias(self):
        d = create_density("normal")
        assert isinstance(d, GaussianDensity)

    def test_empirical(self):
        d = create_density("empirical")
        assert isinstance(d, EmpiricalHistogram)

    def test_histogram_alias(self):
        d = create_density("histogram")
        assert isinstance(d, EmpiricalHistogram)

    def test_eh_alias(self):
        d = create_density("eh")
        assert isinstance(d, EmpiricalHistogram)

    def test_ehw_alias(self):
        d = create_density("ehw")
        assert isinstance(d, EmpiricalHistogramWoods)

    def test_davidian(self):
        d = create_density("davidian", degree=6)
        assert isinstance(d, DavidianCurve)
        assert d.degree == 6

    def test_mixture(self):
        d = create_density("mixture", n_components=3)
        assert isinstance(d, MixtureDensity)
        assert d.n_components == 3

    def test_custom(self):
        d = create_density("custom", log_density_func=lambda t: np.zeros_like(t))
        assert isinstance(d, CustomDensity)

    def test_case_insensitive(self):
        d = create_density("Gaussian")
        assert isinstance(d, GaussianDensity)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown density type"):
            create_density("unknown_type")
