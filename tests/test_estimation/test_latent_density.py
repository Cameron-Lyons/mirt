"""Tests for latent density module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from mirt import simdata
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import (
    CustomDensity,
    DavidianCurve,
    EmpiricalHistogram,
    EmpiricalHistogramWoods,
    GaussianDensity,
    MixtureDensity,
    create_density,
)
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic


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

    def test_extrapolation_factor_is_deprecated_and_ignored(self):
        # The factor blended in a quadratic log-tail fit, which inflated the
        # tails every EM cycle; Woods' standardization has no such factor.
        theta = np.linspace(-3, 3, 20)
        weights = np.exp(-0.5 * (theta + 1) ** 2) + 0.3 * np.exp(
            -0.5 * (theta - 2) ** 2
        )
        reference = EmpiricalHistogramWoods()
        with pytest.warns(DeprecationWarning, match="extrapolation_factor"):
            deprecated = EmpiricalHistogramWoods(extrapolation_factor=2.0)

        reference.update(theta, weights)
        deprecated.update(theta, weights)

        assert_allclose(deprecated.probabilities, reference.probabilities)

    def test_n_parameters(self):
        # The mean and variance are fixed, as in R mirt's dentype="EHW",
        # which counts quadpts - 3 density parameters.
        d = EmpiricalHistogramWoods()
        d.update(np.linspace(-3, 3, 10), np.ones(10))
        assert d.n_parameters == 7

    def test_update_standardizes_the_posterior_histogram(self):
        theta = np.linspace(-6, 6, 61)
        counts = stats.norm.pdf(theta, loc=0.5, scale=1.3) * (1 + 0.4 * np.tanh(theta))
        d = EmpiricalHistogramWoods()

        d.update(theta, counts)

        probabilities = d.probabilities
        mean = probabilities @ theta
        assert abs(mean) < 1e-10
        assert abs(probabilities @ (theta - mean) ** 2 - 1.0) < 1e-10

    def test_shifted_histogram_is_moved_back_on_an_equally_spaced_grid(self):
        # A pure shift is undone exactly, as by R mirt's standardizeQuadrature
        # (which agrees with this rescaling to 4e-10 for such input).
        theta = np.linspace(-6, 6, 121)
        d = EmpiricalHistogramWoods()
        d.update(theta, stats.norm.pdf(theta))
        standard = d.probabilities
        shifted = np.concatenate([np.zeros(4), standard[:-4]])

        d.update(theta, shifted)

        assert_allclose(d.probabilities, standard, atol=1e-9)

    def test_update_never_fits_the_counts_worse(self):
        theta = np.linspace(-6, 6, 61)
        rng = np.random.default_rng(3)
        d = EmpiricalHistogramWoods()
        d.log_quadrature_mass(theta, stats.norm.pdf(theta))
        for _ in range(20):
            counts = stats.norm.pdf(theta, rng.normal(0, 0.3), rng.uniform(0.7, 1.4))
            counts *= rng.uniform(0.5, 1.5, len(theta))
            counts /= counts.sum()
            before = counts @ np.log(d.probabilities)

            d.update(theta, counts)

            assert counts @ np.log(d.probabilities) >= before


class TestDavidianCurve:
    def test_default_initialization(self):
        # Mean 0 and variance 1 constrain two of the four coefficients.
        d = DavidianCurve()
        assert d.degree == 4
        assert d.standardize
        assert d.n_parameters == 2
        assert_allclose([d.mean, d.variance], [0.0, 1.0], atol=1e-14)

    def test_custom_degree(self):
        d = DavidianCurve(degree=6)
        assert d.degree == 6
        assert d.n_parameters == 4
        assert DavidianCurve(degree=6, standardize=False).n_parameters == 6

    def test_standardize_must_be_boolean(self):
        with pytest.raises(TypeError, match="standardize"):
            DavidianCurve(standardize=1)

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
        weights = np.exp(-0.5 * (theta - 0.8) ** 2)
        weights /= weights.sum()
        d.update(theta, weights)
        assert not np.allclose(d._coeffs, coeffs_before)

    def test_custom_coefficients(self):
        coeffs = np.array([1.0, 0.0, 0.0])
        d = DavidianCurve(degree=2, coefficients=coeffs)
        theta = np.array([0.0])
        ld = d.log_density(theta)
        assert np.isfinite(ld[0])

    def test_update_standardizes_the_curve(self):
        # The curve used to follow the counts to mean 0.8 and sd 1.2, moving
        # the latent location and scale into the item parameters.
        theta = np.linspace(-6, 6, 61)
        counts = stats.norm.pdf(theta, loc=0.8, scale=1.2) * (1 + 0.5 * np.tanh(theta))
        d = DavidianCurve(degree=4)

        d.update(theta, counts)

        assert_allclose([d.mean, d.variance], [0.0, 1.0], atol=1e-10)
        assert d.coefficients[3:].any()
        masses = np.exp(d.log_quadrature_mass(theta, stats.norm.pdf(theta)))
        mean = masses @ theta
        assert abs(mean) < 1e-5
        assert abs(masses @ (theta - mean) ** 2 - 1.0) < 1e-5

    def test_update_maximizes_the_expected_log_likelihood(self):
        # On Gauss-Hermite nodes the curve's quadrature masses are w * g**2,
        # so the counts are fitted at least as well as by the standard normal.
        nodes, weights = np.polynomial.hermite_e.hermegauss(21)
        weights /= weights.sum()
        counts = weights * (1 + 0.6 * np.tanh(nodes))
        d = DavidianCurve(degree=4)
        start = counts @ d.log_quadrature_mass(nodes, weights)

        d.update(nodes, counts)

        assert counts @ d.log_quadrature_mass(nodes, weights) > start + 1e-3

    def test_unstandardized_update_follows_the_counts(self):
        theta = np.linspace(-6, 6, 61)
        d = DavidianCurve(degree=4, standardize=False)

        d.update(theta, stats.norm.pdf(theta, loc=0.8))

        assert d.mean == pytest.approx(0.8, abs=0.05)


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
        # Counts at -1 and 1 have mean 0 and variance 1, so the standardized
        # components move toward them.
        d = MixtureDensity(n_components=2)
        means_before = d.means.copy()
        theta = np.concatenate([np.full(20, -1), np.full(20, 1)])
        weights = np.ones(40) / 40
        d.update(theta, weights)
        assert not np.allclose(d.means, means_before, atol=0.01)
        assert np.all(np.abs(d.means) > np.abs(means_before))

    def test_unstandardized_update_follows_the_counts(self):
        d = MixtureDensity(n_components=2, standardize=False)
        theta = np.concatenate([np.full(20, -2), np.full(20, 2)])
        d.update(theta, np.ones(40) / 40)
        assert_allclose(d.means, [-2.0, 2.0], atol=0.01)

    def test_default_start_and_update_are_standardized(self):
        d = MixtureDensity(n_components=3)
        assert_allclose([d.mean, d.variance], [0.0, 1.0], atol=1e-14)
        theta = np.linspace(-6, 6, 61)
        counts = stats.norm.pdf(theta, loc=0.5, scale=1.4) * (1 + 0.5 * np.tanh(theta))

        d.update(theta, counts)

        assert_allclose([d.mean, d.variance], [0.0, 1.0], atol=1e-12)

    def test_n_parameters(self):
        # Mean 0 and variance 1 constrain two of the eight free parameters.
        d = MixtureDensity(n_components=3)
        assert d.n_parameters == 6
        assert MixtureDensity(n_components=3, standardize=False).n_parameters == 8


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


_DENSITY_NAMES = {
    "gaussian": GaussianDensity,
    "normal": GaussianDensity,
    "empirical": EmpiricalHistogram,
    "histogram": EmpiricalHistogram,
    "eh": EmpiricalHistogram,
    "ehw": EmpiricalHistogramWoods,
    "davidian": DavidianCurve,
    "mixture": MixtureDensity,
}


@pytest.fixture(scope="module")
def responses_2pl():
    return simdata("2PL", n_persons=300, n_items=6, seed=12)


@pytest.mark.parametrize("name", list(_DENSITY_NAMES))
def test_em_builds_latent_densities_by_name(name, responses_2pl):
    # Every name but the Gaussian ones used to raise TypeError.
    estimator = EMEstimator(
        latent_density=name, n_quadpts=15, max_iter=20, compute_standard_errors=False
    )
    result = estimator.fit(TwoParameterLogistic(6), responses_2pl)
    density = estimator._latent_density
    assert type(density) is _DENSITY_NAMES[name]
    assert np.isfinite(result.log_likelihood)
    assert result.n_parameters == 12 + density.n_parameters
    # A name builds a new density for every fit.
    estimator.fit(TwoParameterLogistic(6), responses_2pl)
    assert estimator._latent_density is not density


def test_gaussian_density_name_uses_the_fit_prior(responses_2pl):
    estimator = EMEstimator(latent_density=" Gaussian ", max_iter=2)
    estimator.fit(TwoParameterLogistic(6), responses_2pl, prior_mean=np.array([0.5]))
    assert_allclose(estimator._latent_density.mean, [0.5])


def test_univariate_density_names_require_one_factor(responses_2pl):
    for name in ("davidian", "mixture", "ehw"):
        with pytest.raises(MirtValidationError, match="univariate"):
            EMEstimator(latent_density=name).fit(
                TwoParameterLogistic(6, n_factors=2), responses_2pl
            )
    estimator = EMEstimator(
        latent_density="empirical",
        n_quadpts=7,
        max_iter=5,
        compute_standard_errors=False,
    )
    result = estimator.fit(TwoParameterLogistic(6, n_factors=2), responses_2pl)
    assert np.isfinite(result.log_likelihood)
    assert estimator._latent_density.n_parameters == 7**2 - 1
    with pytest.raises(MirtValidationError, match="Unknown latent density"):
        EMEstimator(latent_density="lognormal")


@pytest.fixture(scope="module")
def standard_normal_2pl():
    """2PL responses of standard-normal persons, with their Gaussian fit."""
    rng = np.random.default_rng(8)
    n_items = 12
    truth = TwoParameterLogistic(n_items)
    truth.set_parameters(
        discrimination=rng.uniform(0.8, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    theta = rng.standard_normal((1500, 1))
    responses = (rng.random((1500, n_items)) < truth.probability(theta)).astype(int)
    gaussian = EMEstimator(n_quadpts=21, compute_standard_errors=False).fit(
        TwoParameterLogistic(n_items), responses
    )
    return responses, gaussian


@pytest.mark.parametrize("name", ["ehw", "davidian", "mixture"])
def test_estimated_densities_stay_standardized_under_em(name, standard_normal_2pl):
    # On N(0, 1) data the Woods histogram used to run its mass to the top
    # node (mean +7.3) while the log-likelihood fell by 5,220, and the
    # Davidian curve and mixture drifted to mean +0.2 and sd 1.2, which
    # shifted difficulties by +0.18 and shrank slopes by 10-16%.
    responses, gaussian = standard_normal_2pl
    estimator = EMEstimator(
        latent_density=name,
        n_quadpts=21,
        max_iter=150,
        compute_standard_errors=False,
    )
    result = estimator.fit(TwoParameterLogistic(responses.shape[1]), responses)

    history = np.asarray(estimator.convergence_history)
    assert np.all(np.diff(history) > -1e-8 * np.abs(history[1:]))
    quadrature = estimator._quadrature
    theta = quadrature.nodes[:, 0]
    masses = np.exp(
        estimator._latent_density.log_quadrature_mass(
            quadrature.nodes, quadrature.weights
        )
    )
    mean = masses @ theta
    assert abs(mean) < 1e-5
    assert abs(masses @ (theta - mean) ** 2 - 1.0) < 1e-5
    assert result.latent_mean is None
    assert result.latent_covariance is None

    shift = result.model.difficulty - gaussian.model.difficulty
    ratio = result.model.discrimination / gaussian.model.discrimination
    assert abs(np.mean(shift)) < 0.05
    assert 0.9 < np.mean(ratio) < 1.15


def test_refit_recipe_keeps_the_standardized_density(standard_normal_2pl):
    responses, _ = standard_normal_2pl
    estimator = EMEstimator(
        latent_density="davidian",
        n_quadpts=21,
        max_iter=40,
        compute_standard_errors=False,
    )
    result = estimator.fit(TwoParameterLogistic(responses.shape[1]), responses)

    density = result.refit_recipe.latent_density
    assert isinstance(density, DavidianCurve)
    assert density is not estimator._latent_density
    assert_allclose([density.mean, density.variance], [0.0, 1.0], atol=1e-10)
    rebuilt = result.refit_recipe.build()._latent_density_spec
    assert isinstance(rebuilt, DavidianCurve) and rebuilt is not density
    assert_allclose(rebuilt.coefficients, density.coefficients)


def test_unstandardized_density_reports_its_moments(standard_normal_2pl):
    # Without standardization the fitted population is no longer the
    # standard normal that latent_mean=None stands for.
    responses, _ = standard_normal_2pl
    density = MixtureDensity(standardize=False)
    result = EMEstimator(
        latent_density=density,
        n_quadpts=21,
        max_iter=30,
        compute_standard_errors=False,
    ).fit(TwoParameterLogistic(responses.shape[1]), responses)

    assert_allclose(result.latent_mean, [density.mean])
    assert_allclose(result.latent_covariance, [[density.variance]])
