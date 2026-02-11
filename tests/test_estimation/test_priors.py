"""Tests for prior distributions module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from mirt.estimation.priors import (
    BetaPrior,
    CustomPrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    PriorSpecification,
    TruncatedNormalPrior,
    UniformPrior,
    compute_prior_log_pdf,
    default_priors,
    weakly_informative_priors,
)


class TestNormalPrior:
    def test_default_parameters(self):
        prior = NormalPrior()
        assert prior.mu == 0.0
        assert prior.sigma == 1.0

    def test_custom_parameters(self):
        prior = NormalPrior(mu=1.0, sigma=2.0)
        assert prior.mu == 1.0
        assert prior.sigma == 2.0

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            NormalPrior(sigma=0)
        with pytest.raises(ValueError, match="sigma must be positive"):
            NormalPrior(sigma=-1)

    def test_log_pdf_values(self):
        prior = NormalPrior(mu=0, sigma=1)
        x = np.array([0.0, 1.0, -1.0])
        expected = stats.norm(0, 1).logpdf(x)
        assert_allclose(prior.log_pdf(x), expected)

    def test_log_pdf_nonstandard(self):
        prior = NormalPrior(mu=2.0, sigma=0.5)
        x = np.array([1.5, 2.0, 2.5])
        expected = stats.norm(2.0, 0.5).logpdf(x)
        assert_allclose(prior.log_pdf(x), expected)

    def test_sample_shape_int(self, rng):
        prior = NormalPrior()
        samples = prior.sample(100, rng=rng)
        assert samples.shape == (100,)

    def test_sample_shape_tuple(self, rng):
        prior = NormalPrior()
        samples = prior.sample((10, 5), rng=rng)
        assert samples.shape == (10, 5)

    def test_sample_statistics(self, rng):
        prior = NormalPrior(mu=3.0, sigma=0.5)
        samples = prior.sample(10000, rng=rng)
        assert_allclose(np.mean(samples), 3.0, atol=0.1)
        assert_allclose(np.std(samples), 0.5, atol=0.1)

    def test_mean_property(self):
        prior = NormalPrior(mu=5.0, sigma=2.0)
        assert prior.mean == 5.0

    def test_variance_property(self):
        prior = NormalPrior(mu=0, sigma=3.0)
        assert prior.variance == 9.0

    def test_repr(self):
        prior = NormalPrior(mu=1.0, sigma=2.0)
        assert repr(prior) == "NormalPrior(mu=1.0, sigma=2.0)"


class TestTruncatedNormalPrior:
    def test_default_parameters(self):
        prior = TruncatedNormalPrior()
        assert prior.mu == 0.0
        assert prior.sigma == 1.0
        assert prior.lower == -np.inf
        assert prior.upper == np.inf

    def test_custom_parameters(self):
        prior = TruncatedNormalPrior(mu=1, sigma=0.5, lower=0, upper=3)
        assert prior.mu == 1
        assert prior.sigma == 0.5
        assert prior.lower == 0
        assert prior.upper == 3

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            TruncatedNormalPrior(sigma=0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="lower must be less than upper"):
            TruncatedNormalPrior(lower=5, upper=3)
        with pytest.raises(ValueError, match="lower must be less than upper"):
            TruncatedNormalPrior(lower=5, upper=5)

    def test_log_pdf_in_range(self):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        x = np.array([0.0, 1.0, -1.0])
        log_pdf = prior.log_pdf(x)
        assert np.all(np.isfinite(log_pdf))
        assert log_pdf[0] > log_pdf[1]

    def test_log_pdf_outside_range(self):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        x = np.array([-3.0, 3.0])
        log_pdf = prior.log_pdf(x)
        assert np.all(log_pdf == -np.inf)

    def test_sample_within_bounds(self, rng):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        samples = prior.sample(1000, rng=rng)
        assert samples.shape == (1000,)
        assert np.all(samples >= -2)
        assert np.all(samples <= 2)

    def test_mean_property(self):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        assert isinstance(prior.mean, float)

    def test_variance_property(self):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        assert isinstance(prior.variance, float)
        assert prior.variance > 0
        assert prior.variance < 1.0

    def test_repr(self):
        prior = TruncatedNormalPrior(mu=0, sigma=1, lower=-2, upper=2)
        assert "TruncatedNormalPrior" in repr(prior)


class TestLogNormalPrior:
    def test_default_parameters(self):
        prior = LogNormalPrior()
        assert prior.mu == 0.0
        assert prior.sigma == 0.5

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            LogNormalPrior(sigma=-1)

    def test_log_pdf_positive(self):
        prior = LogNormalPrior(mu=0, sigma=0.5)
        x = np.array([0.5, 1.0, 2.0])
        log_pdf = prior.log_pdf(x)
        assert np.all(np.isfinite(log_pdf))

    def test_log_pdf_matches_scipy(self):
        prior = LogNormalPrior(mu=0.5, sigma=0.3)
        x = np.array([1.0, 2.0, 3.0])
        expected = stats.lognorm(s=0.3, scale=np.exp(0.5)).logpdf(x)
        assert_allclose(prior.log_pdf(x), expected)

    def test_sample_positive(self, rng):
        prior = LogNormalPrior()
        samples = prior.sample(1000, rng=rng)
        assert np.all(samples > 0)

    def test_sample_shape(self, rng):
        prior = LogNormalPrior()
        samples = prior.sample((5, 3), rng=rng)
        assert samples.shape == (5, 3)

    def test_mean_property(self):
        prior = LogNormalPrior(mu=0, sigma=0.5)
        expected = np.exp(0 + 0.5**2 / 2)
        assert_allclose(prior.mean, expected)

    def test_variance_property(self):
        prior = LogNormalPrior(mu=0, sigma=0.5)
        expected = (np.exp(0.5**2) - 1) * np.exp(2 * 0 + 0.5**2)
        assert_allclose(prior.variance, expected)


class TestBetaPrior:
    def test_default_parameters(self):
        prior = BetaPrior()
        assert prior.alpha == 2.0
        assert prior.beta == 8.0

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            BetaPrior(alpha=0, beta=1)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            BetaPrior(alpha=1, beta=-1)

    def test_log_pdf_values(self):
        prior = BetaPrior(alpha=2, beta=5)
        x = np.array([0.1, 0.3, 0.5])
        expected = stats.beta(2, 5).logpdf(x)
        assert_allclose(prior.log_pdf(x), expected)

    def test_sample_range(self, rng):
        prior = BetaPrior(alpha=2, beta=8)
        samples = prior.sample(1000, rng=rng)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_mean_property(self):
        prior = BetaPrior(alpha=2, beta=8)
        assert_allclose(prior.mean, 0.2)

    def test_variance_property(self):
        prior = BetaPrior(alpha=2, beta=8)
        expected = (2 * 8) / (10**2 * 11)
        assert_allclose(prior.variance, expected)

    def test_repr(self):
        prior = BetaPrior(alpha=3, beta=7)
        assert repr(prior) == "BetaPrior(alpha=3, beta=7)"


class TestUniformPrior:
    def test_default_parameters(self):
        prior = UniformPrior()
        assert prior.lower == 0.0
        assert prior.upper == 1.0

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="lower must be less than upper"):
            UniformPrior(lower=5, upper=3)

    def test_log_pdf_in_range(self):
        prior = UniformPrior(lower=0, upper=4)
        x = np.array([1.0, 2.0, 3.0])
        expected = np.full(3, np.log(0.25))
        assert_allclose(prior.log_pdf(x), expected)

    def test_log_pdf_outside_range(self):
        prior = UniformPrior(lower=0, upper=1)
        x = np.array([-0.1, 1.1])
        log_pdf = prior.log_pdf(x)
        assert np.all(log_pdf == -np.inf)

    def test_sample_range(self, rng):
        prior = UniformPrior(lower=-3, upper=3)
        samples = prior.sample(1000, rng=rng)
        assert np.all(samples >= -3)
        assert np.all(samples <= 3)

    def test_mean_property(self):
        prior = UniformPrior(lower=2, upper=8)
        assert prior.mean == 5.0

    def test_variance_property(self):
        prior = UniformPrior(lower=0, upper=6)
        assert_allclose(prior.variance, 3.0)

    def test_repr(self):
        prior = UniformPrior(lower=-1, upper=1)
        assert repr(prior) == "UniformPrior(lower=-1, upper=1)"


class TestGammaPrior:
    def test_default_parameters(self):
        prior = GammaPrior()
        assert prior.shape == 1.0
        assert prior.rate == 1.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="shape and rate must be positive"):
            GammaPrior(shape=0, rate=1)

    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="shape and rate must be positive"):
            GammaPrior(shape=1, rate=-1)

    def test_log_pdf_values(self):
        prior = GammaPrior(shape=2, rate=3)
        x = np.array([0.5, 1.0, 2.0])
        expected = stats.gamma(a=2, scale=1 / 3).logpdf(x)
        assert_allclose(prior.log_pdf(x), expected)

    def test_sample_positive(self, rng):
        prior = GammaPrior(shape=2, rate=1)
        samples = prior.sample(1000, rng=rng)
        assert np.all(samples > 0)

    def test_mean_property(self):
        prior = GammaPrior(shape=3, rate=2)
        assert_allclose(prior.mean, 1.5)

    def test_variance_property(self):
        prior = GammaPrior(shape=3, rate=2)
        assert_allclose(prior.variance, 0.75)

    def test_repr(self):
        prior = GammaPrior(shape=2, rate=3)
        assert repr(prior) == "GammaPrior(shape=2, rate=3)"


class TestCustomPrior:
    def test_log_pdf(self):
        def log_pdf_fn(x):
            return -0.5 * x**2

        def sample_fn(size, rng):
            return np.zeros(size)

        prior = CustomPrior(log_pdf_fn, sample_fn, mean_value=0.0, variance_value=1.0)
        x = np.array([0.0, 1.0, -1.0])
        assert_allclose(prior.log_pdf(x), np.array([0.0, -0.5, -0.5]))

    def test_sample(self, rng):
        def sample_fn(size, rng):
            return np.ones(size)

        prior = CustomPrior(lambda x: x, sample_fn)
        samples = prior.sample(10, rng=rng)
        assert_allclose(samples, np.ones(10))

    def test_mean_property(self):
        prior = CustomPrior(lambda x: x, lambda s, r: np.zeros(s), mean_value=5.0)
        assert prior.mean == 5.0

    def test_variance_property(self):
        prior = CustomPrior(lambda x: x, lambda s, r: np.zeros(s), variance_value=2.0)
        assert prior.variance == 2.0

    def test_repr(self):
        prior = CustomPrior(
            lambda x: x,
            lambda s, r: np.zeros(s),
            mean_value=1.0,
            variance_value=3.0,
        )
        assert "CustomPrior" in repr(prior)


class TestPriorSpecification:
    def test_defaults(self):
        spec = PriorSpecification()
        assert isinstance(spec.discrimination, LogNormalPrior)
        assert isinstance(spec.difficulty, NormalPrior)
        assert isinstance(spec.theta, NormalPrior)
        assert spec.guessing is None
        assert spec.upper is None

    def test_custom_priors(self):
        disc = NormalPrior(mu=1, sigma=0.5)
        diff = NormalPrior(mu=0, sigma=1)
        spec = PriorSpecification(discrimination=disc, difficulty=diff)
        assert spec.discrimination is disc
        assert spec.difficulty is diff

    def test_with_guessing(self):
        guess = BetaPrior(alpha=2, beta=8)
        spec = PriorSpecification(guessing=guess)
        assert spec.guessing is guess


class TestDefaultPriors:
    def test_1pl(self):
        spec = default_priors("1PL")
        assert isinstance(spec.discrimination, LogNormalPrior)
        assert isinstance(spec.difficulty, NormalPrior)
        assert spec.guessing is None
        assert spec.upper is None

    def test_2pl(self):
        spec = default_priors("2PL")
        assert spec.guessing is None
        assert spec.upper is None

    def test_3pl(self):
        spec = default_priors("3PL")
        assert isinstance(spec.guessing, BetaPrior)
        assert spec.upper is None

    def test_4pl(self):
        spec = default_priors("4PL")
        assert isinstance(spec.guessing, BetaPrior)
        assert isinstance(spec.upper, BetaPrior)


class TestWeaklyInformativePriors:
    def test_all_set(self):
        spec = weakly_informative_priors()
        assert isinstance(spec.discrimination, LogNormalPrior)
        assert isinstance(spec.difficulty, NormalPrior)
        assert isinstance(spec.guessing, BetaPrior)
        assert isinstance(spec.upper, BetaPrior)
        assert isinstance(spec.theta, NormalPrior)

    def test_wider_than_default(self):
        default = default_priors("3PL")
        weak = weakly_informative_priors()
        assert weak.discrimination.sigma > default.discrimination.sigma
        assert weak.difficulty.sigma > default.difficulty.sigma


class TestComputePriorLogPdf:
    def test_discrimination_only(self):
        spec = PriorSpecification()
        disc = np.array([1.0, 1.5, 2.0])
        result = compute_prior_log_pdf(spec, discrimination=disc)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_difficulty_only(self):
        spec = PriorSpecification()
        diff = np.array([0.0, -1.0, 1.0])
        result = compute_prior_log_pdf(spec, difficulty=diff)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_with_guessing(self):
        spec = default_priors("3PL")
        disc = np.array([1.0])
        diff = np.array([0.0])
        guess = np.array([0.2])
        result = compute_prior_log_pdf(
            spec, discrimination=disc, difficulty=diff, guessing=guess
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_no_params_returns_zero(self):
        spec = PriorSpecification()
        result = compute_prior_log_pdf(spec)
        assert result == 0.0

    def test_guessing_ignored_without_prior(self):
        spec = default_priors("2PL")
        result_without = compute_prior_log_pdf(spec)
        result_with = compute_prior_log_pdf(spec, guessing=np.array([0.2]))
        assert result_without == result_with

    def test_multiple_params(self):
        spec = default_priors("4PL")
        disc = np.array([1.0, 1.5])
        diff = np.array([0.0, -1.0])
        guess = np.array([0.2, 0.15])
        upper = np.array([0.95, 0.9])
        result = compute_prior_log_pdf(
            spec,
            discrimination=disc,
            difficulty=diff,
            guessing=guess,
            upper=upper,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)
