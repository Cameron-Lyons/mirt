"""Owned-buffer normalization, extreme shifts, and nonfinite EM rows."""

import numpy as np
import pytest

from mirt.estimation import _posterior
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.utils.numeric import logsumexp


@pytest.mark.parametrize("blocked", [False, True])
def test_normalization_matches_independent_likelihood_reference(monkeypatch, blocked):
    rng = np.random.default_rng(219)
    likelihood = rng.normal(-500, 200, (17, 21))[:, ::2]
    prior = rng.random(11)
    prior /= prior.sum()
    expected_log = logsumexp(likelihood + np.log(prior), axis=1)
    expected = np.exp(likelihood + np.log(prior) - expected_log[:, None])
    if blocked:
        monkeypatch.setattr(_posterior, "_MAX_NORMALIZATION_ELEMENTS", 13)
    prior.flags.writeable = False
    actual, actual_log = normalize_log_posterior(likelihood, np.log(prior))
    assert actual is likelihood
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=1e-14)
    np.testing.assert_allclose(actual_log, expected_log, rtol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_unrepresentable_normalizing_offset_still_gives_unit_posterior(offset):
    # At this scale, log(3) cannot be represented accurately when added to the
    # offset. Subtracting the rounded marginal no longer gives unit row sums.
    joint = np.full((4, 3), offset)
    posterior, marginal = normalize_log_posterior(joint, np.zeros(3))
    np.testing.assert_allclose(posterior, 1.0 / 3.0, atol=0.0)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(marginal, offset + np.log(3.0))


def test_nonfinite_log_marginal_semantics_are_preserved():
    joint = np.array([[-np.inf, -np.inf], [np.inf, 0.0], [np.nan, 0.0]])
    posterior, marginal = normalize_log_posterior(joint, np.zeros(2))
    np.testing.assert_array_equal(marginal, [-np.inf, np.inf, np.nan])
    assert np.isnan(posterior).all()


@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_common_likelihood_offset_preserves_nonuniform_prior(offset):
    prior = np.array([0.05, 0.15, 0.8])
    posterior, marginal = normalize_log_posterior(
        np.full((4, 3), offset), np.log(prior)
    )
    np.testing.assert_allclose(posterior, np.tile(prior, (4, 1)), atol=1e-15)
    np.testing.assert_array_equal(marginal, offset)


def test_zero_prior_mass_excludes_largest_likelihood_without_underflow():
    posterior, marginal = normalize_log_posterior(
        np.array([[0.0, -1000.0]]), np.array([-np.inf, 0.0])
    )
    np.testing.assert_array_equal(posterior, [[0.0, 1.0]])
    np.testing.assert_array_equal(marginal, [-1000.0])


def test_opposite_infinities_propagate_undefined_log_joint():
    posterior, marginal = normalize_log_posterior(
        np.array([[np.inf, 0.0]]), np.array([-np.inf, 0.0])
    )
    assert np.isnan(posterior).all()
    assert np.isnan(marginal).all()


@pytest.mark.parametrize("kind", ["ordinary", "weighted"])
@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_estimator_e_steps_preserve_prior_with_large_custom_likelihood_offset(
    kind, offset
):
    model = TwoParameterLogistic(1)
    model.log_likelihood = lambda data, _theta: np.full(len(data), offset)
    model.log_likelihood_batch = lambda data, theta: np.full(
        (len(data), len(theta)), offset
    )
    responses = np.array([[0], [1]])
    mean, covariance = np.array([0.4]), np.array([[1.7]])
    density = GaussianDensity(mean=mean, cov=covariance)
    estimator = (
        EMEstimator(n_quadpts=7, use_gpu=False)
        if kind == "ordinary"
        else WeightedEMEstimator(n_quadpts=7)
    )
    estimator._quadrature = GaussHermiteQuadrature(n_points=7)
    if kind == "ordinary":
        estimator._latent_density = density
        posterior, marginal = estimator._e_step(model, responses)
    else:
        posterior, marginal = estimator._e_step_weighted(
            model, responses, mean, covariance, np.ones(2)
        )
    prior = np.exp(
        density.log_quadrature_mass(
            estimator._quadrature.nodes, estimator._quadrature.weights
        )
    )
    np.testing.assert_allclose(posterior, np.tile(prior, (2, 1)), atol=1e-15)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(marginal, offset)
