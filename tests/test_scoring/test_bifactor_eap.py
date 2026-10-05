"""Dimension-reduced EAP scoring of bifactor models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_hermite

import mirt
from mirt.models import BifactorModel
from mirt.scoring import ability_posterior, fscores
from mirt.scoring._bifactor import reduced_bifactor_grid
from mirt.scoring.eap import EAPScorer


def _product_grid_eap(model, responses, n_points, mean=None, cov=None):
    """Reference EAP on the full ``n_points ** n_factors`` product grid."""
    n_factors = model.n_factors
    mean = np.zeros(n_factors) if mean is None else np.asarray(mean)
    cov = np.eye(n_factors) if cov is None else np.asarray(cov)
    nodes, weights = roots_hermite(n_points)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    grids = np.meshgrid(*([nodes] * n_factors), indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    points = points @ np.linalg.cholesky(cov).T + mean
    weight_grids = np.meshgrid(*([weights] * n_factors), indexing="ij")
    log_weights = np.sum(np.log(weight_grids), axis=0).ravel()
    log_posterior = model.log_likelihood_batch(responses, points) + log_weights
    log_posterior -= log_posterior.max(axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum(axis=1, keepdims=True)
    theta = posterior @ points
    variance = posterior @ points**2 - theta**2
    return theta, np.sqrt(np.maximum(variance, 0.0))


def _model(n_specific: int, items_per_factor: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_items = n_specific * items_per_factor
    # Sparse, unsorted labels exercise the label-to-column mapping.
    labels = np.array([7, 2, 30, 11, 5][:n_specific])
    model = BifactorModel(n_items, np.tile(labels, items_per_factor))
    model.set_parameters(
        general_loadings=rng.uniform(1.2, 2.2, n_items),
        specific_loadings=rng.uniform(0.6, 1.6, n_items),
        intercepts=rng.normal(size=n_items),
    )
    model._is_fitted = True
    return model


def _responses(model, n_persons: int = 40, seed: int = 1):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_persons, model.n_factors))
    responses = model.simulate(theta, seed=seed)
    responses[rng.random(responses.shape) < 0.1] = -1
    responses[0] = -1
    return responses


def _bifactor_data(n_specific: int, n_persons: int = 400, seed: int = 2024):
    rng = np.random.default_rng(seed)
    specific = np.repeat(np.arange(n_specific), 4)
    n_items = specific.size
    general = rng.normal(size=n_persons)
    specific_theta = rng.normal(size=(n_persons, n_specific))
    logits = (
        rng.uniform(1.4, 2.2, n_items) * general[:, None]
        + rng.uniform(1.0, 1.6, n_items) * specific_theta[:, specific]
        + rng.normal(size=n_items)
    )
    data = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    data[rng.random(data.shape) < 0.05] = -1
    return data, specific


def _general_specific_covariance(variance, cross, residual):
    """Covariance whose specific factors are independent given the general."""
    cross = np.asarray(cross, dtype=float)
    cov = np.diag(np.concatenate(([variance], residual)))
    cov[0, 1:] = cov[1:, 0] = cross
    cov[1:, 1:] += np.outer(cross, cross) / variance
    return cov


# Priors under which the specific factors are independent given the general
# factor: standard, diagonal with a mean, and general-specific covariances.
_CONDITIONALLY_INDEPENDENT_PRIORS = {
    "standard": (None, None),
    "diagonal": ([0.3, -0.2, 0.1, 0.4], np.diag([1.5, 0.8, 1.2, 0.6])),
    "general-specific": (
        [0.1, 0.0, -0.3, 0.2],
        _general_specific_covariance(1.3, [0.5, -0.3, 0.2], [0.9, 1.1, 0.7]),
    ),
}


@pytest.mark.parametrize("prior", sorted(_CONDITIONALLY_INDEPENDENT_PRIORS))
@pytest.mark.parametrize("n_quadpts", [5, 7])
def test_reduced_eap_equals_product_grid_eap(prior, n_quadpts):
    model = _model(n_specific=3)
    responses = _responses(model)
    mean, cov = _CONDITIONALLY_INDEPENDENT_PRIORS[prior]

    scorer = EAPScorer(n_quadpts=n_quadpts, prior_mean=mean, prior_cov=cov)
    assert scorer._bifactor_grid(model) is not None
    result = scorer.score(model, responses)

    theta, standard_error = _product_grid_eap(model, responses, n_quadpts, mean, cov)
    assert_allclose(result.theta, theta, rtol=0, atol=1e-12)
    assert_allclose(result.standard_error, standard_error, rtol=0, atol=1e-11)


def test_reduced_eap_respects_batches_and_repeated_patterns():
    model = _model(n_specific=2)
    responses = np.tile(_responses(model, n_persons=15), (3, 1))

    batched = EAPScorer(n_quadpts=11, batch_size=4).score(model, responses)
    theta, standard_error = _product_grid_eap(model, responses, 11)

    assert_allclose(batched.theta, theta, rtol=0, atol=1e-12)
    assert_allclose(batched.standard_error, standard_error, rtol=0, atol=1e-11)


def test_correlated_specific_factors_keep_the_product_grid():
    model = _model(n_specific=2)
    responses = _responses(model, n_persons=12)
    cov = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.4], [0.1, 0.4, 1.0]])

    scorer = EAPScorer(n_quadpts=9, prior_cov=cov)
    result = scorer.score(model, responses)

    assert scorer._bifactor_grid(model) is None
    theta, standard_error = _product_grid_eap(model, responses, 9, cov=cov)
    assert_allclose(result.theta, theta, rtol=0, atol=1e-12)
    assert_allclose(result.standard_error, standard_error, rtol=0, atol=1e-11)


def test_custom_bifactor_subclass_is_not_reduced():
    class ScaledBifactor(BifactorModel):
        pass

    model = ScaledBifactor(4, [0, 0, 1, 1])
    mean, cov = np.zeros(3), np.eye(3)

    assert reduced_bifactor_grid(model, 9, mean, cov) is None
    assert reduced_bifactor_grid(_model(2), 9, mean, cov) is not None


def test_bfactor_scores_match_a_fine_product_grid():
    """Default EAP of a bfactor fit with three specific factors is accurate.

    The former four-factor default, a 9-point product grid, is off by about
    0.03 here.
    """
    data, specific = _bifactor_data(n_specific=3)
    result = mirt.bfactor(data, specific, compute_standard_errors=False)
    responses = data[:25]

    scores = fscores(result, responses)

    theta, standard_error = _product_grid_eap(result.model, responses, 21)
    assert_allclose(scores.theta, theta, rtol=0, atol=0.01)
    assert_allclose(scores.standard_error, standard_error, rtol=0, atol=0.01)


def test_many_specific_factors_score_on_the_fine_default_grid():
    model = _model(n_specific=5, items_per_factor=3)
    responses = _responses(model, n_persons=20)

    default = fscores(model, responses)
    explicit = fscores(model, responses, n_quadpts=49)

    np.testing.assert_array_equal(default.theta, explicit.theta)
    np.testing.assert_array_equal(default.standard_error, explicit.standard_error)
    # Six factors once defaulted to a 5-point product grid.
    coarse = EAPScorer(n_quadpts=5).score(model, responses)
    assert np.max(np.abs(coarse.theta - default.theta)) > 0.05


def test_coarse_automatic_grid_warns_when_the_prior_prevents_reduction():
    model = _model(n_specific=3)
    responses = _responses(model, n_persons=5)
    cov = np.eye(4)
    cov[1, 2] = cov[2, 1] = 0.3

    with pytest.warns(RuntimeWarning, match="9 points per dimension"):
        coarse = fscores(model, responses, prior_cov=cov)

    theta, _ = _product_grid_eap(model, responses, 9, cov=cov)
    assert_allclose(coarse.theta, theta, rtol=0, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fscores(model, responses, prior_cov=cov, n_quadpts=9)
        fscores(model, responses)


def test_bifactor_posterior_warns_about_a_coarse_automatic_grid():
    model = _model(n_specific=3)
    responses = _responses(model, n_persons=5)

    with pytest.warns(RuntimeWarning, match="dimension reduction"):
        ability_posterior(model, responses)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        posterior = ability_posterior(model, responses, n_quadpts=7)
        two_factor_posterior = ability_posterior(_model(n_specific=1), responses[:, :4])
    theta, _ = _product_grid_eap(model, responses, 7)
    assert_allclose(posterior.mean, theta, rtol=0, atol=1e-12)
    assert two_factor_posterior.n_factors == 2
