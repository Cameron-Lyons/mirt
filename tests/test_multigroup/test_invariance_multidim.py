"""Nested invariance fits of two-factor multigroup models."""

import numpy as np
import pytest

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.multigroup import MultigroupEMEstimator, MultigroupModel


def _group_responses(seed=0, n_persons=500, n_items=10):
    rng = np.random.default_rng(seed)
    half = n_items // 2
    slopes = np.zeros((n_items, 2))
    slopes[:half, 0] = rng.uniform(0.8, 1.6, half)
    slopes[half:, 1] = rng.uniform(0.8, 1.6, n_items - half)
    intercepts = rng.normal(0.0, 1.0, n_items)
    responses = []
    for mean in ([0.0, 0.0], [-0.4, 0.3]):
        theta = rng.multivariate_normal(mean, [[1.0, 0.4], [0.4, 1.0]], n_persons)
        probability = 1.0 / (1.0 + np.exp(-(theta @ slopes.T + intercepts)))
        responses.append((rng.random(probability.shape) < probability).astype(int))
    return responses, n_items


def _fit(responses, n_items, invariance):
    model = MultigroupModel(TwoParameterLogistic(n_items=n_items, n_factors=2), 2)
    estimator = MultigroupEMEstimator(n_quadpts=11, max_iter=300)
    return estimator.fit(model, responses, invariance=invariance)


def test_configural_two_factor_fit_nests_the_metric_fit():
    responses, n_items = _group_responses()
    configural = _fit(responses, n_items, "configural")
    metric = _fit(responses, n_items, "metric")

    # Equal default slopes kept the configural factors identical, so its
    # log-likelihood fell below that of the nested metric model.
    assert configural.log_likelihood >= metric.log_likelihood - 1e-3
    for group in range(2):
        slopes = configural.model.get_group_model(group).parameters["discrimination"]
        assert np.max(np.abs(slopes[:, 0] - slopes[:, 1])) > 0.1


def _confirmatory_responses(seed=13, n_persons=400):
    """Simple structure, items 0-3 on F1 and 4-7 on F2, correlated factors."""
    rng = np.random.default_rng(seed)
    slopes = np.zeros((8, 2))
    slopes[:4, 0] = rng.uniform(1.2, 2.0, 4)
    slopes[4:, 1] = rng.uniform(1.2, 2.0, 4)
    intercepts = rng.normal(0.0, 0.8, 8)
    responses = []
    for mean, cov in (
        ([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]]),
        ([0.4, 0.2], [[1.2, 0.7], [0.7, 1.1]]),
    ):
        theta = rng.multivariate_normal(mean, cov, n_persons)
        probability = 1.0 / (1.0 + np.exp(-(theta @ slopes.T + intercepts)))
        responses.append((rng.random(probability.shape) < probability).astype(int))
    return responses


def _confirmatory_fit(responses, invariance, **kwargs):
    base = TwoParameterLogistic(n_items=8, n_factors=2)
    pattern = np.zeros((8, 2), dtype=bool)
    pattern[:4, 0] = True
    pattern[4:, 1] = True
    base.set_parameters(discrimination=np.where(pattern, 1.0, 0.0))
    base.set_free_parameter_masks({"discrimination": pattern})
    estimator = MultigroupEMEstimator(n_quadpts=11, max_iter=300, tol=1e-4)
    return estimator.fit(MultigroupModel(base, 2), responses, invariance, **kwargs)


def test_confirmatory_pattern_estimates_reference_factor_correlation():
    from mirt.multigroup import invariance_lrt

    responses = _confirmatory_responses()
    configural = _confirmatory_fit(responses, "configural")
    scalar = _confirmatory_fit(responses, "scalar")

    # The reference covariance was the identity, so the configural model was
    # orthogonal and fitted below the nested scalar model.
    reference = configural.latent_distributions[0]
    assert np.diag(reference.cov) == pytest.approx([1.0, 1.0])
    assert reference.cov[0, 1] == pytest.approx(0.6, abs=0.15)
    assert scalar.latent_distributions[0].cov[0, 1] == pytest.approx(0.6, abs=0.15)
    assert configural.log_likelihood >= scalar.log_likelihood
    # 32 item parameters and one correlation per group, against 16 item
    # parameters, the reference correlation and the focal mean and covariance.
    assert configural.n_parameters == 34
    assert scalar.n_parameters == 22
    assert invariance_lrt(scalar, configural)["df"] == 12


def test_free_correlations_option_fixes_or_rejects_correlations():
    from mirt.exceptions import MirtValidationError

    responses = _confirmatory_responses()
    orthogonal = _confirmatory_fit(responses, "configural", free_correlations=False)
    correlated = _confirmatory_fit(responses, "configural")

    assert orthogonal.latent_distributions[0].cov[0, 1] == 0.0
    assert orthogonal.n_parameters == correlated.n_parameters - 2
    assert correlated.log_likelihood > orthogonal.log_likelihood + 5

    exploratory = MultigroupModel(TwoParameterLogistic(n_items=8, n_factors=2), 2)
    with pytest.raises(MirtValidationError, match="does not identify factor"):
        MultigroupEMEstimator(n_quadpts=7, max_iter=2).fit(
            exploratory, responses, free_correlations=True
        )


def test_squarem_extrapolates_factor_correlations_and_covariances():
    responses = _confirmatory_responses()
    base = TwoParameterLogistic(n_items=8, n_factors=2)
    pattern = np.zeros((8, 2), dtype=bool)
    pattern[:4, 0] = True
    pattern[4:, 1] = True
    base.set_parameters(discrimination=np.where(pattern, 1.0, 0.0))
    base.set_free_parameter_masks({"discrimination": pattern})
    fits = {}
    for accelerate in ("none", "squarem"):
        estimator = MultigroupEMEstimator(
            n_quadpts=11, max_iter=500, tol=1e-6, accelerate=accelerate
        )
        fits[accelerate] = estimator.fit(
            MultigroupModel(base.copy(), 2), responses, "scalar"
        )
    plain, squarem = fits["none"], fits["squarem"]
    assert plain.converged and squarem.converged
    assert squarem.n_iterations < plain.n_iterations
    assert squarem.log_likelihood == pytest.approx(plain.log_likelihood, abs=1e-4)
    for accelerated, reference in zip(
        squarem.latent_distributions, plain.latent_distributions, strict=True
    ):
        np.testing.assert_allclose(accelerated.mean, reference.mean, atol=2e-3)
        np.testing.assert_allclose(accelerated.cov, reference.cov, atol=2e-3)
