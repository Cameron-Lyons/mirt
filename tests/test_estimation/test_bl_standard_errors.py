"""Bock-Lieberman standard errors invert the full marginal Hessian."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import mirt
import mirt.estimation.bl as bl_module
from mirt.estimation.bl import BLEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import (
    _posterior_from_model,
    compute_observed_information,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _fit(kind):
    if kind == "2PL":
        data = mirt.simdata(model="2PL", n_persons=1500, n_items=6, seed=12)
        model = TwoParameterLogistic(6)
    else:
        data = mirt.simdata(
            model="GRM", n_persons=1500, n_items=5, n_categories=4, seed=12
        )
        model = GradedResponseModel(5, n_categories=4)
    return BLEstimator(n_quadpts=15, tol=1e-12, max_iter=2000), model, data


@pytest.mark.parametrize("kind", ["2PL", "GRM"])
def test_bl_errors_invert_the_full_observed_information(kind):
    estimator, model, data = _fit(kind)
    result = estimator.fit(model, data)
    quadrature = GaussHermiteQuadrature(15)
    posterior = _posterior_from_model(result.model, data, quadrature)
    information = compute_observed_information(
        result.model, data, posterior, quadrature
    )
    covariance = np.linalg.inv(information)
    flat = np.concatenate(
        [
            result.standard_errors[name][mask]
            for name, mask in result.model.free_parameter_masks.items()
        ]
    )

    # The former diagonal-Hessian errors, sqrt(1 / H_ii), were up to 60% small.
    assert np.max(flat / np.sqrt(1.0 / np.diag(information))) > 1.05
    np.testing.assert_allclose(flat, np.sqrt(np.diag(covariance)), rtol=1e-8)
    assert result.se_method == "hessian"
    np.testing.assert_allclose(result.vcov, covariance, rtol=1e-8, atol=1e-12)
    assert result.vcov_labels[0] == f"discrimination[{result.model.item_names[0]}]"


class _OverriddenLikelihood(BLEstimator):
    """An estimator-specific likelihood, which exact item terms cannot describe."""

    def _compute_marginal_log_likelihood(self, model, responses):
        return super()._compute_marginal_log_likelihood(model, responses)


def test_bl_likelihood_differences_match_exact_information():
    estimator, model, data = _fit("2PL")
    analytic = estimator.fit(deepcopy(model), data)
    numeric = _OverriddenLikelihood(n_quadpts=15, tol=1e-12, max_iter=2000).fit(
        deepcopy(model), data
    )
    for name, values in analytic.standard_errors.items():
        np.testing.assert_allclose(numeric.standard_errors[name], values, rtol=2e-3)


def test_bl_without_gradients_still_uses_exact_information(monkeypatch):
    # Optimizers without analytic gradients previously fell back to O(P^2)
    # likelihood differences even for built-in item models.
    monkeypatch.setattr(bl_module, "prepare_bl_objective", lambda *args: None)
    estimator, model, data = _fit("2PL")
    result = estimator.fit(model, data)
    quadrature = GaussHermiteQuadrature(15)
    posterior = _posterior_from_model(result.model, data, quadrature)
    information = compute_observed_information(
        result.model, data, posterior, quadrature
    )
    assert result.se_method == "hessian"
    np.testing.assert_allclose(
        result.vcov, np.linalg.inv(information), rtol=1e-10, atol=1e-14
    )


def test_bl_holds_bound_coordinates_fixed():
    estimator, model, data = _fit("2PL")
    result = estimator.fit(model, data)
    params, bounds, structure = estimator._flatten_parameters(result.model)
    params[0] = bounds[0][1]
    estimator._unflatten_parameters(result.model, params, structure)

    covariance = estimator._parameter_covariance(
        result.model, data, params, structure, bounds=bounds
    )
    errors = estimator._unflatten_standard_errors(result.model, covariance, structure)

    assert np.all(np.isnan(covariance[0]))
    assert np.isnan(errors["discrimination"][0])
    assert np.all(np.isfinite(np.diag(covariance)[1:]))
