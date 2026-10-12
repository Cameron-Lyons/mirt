"""Bock-Lieberman standard errors invert the full marginal Hessian."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import mirt
import mirt.estimation.bl as bl_module
from mirt.estimation._louis_information import louis_information
from mirt.estimation.bl import BLEstimator
from mirt.estimation.quadrature import build_quadrature
from mirt.estimation.standard_errors import (
    _flatten_parameters,
    _posterior_from_model,
    compute_observed_information,
)
from mirt.models.dichotomous import ComplementaryLogLog, TwoParameterLogistic
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
    quadrature = build_quadrature(15, 1)
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
    quadrature = build_quadrature(15, 1)
    posterior = _posterior_from_model(result.model, data, quadrature)
    information = compute_observed_information(
        result.model, data, posterior, quadrature
    )
    assert result.se_method == "hessian"
    np.testing.assert_allclose(
        result.vcov, np.linalg.inv(information), rtol=1e-10, atol=1e-14
    )


def test_bl_curve_product_models_use_exact_information():
    # Built-ins outside the pattern-compressible families previously
    # differenced the likelihood over every pair of parameters, 2P^2 + 1
    # likelihood evaluations.
    data = mirt.simdata(model="2PL", n_persons=800, n_items=6, seed=4)
    result = BLEstimator(n_quadpts=15, tol=1e-10).fit(ComplementaryLogLog(6), data)
    quadrature = build_quadrature(15, 1)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(result.model)
    information = louis_information(
        result.model, data, quadrature.nodes, mass, layouts
    ).information
    assert result.se_method == "hessian"
    np.testing.assert_allclose(
        result.vcov, np.linalg.inv(information), rtol=1e-10, atol=1e-14
    )

    pairwise = _OverriddenLikelihood(n_quadpts=15, tol=1e-10).fit(
        ComplementaryLogLog(6), data
    )
    for name, values in result.model.parameters.items():
        np.testing.assert_array_equal(pairwise.model.parameters[name], values)
    for name, values in result.standard_errors.items():
        np.testing.assert_allclose(pairwise.standard_errors[name], values, rtol=1e-3)


def test_bl_holds_bound_coordinates_fixed():
    estimator, model, data = _fit("2PL")
    result = estimator.fit(model, data)
    params, bounds, structure = estimator._flatten_parameters(result.model)
    # Move the first slope's upper limit onto its estimate, which keeps the
    # other coordinates at their constrained optimum. Pushing the slope out to
    # its limit of 20 instead turns item 1 into a step between quadrature
    # nodes, and whether its difficulty then keeps positive curvature depends
    # on the simulated responses.
    bounds[0] = (bounds[0][0], params[0])

    covariance = estimator._parameter_covariance(
        result.model, data, params, structure, bounds=bounds
    )
    errors = estimator._unflatten_standard_errors(result.model, covariance, structure)
    hessian = estimator._marginal_hessian(result.model, data, params, structure)

    assert np.all(np.isnan(covariance[0]))
    assert np.all(np.isnan(covariance[:, 0]))
    assert np.isnan(errors["discrimination"][0])
    np.testing.assert_allclose(
        covariance[1:, 1:], np.linalg.inv(hessian[1:, 1:]), rtol=1e-8, atol=1e-14
    )


@pytest.mark.performance
def test_bl_curve_product_errors_outpace_pairwise_differences():
    import time

    data = mirt.simdata(model="2PL", n_persons=1000, n_items=20, seed=3)
    estimator = BLEstimator(n_quadpts=21)
    model = estimator.fit(ComplementaryLogLog(20), data).model
    params, _, structure = estimator._flatten_parameters(model)

    def objective(candidate):
        estimator._unflatten_parameters(model, candidate, structure)
        return -estimator._compute_marginal_log_likelihood(model, data)

    start = time.perf_counter()
    exact = estimator._marginal_hessian(model, data, params, structure)
    exact_time = time.perf_counter() - start
    start = time.perf_counter()
    pairwise = estimator._marginal_hessian(
        model, data, params, structure, objective=objective
    )
    pairwise_time = time.perf_counter() - start
    estimator._unflatten_parameters(model, params, structure)

    np.testing.assert_allclose(exact, pairwise, rtol=0, atol=1e-3 * np.abs(exact).max())
    assert exact_time * 10 < pairwise_time


def _relative_tolerance_minimize(monkeypatch):
    """Make BL's optimizer stop on a relative 1e-6 change, as it used to."""
    original = bl_module.minimize

    def minimize(*args, options, **kwargs):
        return original(*args, options={**options, "ftol": 1e-6}, **kwargs)

    monkeypatch.setattr(bl_module, "minimize", minimize)


def _numpy_simdata(**options):
    """Simulate NumPy responses, which differ from the native backend's draws."""
    previous = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        return mirt.simdata(**options)
    finally:
        mirt.set_backend(previous)


def test_bl_reaches_the_3pl_optimum_that_relative_tolerances_missed(monkeypatch):
    # Regression: tol was L-BFGS-B's relative ftol, so on these data a 3PL
    # fit stopped 1.9 log-likelihood units short, with parameters off by 3.3
    # and errors off up to 15-fold, and still reported convergence. The
    # shortfall varies severalfold between data sets, so the responses are
    # the same with and without the native backend.
    from mirt.estimation.em import EMEstimator
    from mirt.models.dichotomous import ThreeParameterLogistic

    data = _numpy_simdata(model="3PL", n_persons=1000, n_items=8, seed=6)
    result = BLEstimator().fit(ThreeParameterLogistic(8), data)
    with monkeypatch.context() as patch:
        _relative_tolerance_minimize(patch)
        patch.setattr(bl_module, "_NEWTON_STEPS", 0)
        early = BLEstimator().fit(ThreeParameterLogistic(8), data)

    assert result.converged
    assert not early.converged
    assert result.log_likelihood > early.log_likelihood + 1.0
    # EM started at the BL estimate, on the same grid, barely moves.
    polished = EMEstimator(tol=1e-9, accelerate="squarem").fit(
        deepcopy(result.model), data, start="model"
    )
    assert polished.log_likelihood - result.log_likelihood < 1e-4


def test_bl_tolerances_are_in_log_likelihood_units():
    estimator = BLEstimator(tol=1e-6, max_iter=300)
    options = estimator._optimizer_options(-5000.0)
    assert options == {"maxiter": 300, "ftol": pytest.approx(2e-10)}
    assert estimator._optimizer_options(0.5) == {"maxiter": 300, "ftol": 1e-6}
    # TNC and SLSQP compare absolute changes; TNC counts evaluations.
    tnc = BLEstimator(tol=1e-6, max_iter=300, method="TNC")
    assert tnc._optimizer_options(-5000.0) == {"maxfun": 300, "ftol": 1e-6}
    slsqp = BLEstimator(tol=1e-6, max_iter=300, method="SLSQP")
    assert slsqp._optimizer_options(-5000.0) == {"maxiter": 300, "ftol": 1e-6}


def test_bl_reports_convergence_only_at_a_certified_optimum(monkeypatch):
    # Regression: converged echoed the optimizer's own stopping rule.
    data = mirt.simdata(model="GRM", n_persons=1000, n_items=6, n_categories=4, seed=2)
    reference = BLEstimator(tol=1e-9).fit(GradedResponseModel(6, 4), data)
    _relative_tolerance_minimize(monkeypatch)
    monkeypatch.setattr(bl_module, "_NEWTON_STEPS", 0)
    early = BLEstimator().fit(GradedResponseModel(6, 4), data)
    assert not early.converged
    assert reference.log_likelihood - early.log_likelihood > 1e-4

    # Newton steps on the exact information finish the solution instead.
    monkeypatch.setattr(bl_module, "_NEWTON_STEPS", 20)
    finished = BLEstimator().fit(GradedResponseModel(6, 4), data)
    assert finished.converged
    assert finished.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-6)


@pytest.mark.parametrize("method", ["BFGS", "CG", "trust-exact"])
def test_bl_rejects_methods_that_ignore_the_bounds(method):
    with pytest.raises(mirt.MirtValidationError, match="bounds"):
        BLEstimator(method=method)
