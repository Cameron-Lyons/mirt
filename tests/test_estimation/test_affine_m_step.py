"""Analytic affine M-step gradients, clipping, constraints, and thread isolation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from mirt.estimation import em as em_module
from mirt.estimation._affine_objective import prepare_affine_objective
from mirt.estimation.em import EMEstimator
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.multidimensional import MultidimensionalModel


@pytest.mark.parametrize("kind", ["mirt", "bifactor"])
@pytest.mark.parametrize("parallel", [False, True])
def test_private_theta_overrides_use_the_public_curve(monkeypatch, kind, parallel):
    model = _model(kind)

    def transform(self, theta):
        return BaseItemModel._ensure_theta_2d(self, theta) * 1.2 + 0.4

    model._ensure_theta_2d = MethodType(transform, model)
    original = model.parameters
    estimator = EMEstimator(use_rust=False, use_gpu=False)
    theta = np.random.default_rng(915).normal(size=(4, model.n_factors))
    responses = np.array([[0, 1, 0], [1, 1, 1], [-1, 0, 1]])
    posterior = np.full((3, 4), 0.25)

    def optimize(objective, x0, *, jac, **kwargs):
        assert not jac
        trial = x0 + 0.2
        value = objective(trial)
        local = deepcopy(model)
        estimator._set_item_params(local, 0, trial)
        p = np.clip(local.probability(theta, 0), 1e-10, 1 - 1e-10)
        expected = -np.sum(0.25 * np.log(p) + 0.25 * np.log1p(-p))
        np.testing.assert_allclose(value, expected, atol=1e-13)
        return SimpleNamespace(x=trial)

    monkeypatch.setattr(em_module, "minimize", optimize)
    method = (
        estimator._optimize_item_return if parallel else estimator._optimize_item_params
    )
    method(model, 0, responses, posterior, theta)
    for name in original:
        np.testing.assert_array_equal(model.parameters[name], original[name])


def _model(kind):
    if kind == "bifactor":
        return BifactorModel(3, [5, 23, 5]).set_parameters(
            general_loadings=np.array([0.7, 1.1, 0.9]),
            specific_loadings=np.array([-0.5, 0.4, 0.8]),
            intercepts=np.array([-0.2, 0.1, 0.4]),
        )
    pattern = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    kwargs = (
        {}
        if kind == "mirt"
        else dict(model_type="confirmatory", loading_pattern=pattern)
    )
    return MultidimensionalModel(3, 3, **kwargs).set_parameters(
        slopes=np.array([[0.7, 1.1, 1.3], [0.9, 0.4, 1.2], [0.5, 0.6, 0.8]]),
        intercepts=np.array([-0.2, 0.1, 0.4]),
    )


def _public_objective(model, item, theta, observed, correct, epsilon, params):
    working = model.copy()
    EMEstimator()._set_item_params(working, item, params)
    p = np.clip(working.probability(theta, item), epsilon, 1.0 - epsilon)
    return -np.sum(correct * np.log(p) + (observed - correct) * np.log1p(-p))


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
@pytest.mark.parametrize("item", [0, 2])
@pytest.mark.parametrize("epsilon", [1e-10, 0.1])
def test_affine_gradient_matches_public_clipped_likelihood(kind, item, epsilon):
    model = _model(kind)
    original = model.parameters
    rng = np.random.default_rng(614)
    theta = rng.normal(size=(29, model.n_factors)) * 3.0
    observed = rng.uniform(0.0, 10.0, 29)
    correct = observed * rng.random(29)
    observed[::4] = correct[::4] = 0.0
    observed.flags.writeable = correct.flags.writeable = theta.flags.writeable = False
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, item)
    objective = prepare_affine_objective(
        model, item, theta, observed, correct, epsilon, bounds
    )
    assert objective is not None
    actual, gradient = objective(params)
    expected = _public_objective(model, item, theta, observed, correct, epsilon, params)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    numerical = np.empty_like(params)
    for coordinate in range(len(params)):
        offset = np.zeros_like(params)
        offset[coordinate] = 1e-5
        high = _public_objective(
            model, item, theta, observed, correct, epsilon, params + offset
        )
        low = _public_objective(
            model, item, theta, observed, correct, epsilon, params - offset
        )
        numerical[coordinate] = (high - low) / 2e-5
    np.testing.assert_allclose(gradient, numerical, rtol=3e-7, atol=2e-8)
    for key, value in original.items():
        np.testing.assert_array_equal(model.parameters[key], value)


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
def test_fixed_coordinates_keep_the_affine_objective(kind):
    # A masked loading used to send the item to numerical differentiation.
    model = _model(kind)
    masks = model.free_parameter_masks
    loadings = "slopes" if "slopes" in masks else "general_loadings"
    row = masks[loadings].reshape(model.n_items, -1)[1]
    row[np.flatnonzero(row)[0]] = False
    model.set_free_parameter_masks(masks)
    rng = np.random.default_rng(615)
    theta = rng.normal(size=(29, model.n_factors)) * 3.0
    observed = rng.uniform(0.0, 10.0, 29)
    correct = observed * rng.random(29)
    estimator = EMEstimator()
    params, _, objective, analytic = estimator._item_objective(
        model, 1, np.zeros((1, 3), dtype=int), None, theta, None, correct, observed
    )
    assert analytic
    params = params + 0.1
    actual, gradient = objective(params)
    expected = _public_objective(model, 1, theta, observed, correct, 1e-10, params)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    numerical = np.empty_like(params)
    for coordinate in range(len(params)):
        offset = np.zeros_like(params)
        offset[coordinate] = 1e-5
        numerical[coordinate] = (
            _public_objective(
                model, 1, theta, observed, correct, 1e-10, params + offset
            )
            - _public_objective(
                model, 1, theta, observed, correct, 1e-10, params - offset
            )
        ) / 2e-5
    np.testing.assert_allclose(gradient, numerical, rtol=3e-7, atol=2e-8)


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
def test_affine_clipped_tails_have_zero_gradient(kind):
    model = _model(kind)
    item = 0
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, item)
    theta = np.zeros((4, model.n_factors))
    theta[:, 0] = [-1000.0, -40.0, 40.0, 1000.0]
    observed = np.array([3.0, 5.0, 2.0, 4.0])
    correct = np.array([2.0, 4.0, 1.0, 2.0])
    objective = prepare_affine_objective(
        model, item, theta, observed, correct, 1e-8, bounds
    )
    assert objective is not None
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = objective(params)
    expected = _public_objective(model, item, theta, observed, correct, 1e-8, params)
    np.testing.assert_allclose(loss, expected, rtol=1e-14)
    np.testing.assert_array_equal(gradient, 0.0)


def test_large_quadrature_coordinates_use_exact_logit_recovery():
    model = BifactorModel(1, [8]).set_parameters(
        general_loadings=np.array([5.0]),
        specific_loadings=np.array([-5.0]),
        intercepts=np.array([0.3]),
    )
    theta = np.array([[1e308, 1e308]])
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    objective = prepare_affine_objective(
        model, 0, theta, np.array([1.0]), np.array([0.0]), 1e-10, bounds
    )
    assert objective is not None
    probability = 1.0 / (1.0 + np.exp(-0.3))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = objective(params)
    np.testing.assert_allclose(loss, -np.log1p(-probability), rtol=1e-14)
    np.testing.assert_allclose(
        gradient, [1e308 * probability, 1e308 * probability, probability], rtol=1e-14
    )


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
def test_analytic_and_numeric_item_optimizers_agree(kind):
    model = _model(kind)
    rng = np.random.default_rng(63)
    theta = rng.normal(size=(41, model.n_factors))
    observed = rng.uniform(1.0, 15.0, 41)
    correct = observed * model.probability(theta, 0)
    initial, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    initial = initial + 0.1
    objective = prepare_affine_objective(
        model, 0, theta, observed, correct, 1e-10, bounds
    )
    assert objective is not None

    def numerical(params):
        return _public_objective(model, 0, theta, observed, correct, 1e-10, params)

    analytic_fit = minimize(
        objective,
        initial,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    numeric_fit = minimize(
        numerical,
        initial,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    np.testing.assert_allclose(analytic_fit.fun, numeric_fit.fun, rtol=1e-11)
    np.testing.assert_allclose(analytic_fit.x, numeric_fit.x, rtol=2e-5, atol=2e-5)
    assert analytic_fit.nfev < numeric_fit.nfev


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
def test_prepared_item_optimization_does_not_mutate_model(kind):
    model = _model(kind)
    original = model.parameters
    rng = np.random.default_rng(732)
    responses = rng.integers(0, 2, (18, 3))
    responses[::3, 0] = -1
    theta = rng.normal(size=(11, model.n_factors))
    posterior = rng.dirichlet(np.ones(11), size=18)
    estimator = EMEstimator(use_rust=False)
    result = estimator._optimize_item_params(model, 0, responses, posterior, theta)
    assert np.isfinite(result).all()
    for key in original:
        np.testing.assert_array_equal(model.parameters[key], original[key])


class _PowerMIRT(MultidimensionalModel):
    def probability(self, theta, item_idx=None):
        return super().probability(theta, item_idx) ** 2


@pytest.mark.parametrize("change", ["subclass", "instance", "weighted_pattern"])
def test_custom_curves_retain_numerical_objective(change):
    model = _PowerMIRT(1, 2) if change == "subclass" else MultidimensionalModel(1, 2)
    if change == "instance":
        model.probability = lambda theta, item_idx=None: np.full(len(theta), 0.25)
    elif change == "weighted_pattern":
        model = MultidimensionalModel(
            1, 2, model_type="confirmatory", loading_pattern=np.array([[0.5, 1.0]])
        )
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    assert (
        prepare_affine_objective(
            model, 0, np.zeros((2, 2)), np.ones(2), np.ones(2), 1e-10, bounds
        )
        is None
    )


@pytest.mark.parametrize("customization", ["subclass", "instance"])
def test_parallel_numerical_objectives_do_not_mutate_shared_model(
    monkeypatch, customization
):
    model = (
        _PowerMIRT(2, 2) if customization == "subclass" else MultidimensionalModel(2, 2)
    )
    if customization == "instance":
        model.probability = lambda theta, item_idx=None: np.full(len(theta), 0.25)
    original = model.parameters
    responses = np.array([[0, 1], [1, 0], [1, 1]])
    theta = np.array([[-1.0, 0.3], [0.2, 1.0], [1.0, -0.4]])
    posterior = np.full((3, 3), 1.0 / 3.0)

    def one_trial(objective, x0, **kwargs):
        trial = x0 + 0.2
        value = objective(trial)
        assert np.isfinite(value)
        if customization == "instance":
            np.testing.assert_allclose(value, -2 * np.log(0.25) - np.log(0.75))
        return SimpleNamespace(x=trial)

    monkeypatch.setattr(em_module, "minimize", one_trial)
    estimator = EMEstimator(use_rust=False)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: estimator._optimize_item_return(
                    model, item, responses, posterior, theta
                ),
                range(2),
            )
        )
    assert len(results) == 2
    for key in original:
        np.testing.assert_array_equal(model.parameters[key], original[key])


@pytest.mark.parametrize("kind", ["mirt", "bifactor", "confirmatory"])
def test_affine_fits_match_across_worker_counts_with_missing_responses(kind):
    model = _model(kind)
    rng = np.random.default_rng(222)
    responses = rng.integers(0, 2, (50, model.n_items))
    responses[::4, 1] = -1
    responses[:, 2] = -1
    kwargs = dict(
        n_quadpts=5,
        max_iter=4,
        tol=1e-7,
        use_rust=False,
        use_gpu=False,
        compute_standard_errors=False,
    )
    serial = EMEstimator(n_jobs=1, **kwargs).fit(model.copy(), responses)
    parallel = EMEstimator(n_jobs=2, **kwargs).fit(model.copy(), responses)
    np.testing.assert_allclose(
        serial.log_likelihood, parallel.log_likelihood, rtol=1e-13
    )
    for key in serial.model.parameters:
        np.testing.assert_allclose(
            serial.model.parameters[key],
            parallel.model.parameters[key],
            rtol=1e-12,
            atol=1e-12,
        )
    if kind == "confirmatory":
        assert parallel.model.n_parameters == 7
        np.testing.assert_array_equal(
            parallel.model.slopes[model.loading_pattern == 0.0], 0.0
        )


def test_confirmatory_copy_preserves_constraints_and_owns_pattern():
    pattern = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = MultidimensionalModel(
        2, 2, model_type="confirmatory", loading_pattern=pattern, item_names=["A", "B"]
    )
    model._is_fitted = True
    pattern[:] = 1.0
    clone = model.copy()
    assert clone.model_type == "confirmatory"
    assert clone.is_fitted
    assert model.n_parameters == clone.n_parameters == 4
    np.testing.assert_array_equal(clone.loading_pattern, [[1, 0], [0, 1]])
    clone.set_item_parameter(0, "slopes", np.array([2.0, 3.0]))
    np.testing.assert_array_equal(clone.slopes[0], [2.0, 0.0])
    np.testing.assert_array_equal(model.slopes[0], [0.8, 0.0])
    clone.item_names[0] = "C"
    assert model.item_names[0] == "A"


def test_bifactor_copy_preserves_subclass_probability():
    class PowerBifactor(BifactorModel):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx) ** 2

    model = PowerBifactor(2, [4, 9])
    clone = model.copy()
    assert type(clone) is PowerBifactor
    theta = np.array([[0.2, -0.5, 0.8]])
    np.testing.assert_array_equal(clone.probability(theta), model.probability(theta))


@pytest.mark.parametrize("observed", [0.0, 3.0])
def test_zero_counts_do_not_multiply_saturated_log_probabilities(observed):
    model = MultidimensionalModel(1, 2)
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    objective = prepare_affine_objective(
        model,
        0,
        np.array([[1000.0, 0.0]]),
        np.array([observed]),
        np.array([observed]),
        1e-20,
        bounds,
    )
    assert objective is not None
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = objective(params)
    assert loss == 0.0
    np.testing.assert_array_equal(gradient, 0.0)
