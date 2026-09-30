"""Independent likelihood, gradient, and optimization checks for category items."""

import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

import mirt.estimation.em as em_module
from mirt.estimation._polytomous_objective import prepare_polytomous_objective
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)

_MODELS = [
    (GradedResponseModel, 1),
    (GradedResponseModel, 2),
    (GradedResponseModel, 3),
    (GeneralizedPartialCredit, 1),
    (GeneralizedPartialCredit, 2),
    (GeneralizedPartialCredit, 3),
    (PartialCreditModel, 1),
    (NominalResponseModel, 1),
    (NominalResponseModel, 2),
    (NominalResponseModel, 3),
]


def _reference(model, item, theta, counts, epsilon, params):
    local = model.copy()
    EMEstimator()._set_item_params(local, item, params)
    probability = np.clip(local.probability(theta, item), epsilon, 1.0 - epsilon)
    return -float(np.sum(counts * np.log(probability)))


@pytest.mark.parametrize(("factory", "factors"), _MODELS)
@pytest.mark.parametrize("categories", [2, 5])
@pytest.mark.parametrize("epsilon", [1e-10, 0.1])
def test_objective_and_gradient_match_public_curves(
    factory, factors, categories, epsilon
):
    rng = np.random.default_rng(729)
    model = factory(2, n_categories=[2, categories], n_factors=factors)
    estimator = EMEstimator()
    params, _ = estimator._get_item_params_and_bounds(model, 1)
    params += rng.uniform(-0.1, 0.1, params.size)
    theta = rng.normal(size=(43, factors)) * 3.0
    counts = rng.uniform(0.0, 5.0, (43, categories))
    counts[::7] = 0.0
    theta.setflags(write=False)
    counts.setflags(write=False)
    objective = prepare_polytomous_objective(model, 1, theta, counts, epsilon)
    assert objective is not None
    original = model.parameters
    value, gradient = objective(params)

    def reference(trial):
        return _reference(model, 1, theta, counts, epsilon, trial)

    assert value == pytest.approx(reference(params), rel=1e-12, abs=1e-12)
    expected = np.zeros_like(params)
    for coordinate in range(params.size):
        delta = np.zeros_like(params)
        delta[coordinate] = 1e-5
        expected[coordinate] = (
            reference(params + delta) - reference(params - delta)
        ) / 2e-5
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=2e-7)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(("factory", "factors"), _MODELS)
def test_optimizer_recovers_interior_category_curves(factory, factors):
    rng = np.random.default_rng(328)
    truth = factory(1, n_categories=4, n_factors=factors)
    estimator = EMEstimator()
    target, bounds = estimator._get_item_params_and_bounds(truth, 0)
    target += rng.uniform(-0.2, 0.2, target.size)
    estimator._set_item_params(truth, 0, target)
    theta = rng.normal(size=(150, factors)) * 1.4
    counts = 200 * truth.probability(theta, 0)
    model = factory(1, n_categories=4, n_factors=factors)
    initial, _ = estimator._get_item_params_and_bounds(model, 0)
    objective = prepare_polytomous_objective(model, 0, theta, counts, 1e-10)
    result = minimize(
        objective,
        initial,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-7},
    )
    assert np.isfinite(result.fun)
    np.testing.assert_allclose(result.x, target, rtol=1e-4, atol=5e-5)
    assert result.fun <= _reference(model, 0, theta, counts, 1e-10, initial)


@pytest.mark.parametrize("factors", [1, 2])
def test_crossed_graded_thresholds_keep_clipped_gradient(factors):
    model = GradedResponseModel(1, n_categories=4, n_factors=factors)
    params, _ = EMEstimator()._get_item_params_and_bounds(model, 0)
    params[factors:] = [1.1, -0.8, 0.3]
    theta = np.linspace(-3, 3, 21 * factors).reshape(21, factors)
    counts = np.arange(84, dtype=float).reshape(21, 4) / 10
    objective = prepare_polytomous_objective(model, 0, theta, counts, 1e-3)
    value, gradient = objective(params)
    assert value == pytest.approx(_reference(model, 0, theta, counts, 1e-3, params))
    for index in range(params.size):
        delta = np.zeros_like(params)
        delta[index] = 1e-5
        numerical = (
            _reference(model, 0, theta, counts, 1e-3, params + delta)
            - _reference(model, 0, theta, counts, 1e-3, params - delta)
        ) / 2e-5
        assert gradient[index] == pytest.approx(numerical, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize(("factory", "factors"), _MODELS)
def test_clip_boundary_has_zero_count_derivative(factory, factors):
    model = factory(1, n_categories=4, n_factors=factors)
    theta = np.zeros((1, factors))
    params, _ = EMEstimator()._get_item_params_and_bounds(model, 0)
    probabilities = model.probability(theta, 0)
    category = np.argmin(probabilities[0])
    epsilon = probabilities[0, category]
    counts = np.zeros((1, 4))
    counts[0, category] = 1.0
    objective = prepare_polytomous_objective(model, 0, theta, counts, epsilon)
    value, gradient = objective(params)
    assert value == pytest.approx(-np.log(epsilon))
    np.testing.assert_array_equal(gradient, 0.0)


def test_nominal_reference_and_padding_are_canonical_without_model_mutation():
    model = NominalResponseModel(2, n_categories=[2, 4], n_factors=2)
    rng = np.random.default_rng(817)
    model.set_parameters(
        slopes=rng.normal(size=(2, 4, 2)), intercepts=rng.normal(size=(2, 4))
    )
    theta = rng.normal(size=(23, 2))
    counts = rng.uniform(0.1, 2.0, (23, 2))
    params, _ = EMEstimator()._get_item_params_and_bounds(model, 0)
    original = model.parameters
    objective = prepare_polytomous_objective(model, 0, theta, counts, 1e-10)
    loss, gradient = objective(params)
    assert params.size == gradient.size == 3
    assert loss == pytest.approx(-np.sum(counts * np.log(model.probability(theta, 0))))
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(
    "override",
    [
        "subclass",
        "probability",
        "_category_probabilities",
        "_ensure_theta_2d",
        "set_item_parameter",
    ],
)
def test_custom_category_behavior_retains_numerical_objective(monkeypatch, override):
    class CustomModel(GradedResponseModel):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx)[:, ::-1]

    model = (
        CustomModel(1, n_categories=3)
        if override == "subclass"
        else GradedResponseModel(1, n_categories=3)
    )
    calls = []
    if override != "subclass":
        original = getattr(model, override)

        def custom(*args, **kwargs):
            calls.append(override)
            result = original(*args, **kwargs)
            if override in ("probability", "_category_probabilities"):
                return result[:, ::-1]
            if override == "_ensure_theta_2d":
                return result * 1.2
            return result

        setattr(model, override, custom)
    estimator = EMEstimator(n_quadpts=5, use_rust=False)
    estimator._quadrature = GaussHermiteQuadrature(5)
    responses = np.array([[0], [1], [2], [-1]])
    posterior = np.full((4, 5), 0.2)
    theta = estimator._quadrature.nodes
    counts = np.column_stack(
        [posterior[responses[:, 0] == c].sum(axis=0) for c in range(3)]
    )
    assert prepare_polytomous_objective(model, 0, theta, counts, 1e-10) is None

    def optimize(objective, x0, *, jac, **kwargs):
        assert not jac
        candidate = x0 + 0.05
        value = objective(candidate)
        probability = np.clip(model.probability(theta, 0), 1e-10, 1 - 1e-10)
        assert value == pytest.approx(-np.sum(counts * np.log(probability)))
        return SimpleNamespace(x=x0)

    original_params = model.parameters
    monkeypatch.setattr(em_module, "minimize", optimize)
    estimator._m_step(model, responses, posterior)
    assert calls or override == "subclass"
    for name, values in original_params.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_item_counts_respect_explicit_observation_mask(monkeypatch):
    model = GradedResponseModel(1, n_categories=3)
    estimator = EMEstimator(n_quadpts=5, use_rust=False)
    theta = GaussHermiteQuadrature(5).nodes
    responses = np.array([[0], [1], [2], [1]])
    posterior = np.full((4, 5), 0.2)
    observed = np.array([True, False, True, False])
    counts = np.column_stack(
        [posterior[observed & (responses[:, 0] == c)].sum(axis=0) for c in range(3)]
    )

    def optimize(objective, x0, *, jac, **kwargs):
        assert jac
        value, _ = objective(x0)
        assert value == pytest.approx(_reference(model, 0, theta, counts, 1e-10, x0))
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(em_module, "minimize", optimize)
    estimator._optimize_item_params(
        model,
        0,
        responses,
        posterior,
        theta,
        posterior.sum(axis=0),
        valid_mask=observed,
    )


@pytest.mark.parametrize(
    "factory", [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel]
)
def test_category_mstep_avoids_full_posterior_row_copies(monkeypatch, factory):
    model = factory(2, n_categories=4, n_factors=2)
    estimator = EMEstimator(n_quadpts=7, use_rust=False)
    estimator._quadrature = GaussHermiteQuadrature(7, 2)
    rng = np.random.default_rng(90)
    responses = rng.integers(-1, 4, (4000, 2))
    posterior = np.full((4000, 49), 1 / 49)
    posterior.setflags(write=False)
    monkeypatch.setattr(
        em_module, "minimize", lambda objective, x0, **kwargs: SimpleNamespace(x=x0)
    )
    tracemalloc.start()
    try:
        estimator._m_step(model, responses, posterior)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < posterior.nbytes / 2
