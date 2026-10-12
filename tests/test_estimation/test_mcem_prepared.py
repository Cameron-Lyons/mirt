"""Shared MC item gradients agree with public curves on person-specific draws."""

from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import mirt.estimation._mc_objective as objective_module
import mirt.estimation.mcem as mc_module
from mirt.constants import PROB_EPSILON
from mirt.estimation._mc_objective import prepare_mc_objective
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)

KINDS = ("1pl", "2pl", "3pl", "4pl", "grm", "gpcm", "pcm", "nrm", "mirt", "bifactor")


def _model(kind):
    if kind == "mirt":
        return MultidimensionalModel(
            3, 3, loading_pattern=np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        )
    if kind == "bifactor":
        return BifactorModel(3, [4, 19, 4])
    cls = {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "4pl": FourParameterLogistic,
        "grm": GradedResponseModel,
        "gpcm": GeneralizedPartialCredit,
        "pcm": PartialCreditModel,
        "nrm": NominalResponseModel,
    }[kind]
    options = {"n_factors": 1 if kind in ("1pl", "3pl", "4pl", "pcm") else 2}
    if kind in ("grm", "gpcm", "pcm", "nrm"):
        options["n_categories"] = [2, 4, 3]
    return cls(3, **options)


def _problem(kind, *, persons=11, samples=50):
    model = _model(kind)
    rng = np.random.default_rng(735)
    categories = model.n_categories if model.is_polytomous else [2] * model.n_items
    responses = np.column_stack(
        [rng.integers(0, count, persons) for count in categories]
    )
    responses[0] = -1
    responses[3, 1] = -1
    # Non-contiguous inputs exercise both row and sample slicing.
    theta = rng.normal(size=(persons, samples * 2, model.n_factors))[:, ::2]
    weights = rng.uniform(size=(persons, samples))
    weights /= weights.sum(axis=1, keepdims=True)
    weights[::4] *= 3
    weights[1, ::3] = 0
    for values in (responses, theta, weights):
        values.setflags(write=False)
    estimator = MCEMEstimator(n_samples=samples)
    return model, responses, theta, weights, estimator


def _reference(model, item, responses, theta, weights):
    loss = 0.0
    for person, decision in enumerate(responses[:, item]):
        if decision < 0:
            continue
        probabilities = model.probability(theta[person], item)
        if model.is_polytomous:
            log_probability = np.log(
                np.clip(probabilities[:, decision], PROB_EPSILON, 1.0)
            )
        else:
            p = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
            log_probability = np.log(p) if decision == 1 else np.log1p(-p)
        loss -= float(weights[person] @ log_probability)
    return loss


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("blocked", [False, True])
def test_prepared_loss_and_gradient_match_direct_person_sample_curves(
    kind, blocked, monkeypatch
):
    model, responses, theta, weights, estimator = _problem(kind)
    original = model.parameters
    params, bounds = estimator._get_item_params_and_bounds(model, 1)
    if blocked:
        monkeypatch.setattr(objective_module, "_MAX_MC_OBJECTIVE_ENTRIES", 97)
    objective = prepare_mc_objective(
        model, 1, responses[:, 1], theta, weights, 50, bounds
    )
    assert objective is not None
    trial = params + 0.07
    trial = np.clip(trial, np.array(bounds)[:, 0] + 0.02, np.array(bounds)[:, 1] - 0.02)
    local = deepcopy(model)

    def reference(candidate):
        estimator._set_item_params(local, 1, candidate)
        return _reference(local, 1, responses, theta, weights)

    value, gradient = objective(trial)
    np.testing.assert_allclose(value, reference(trial), atol=2e-12)
    for index in range(len(params)):
        delta = np.zeros_like(params)
        delta[index] = 1e-6
        expected = (reference(trial + delta) - reference(trial - delta)) / 2e-6
        np.testing.assert_allclose(gradient[index], expected, rtol=1e-5, atol=2e-8)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("kind", KINDS)
def test_builtin_optimizer_uses_gradient_and_installs_only_final_parameters(
    kind, monkeypatch
):
    model, responses, theta, weights, estimator = _problem(kind)
    original = model.parameters
    current, bounds = estimator._get_item_params_and_bounds(model, 1)
    trial = np.clip(current + 0.04, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    expected = deepcopy(model)
    estimator._set_item_params(expected, 1, trial)

    def optimize(objective, *, x0, jac, **kwargs):
        assert jac
        np.testing.assert_array_equal(x0, current)
        value, gradient = objective(trial)
        assert np.all(np.isfinite(gradient))
        np.testing.assert_allclose(
            value, _reference(expected, 1, responses, theta, weights), atol=1e-12
        )
        for name, values in original.items():
            np.testing.assert_array_equal(model.parameters[name], values)
        return SimpleNamespace(x=trial, fun=value)

    monkeypatch.setattr(mc_module, "minimize", optimize)
    estimator._optimize_item_mc(model, 1, responses, theta, weights)
    for name in original:
        np.testing.assert_array_equal(model.parameters[name], expected.parameters[name])


@pytest.mark.parametrize(
    "change", ["subclass", "probability", "theta", "parameter_order"]
)
def test_custom_models_retain_numerical_public_probability_objective(
    change, monkeypatch
):
    model, responses, theta, weights, estimator = _problem("2pl")
    if change == "subclass":

        class CustomModel(TwoParameterLogistic):
            def probability(self, points, item_idx=None):
                return super().probability(points, item_idx) ** 1.2

        model = CustomModel(3, n_factors=2)
    elif change == "probability":
        model.probability = lambda points, item_idx=None: np.full(len(points), 0.37)
    elif change == "theta":

        def transform(self, points):
            return TwoParameterLogistic._ensure_theta_2d(self, points) * 1.2 + 0.4

        model._ensure_theta_2d = MethodType(transform, model)
    else:
        model._parameters = dict(reversed(list(model._parameters.items())))
    current, _ = estimator._get_item_params_and_bounds(model, 1)
    expected = deepcopy(model)
    estimator._set_item_params(expected, 1, current + 0.05)

    def optimize(objective, *, x0, jac, **kwargs):
        assert not jac
        candidate = x0 + 0.05
        value = objective(candidate)
        np.testing.assert_allclose(
            value, _reference(expected, 1, responses, theta, weights), atol=1e-12
        )
        return SimpleNamespace(x=candidate, fun=value)

    monkeypatch.setattr(mc_module, "minimize", optimize)
    estimator._optimize_item_mc(model, 1, responses, theta, weights)


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
def test_custom_estimator_item_objective_is_preserved(override, monkeypatch):
    model, responses, theta, weights, estimator = _problem("2pl")
    calls = []

    def custom(self, model, item, responses, theta, weights):
        calls.append(True)
        return -42.0

    if override == "instance":
        estimator._item_expected_log_likelihood = MethodType(custom, estimator)
    elif override == "class":
        monkeypatch.setattr(MCEMEstimator, "_item_expected_log_likelihood", custom)
    else:

        class CustomEstimator(MCEMEstimator):
            _item_expected_log_likelihood = custom

        estimator = CustomEstimator(n_samples=50)

    def optimize(objective, *, x0, jac, **kwargs):
        assert not jac and objective(x0) == 42.0
        return SimpleNamespace(x=x0, fun=42.0)

    monkeypatch.setattr(mc_module, "minimize", optimize)
    estimator._optimize_item_mc(model, 1, responses, theta, weights)
    assert calls == [True]


@pytest.mark.parametrize(
    "bad",
    ["weight_shape", "theta_shape", "negative", "nan_weight", "nan_theta", "category"],
)
def test_preparation_validates_observed_inputs_before_optimization(bad):
    model, responses, theta, weights, estimator = _problem("grm")
    responses, theta, weights = responses.copy(), theta.copy(), weights.copy()
    if bad == "weight_shape":
        weights = weights[:, :-1]
    elif bad == "theta_shape":
        theta = theta[:, :-1]
    elif bad == "negative":
        weights[1, 0] = -1
    elif bad == "nan_weight":
        weights[1, 0] = np.nan
    elif bad == "nan_theta":
        theta[1, 0, 0] = np.nan
    else:
        responses[1, 1] = 4
    _, bounds = estimator._get_item_params_and_bounds(model, 1)
    with pytest.raises(ValueError):
        prepare_mc_objective(model, 1, responses[:, 1], theta, weights, 50, bounds)


def test_missing_person_samples_remain_ignored():
    model, responses, theta, weights, estimator = _problem("2pl")
    theta, weights = theta.copy(), weights.copy()
    theta[0] = np.nan
    weights[0] = np.nan
    params, bounds = estimator._get_item_params_and_bounds(model, 1)
    objective = prepare_mc_objective(
        model, 1, responses[:, 1], theta, weights, 50, bounds
    )
    value, gradient = objective(params)
    np.testing.assert_allclose(
        value, _reference(model, 1, responses, theta, weights), atol=1e-12
    )
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("kind", ["grm", "gpcm", "pcm", "nrm"])
def test_category_objective_keeps_monte_carlo_upper_clip_at_one(kind):
    model = _model(kind)
    estimator = MCEMEstimator(n_samples=50)
    params, bounds = estimator._get_item_params_and_bounds(model, 1)
    if kind == "nrm":
        params[:] = 0
        params[4:6] = 5
    theta = np.full((2, 50, model.n_factors), 1000.0)
    responses = np.full(2, 3)
    weights = np.full((2, 50), 0.02)
    objective = prepare_mc_objective(model, 1, responses, theta, weights, 50, bounds)
    value, gradient = objective(params)
    assert value == 0.0
    np.testing.assert_array_equal(gradient, np.zeros_like(params))


@pytest.mark.parametrize("bad", ["nan_params", "shape", "nan_loss"])
def test_failed_prepared_optimizer_results_restore_parameters(bad, monkeypatch):
    model, responses, theta, weights, estimator = _problem("2pl")
    original = model.parameters

    def optimize(objective, *, x0, **kwargs):
        value, _ = objective(x0 + 0.1)
        candidate = x0.copy()
        if bad == "nan_params":
            candidate[0] = np.nan
        elif bad == "shape":
            candidate = candidate[:-1]
        else:
            value = np.nan
        return SimpleNamespace(x=candidate, fun=value)

    monkeypatch.setattr(mc_module, "minimize", optimize)
    with pytest.raises(RuntimeError, match="invalid parameters"):
        estimator._optimize_item_mc(model, 1, responses, theta, weights)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("cls", [MCEMEstimator, QMCEMEstimator, StochasticEMEstimator])
@pytest.mark.parametrize("kind", ["2pl", "grm", "nrm"])
def test_complete_fit_matches_numerical_item_updates(cls, kind):
    model, responses, _, _, _ = _problem(kind, persons=17)
    options = {"n_chains": 3} if cls is StochasticEMEstimator else {"n_samples": 50}
    actual = cls(**options, max_iter=2, seed=819)
    numerical = cls(**options, max_iter=2, seed=819)
    numerical._item_expected_log_likelihood = MethodType(
        MCEMEstimator._item_expected_log_likelihood, numerical
    )
    result = actual.fit(model.copy(), responses)
    reference = numerical.fit(model.copy(), responses)
    np.testing.assert_allclose(
        result.log_likelihood, reference.log_likelihood, rtol=2e-6, atol=2e-5
    )
    for name, values in result.model.parameters.items():
        np.testing.assert_allclose(
            values, reference.model.parameters[name], rtol=2e-4, atol=2e-4
        )


@pytest.mark.parametrize("analytic", [False, True])
def test_graded_item_update_reaches_stationary_sampled_objective(analytic):
    model, responses, _, _, _ = _problem("grm", persons=17)
    estimator = MCEMEstimator(n_samples=50, seed=819)
    estimator._rng = np.random.default_rng(estimator.seed)
    theta, weights = estimator._e_step_mc(
        model,
        responses,
        np.zeros(model.n_factors),
        np.eye(model.n_factors),
        model.n_factors,
    )
    _, bounds = estimator._get_item_params_and_bounds(model, 0)
    objective = prepare_mc_objective(
        model, 0, responses[:, 0], theta, weights, estimator.n_samples, bounds
    )
    if not analytic:
        estimator._item_expected_log_likelihood = MethodType(
            MCEMEstimator._item_expected_log_likelihood, estimator
        )

    estimator._optimize_item_mc(model, 0, responses, theta, weights)

    fitted, _ = estimator._get_item_params_and_bounds(model, 0)
    _, gradient = objective(fitted)
    lower, upper = np.asarray(bounds).T
    # A bound's outward derivative does not prevent constrained stationarity.
    gradient[(fitted == lower) & (gradient > 0.0)] = 0.0
    gradient[(fitted == upper) & (gradient < 0.0)] = 0.0
    assert np.max(np.abs(gradient)) < 2e-5


def test_large_sample_objective_uses_bounded_scratch(monkeypatch):
    import tracemalloc

    model, responses, theta, weights, estimator = _problem(
        "mirt", persons=2000, samples=64
    )
    params, bounds = estimator._get_item_params_and_bounds(model, 1)
    expected = _reference(model, 1, responses, theta, weights)
    monkeypatch.setattr(objective_module, "_MAX_MC_OBJECTIVE_ENTRIES", 4096)
    tracemalloc.start()
    try:
        objective = prepare_mc_objective(
            model, 1, responses[:, 1], theta, weights, 64, bounds
        )
        value, gradient = objective(params)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(value, expected, atol=1e-10)
    assert np.all(np.isfinite(gradient))
    assert peak < 0.5 * theta.nbytes


@pytest.mark.parametrize("kind", ["2pl", "pcm", "nrm"])
def test_unobserved_items_skip_optimization(kind, monkeypatch):
    model, responses, theta, weights, estimator = _problem(kind)
    responses = responses.copy()
    responses[:, 1] = -1
    original = model.parameters

    def fail(*args, **kwargs):
        raise AssertionError("unobserved item should not be optimized")

    monkeypatch.setattr(mc_module, "minimize", fail)
    estimator._optimize_item_mc(model, 1, responses, theta, weights)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def _sparse_middle_category_responses():
    """Graded responses whose item 0 uses category 1 twice in 600 persons."""
    rng = np.random.default_rng(5)
    theta = rng.standard_normal(600)
    thresholds = np.tile([-1.0, 0.0, 1.0], (8, 1))
    thresholds[0] = [-1.0, -0.98, 1.0]
    cumulative = 1 / (1 + np.exp(-1.5 * (theta[:, None, None] - thresholds[None])))
    responses = (rng.random((600, 8))[:, :, None] < cumulative).sum(axis=2)
    assert np.bincount(responses[:, 0], minlength=4)[1] == 2
    return responses


def _assert_ordered_graded_item(model, item):
    from mirt.estimation._graded_order import THRESHOLD_GAP

    thresholds = model.parameters["thresholds"][item]
    assert np.all(np.diff(thresholds) >= THRESHOLD_GAP * (1 - 1e-9))
    probabilities = model.probability(np.linspace(-4, 4, 17)[:, None], item)
    assert np.all(probabilities >= 0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: QMCEMEstimator(n_samples=64, seed=1),
        lambda: MCEMEstimator(n_samples=50, seed=1),
        lambda: StochasticEMEstimator(n_chains=2, seed=1),
    ],
    ids=["qmcem", "mcem", "stochastic"],
)
def test_sparse_graded_category_keeps_thresholds_ordered(factory):
    # Shared-grid and person-specific M-steps both used to install [6, -6, 1.4]
    # after one iteration, giving category 1 probabilities near -1.
    responses = _sparse_middle_category_responses()
    model = GradedResponseModel(8, n_categories=4)
    estimator = factory()
    estimator._rng = np.random.default_rng(1)
    samples, weights = estimator._e_step_mc(model, responses, np.zeros(1), np.eye(1), 1)

    estimator._m_step_mc(model, responses, samples, weights)

    for item in range(model.n_items):
        _assert_ordered_graded_item(model, item)


def test_sparse_graded_category_fit_matches_em_thresholds():
    from mirt.estimation.em import EMEstimator

    responses = _sparse_middle_category_responses()
    reference = EMEstimator(compute_standard_errors=False).fit(
        GradedResponseModel(8, n_categories=4), responses
    )
    result = QMCEMEstimator(seed=1).fit(
        GradedResponseModel(8, n_categories=4), responses
    )

    assert result.converged
    _assert_ordered_graded_item(result.model, 0)
    np.testing.assert_allclose(
        result.model.parameters["thresholds"][0],
        reference.model.parameters["thresholds"][0],
        atol=0.05,
    )
