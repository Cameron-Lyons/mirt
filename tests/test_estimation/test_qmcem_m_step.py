"""Shared-grid QMCEM likelihoods, gradients, constraints, and failure recovery."""

from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.special import xlog1py, xlogy

from mirt.constants import PROB_EPSILON
from mirt.estimation import mcem as mcem_module
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator
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


def _model(kind):
    if kind == "bifactor":
        return BifactorModel(3, [4, 9, 4])
    if kind == "confirmatory":
        return MultidimensionalModel(
            3,
            2,
            model_type="confirmatory",
            loading_pattern=np.array([[1, 0], [0, 1], [1, 1]]),
        )
    if kind == "mirt":
        return MultidimensionalModel(3, 2)
    if kind == "2pl_multi":
        return TwoParameterLogistic(3, 2)
    if kind in ("grm", "gpcm", "pcm", "nrm"):
        cls = {
            "grm": GradedResponseModel,
            "gpcm": GeneralizedPartialCredit,
            "pcm": PartialCreditModel,
            "nrm": NominalResponseModel,
        }[kind]
        return cls(3, n_categories=[2, 4, 3])
    return {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "4pl": FourParameterLogistic,
    }[kind](3)


def _state(model):
    rng = np.random.default_rng(937)
    responses = np.column_stack(
        [
            rng.integers(0, model._n_categories[j] if model.is_polytomous else 2, 17)
            for j in range(model.n_items)
        ]
    )
    responses[::4, 0] = -1
    responses[:, 2] = -1
    grid = rng.normal(size=(50, model.n_factors))
    samples = np.broadcast_to(grid, (len(responses), *grid.shape))
    weights = rng.random((len(responses), len(grid)))
    weights /= weights.sum(axis=1, keepdims=True)
    responses.flags.writeable = weights.flags.writeable = False
    return responses, samples, weights


def _expanded_loss(model, item, responses, samples, weights, params):
    """Evaluate the public curve on every observed respondent/sample pair."""
    working = deepcopy(model)
    MCEMEstimator(n_samples=50)._set_item_params(working, item, params)
    valid = responses[:, item] >= 0
    theta = samples[valid].reshape(-1, model.n_factors)
    data = np.repeat(responses[valid, item], samples.shape[1])
    probabilities = working.probability(theta, item)
    if model.is_polytomous:
        selected = probabilities[np.arange(len(data)), data]
        log_probability = np.log(np.clip(selected, PROB_EPSILON, 1.0))
    else:
        probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
        log_probability = xlogy(data, probabilities) + xlog1py(1 - data, -probabilities)
    return -float(weights[valid].reshape(-1) @ log_probability)


@pytest.mark.parametrize(
    "kind",
    [
        "1pl",
        "2pl",
        "3pl",
        "4pl",
        "2pl_multi",
        "mirt",
        "confirmatory",
        "bifactor",
        "grm",
        "gpcm",
        "pcm",
        "nrm",
    ],
)
@pytest.mark.parametrize("blocked", [False, True])
def test_shared_objective_matches_expanded_public_likelihood(
    monkeypatch, kind, blocked
):
    model = _model(kind)
    responses, samples, weights = _state(model)
    original = model.parameters
    estimator = QMCEMEstimator(n_samples=50)
    if blocked:
        monkeypatch.setattr(mcem_module, "_MAX_QMC_COUNT_ELEMENTS", 13)
        from mirt.estimation import _em_context

        monkeypatch.setattr(_em_context, "_MAX_COUNT_ENTRIES", 13)
    calls = 0

    def compare(objective, x0, jac, **_kwargs):
        nonlocal calls
        item = calls
        calls += 1
        assert jac
        params = x0 + 0.03
        if kind == "4pl":
            params[-1] = 0.96
        actual = objective(params)
        expected = _expanded_loss(model, item, responses, samples, weights, params)
        if jac:
            loss, gradient = actual
            numerical = np.empty_like(params)
            for coordinate in range(len(params)):
                offset = np.zeros_like(params)
                offset[coordinate] = 1e-5
                high = _expanded_loss(
                    model, item, responses, samples, weights, params + offset
                )
                low = _expanded_loss(
                    model, item, responses, samples, weights, params - offset
                )
                numerical[coordinate] = (high - low) / 2e-5
            np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
        else:
            loss = actual
        np.testing.assert_allclose(loss, expected, rtol=1e-13)
        return SimpleNamespace(x=x0, fun=loss)

    monkeypatch.setattr(mcem_module, "minimize", compare)
    estimator._m_step_mc(model, responses, samples, weights)
    assert calls == 2
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize(
    "kind", ["1pl", "2pl", "3pl", "4pl", "mirt", "grm", "gpcm", "nrm"]
)
def test_shared_grid_optimization_matches_expanded_optimizer(kind):
    original = _model(kind)
    responses, samples, weights = _state(original)
    expanded, shared = deepcopy(original), deepcopy(original)
    MCEMEstimator(n_samples=50)._m_step_mc(expanded, responses, samples, weights)
    estimator = QMCEMEstimator(n_samples=50)
    estimator._m_step_mc(shared, responses, samples, weights)
    for item in (0, 1):
        expanded_params, _ = estimator._get_item_params_and_bounds(expanded, item)
        shared_params, _ = estimator._get_item_params_and_bounds(shared, item)
        expected = _expanded_loss(
            expanded, item, responses, samples, weights, expanded_params
        )
        actual = _expanded_loss(
            shared, item, responses, samples, weights, shared_params
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if kind == "1pl":
        np.testing.assert_array_equal(shared.discrimination, original.discrimination)


def test_custom_probability_uses_only_shared_grid_and_recovers_after_failure(
    monkeypatch,
):
    model = TwoParameterLogistic(3)
    responses, samples, weights = _state(model)
    original = model.parameters
    curve = model.probability
    sizes = []

    def custom(theta, item_idx=None):
        sizes.append(len(theta))
        return 0.15 + 0.7 * curve(theta, item_idx)

    monkeypatch.setattr(model, "probability", custom)
    estimator = QMCEMEstimator(n_samples=50)

    def fail(objective, x0, jac, **_kwargs):
        assert not jac
        trial = x0 + 0.2
        loss = objective(trial)
        # Compute an independent expected loss without copying the bound closure.
        valid = responses[:, 0] >= 0
        p = 0.15 + 0.7 * curve(samples[0], 0)
        log_probability = np.where(
            responses[valid, 0, None] == 1, np.log(p), np.log1p(-p)
        )
        np.testing.assert_allclose(loss, -np.sum(weights[valid] * log_probability))
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(mcem_module, "minimize", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        estimator._m_step_mc(model, responses, samples, weights)
    assert sizes == [50]
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(x=np.array([np.nan, 0.0]), fun=1.0),
        SimpleNamespace(x=np.array([1.0, 0.0]), fun=np.inf),
        SimpleNamespace(x=np.ones(3), fun=1.0),
    ],
)
def test_invalid_optimizer_result_restores_shared_grid_item(monkeypatch, result):
    model = TwoParameterLogistic(3)
    original = model.parameters
    responses, samples, weights = _state(model)
    monkeypatch.setattr(mcem_module, "minimize", lambda *_args, **_kwargs: result)
    with pytest.raises(RuntimeError, match="invalid parameters"):
        QMCEMEstimator(n_samples=50)._m_step_mc(model, responses, samples, weights)
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize("bad_weight", [-0.01, np.nan, np.inf])
def test_shared_grid_rejects_invalid_weights_before_mutating_model(bad_weight):
    model = TwoParameterLogistic(3)
    original = model.parameters
    responses, samples, weights = _state(model)
    weights = weights.copy()
    weights[-1, -1] = bad_weight
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        QMCEMEstimator(n_samples=50)._m_step_mc(model, responses, samples, weights)
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize("array", ["theta_samples", "weights"])
def test_shared_grid_rejects_incompatible_shapes(array):
    model = TwoParameterLogistic(3)
    responses, samples, weights = _state(model)
    if array == "theta_samples":
        samples = samples[:, :-1]
    else:
        weights = weights[:, :-1]
    with pytest.raises(ValueError, match=f"{array} has an incompatible shape"):
        QMCEMEstimator(n_samples=50)._m_step_mc(model, responses, samples, weights)


def test_independent_sample_grids_keep_monte_carlo_path(monkeypatch):
    model = TwoParameterLogistic(3)
    responses, samples, weights = _state(model)
    samples = samples.copy()
    calls = []
    monkeypatch.setattr(MCEMEstimator, "_m_step_mc", lambda *args: calls.append(args))
    estimator = QMCEMEstimator(n_samples=50)
    estimator._m_step_mc(model, responses, samples, weights)
    assert len(calls) == 1
    assert calls[0][0] is estimator
    assert calls[0][3] is samples


@pytest.mark.parametrize("kind", ["2pl", "grm"])
@pytest.mark.parametrize("malformed", ["shape", "nonfinite"])
def test_invalid_custom_probabilities_restore_item(monkeypatch, kind, malformed):
    model = _model(kind)
    original = model.parameters
    responses, samples, weights = _state(model)

    def invalid(theta, item_idx=None):
        shape = (
            (len(theta), model._n_categories[item_idx])
            if model.is_polytomous
            else (len(theta),)
        )
        probabilities = np.full(shape, 0.5)
        if malformed == "shape":
            return probabilities[:-1]
        probabilities[0] = np.nan
        return probabilities

    monkeypatch.setattr(model, "probability", invalid)
    with pytest.raises(ValueError, match="invalid item"):
        QMCEMEstimator(n_samples=50)._m_step_mc(model, responses, samples, weights)
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


def test_fully_fixed_items_skip_optimization(monkeypatch):
    class FixedModel(TwoParameterLogistic):
        @property
        def free_parameter_masks(self):
            return {
                name: np.zeros_like(mask)
                for name, mask in super().free_parameter_masks.items()
            }

    model = FixedModel(3)
    original = model.parameters
    responses, samples, weights = _state(model)

    def unexpected(*_args, **_kwargs):
        pytest.fail("fixed items must skip optimization")

    monkeypatch.setattr(mcem_module, "minimize", unexpected)
    from mirt.estimation._em_context import EMFitContext

    monkeypatch.setattr(EMFitContext, "expected_counts", unexpected)
    QMCEMEstimator(n_samples=50)._m_step_mc(model, responses, samples, weights)
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
@pytest.mark.parametrize(
    "method", ["_item_expected_log_likelihood", "_optimize_item_mc"]
)
def test_shared_grid_preserves_custom_estimator_callbacks(
    override, method, monkeypatch
):
    model = TwoParameterLogistic(3)
    responses, samples, weights = _state(model)
    original = model.parameters
    estimator = QMCEMEstimator(n_samples=50)
    calls = []

    def custom(self, model, item, *args):
        calls.append(item)
        return -42.0

    if override == "instance":
        setattr(estimator, method, MethodType(custom, estimator))
    elif override == "class":
        monkeypatch.setattr(MCEMEstimator, method, custom)
    else:
        cls = type("CustomQMC", (QMCEMEstimator,), {method: custom})
        estimator = cls(n_samples=50)

    def optimize(objective, *, x0, jac, **kwargs):
        assert not jac and objective(x0) == 42.0
        return SimpleNamespace(x=x0, fun=42.0)

    monkeypatch.setattr(mcem_module, "minimize", optimize)
    estimator._m_step_mc(model, responses, samples, weights)
    assert calls == ([0, 1, 2] if method == "_optimize_item_mc" else [0, 1])
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("kind", ["grm", "gpcm", "pcm", "nrm"])
def test_shared_category_objective_preserves_upper_clip_at_one(kind, monkeypatch):
    model = _model(kind)
    responses = np.full((2, 3), -1)
    responses[:, 1] = 3
    samples = np.broadcast_to(
        np.full((50, model.n_factors), 1000.0), (2, 50, model.n_factors)
    )
    weights = np.full((2, 50), 0.02)
    estimator = QMCEMEstimator(n_samples=50)

    def optimize(objective, *, x0, jac, **kwargs):
        assert jac
        trial = x0.copy()
        if kind == "nrm":
            trial[:] = 0
            trial[2] = 5
        value, gradient = objective(trial)
        assert value == 0.0
        np.testing.assert_array_equal(gradient, np.zeros_like(trial))
        return SimpleNamespace(x=x0, fun=value)

    monkeypatch.setattr(mcem_module, "minimize", optimize)
    estimator._m_step_mc(model, responses, samples, weights)
