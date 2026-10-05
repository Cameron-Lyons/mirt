"""Joint marginal gradients agree with direct person-level likelihoods."""

import tracemalloc
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt.estimation._bl_objective as objective_module
import mirt.estimation.bl as bl_module
from mirt.constants import PROB_EPSILON
from mirt.estimation._bl_objective import prepare_bl_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.bl import BLEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.base import BaseItemModel
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

_KINDS = (
    "1pl",
    "2pl",
    "2pl_3d",
    "3pl",
    "4pl",
    "grm",
    "grm_2d",
    "gpcm",
    "gpcm_2d",
    "pcm",
    "nrm",
    "nrm_2d",
    "mirt",
    "bifactor",
)


def _model(kind):
    if kind == "mirt":
        return MultidimensionalModel(
            3,
            2,
            model_type="confirmatory",
            loading_pattern=np.array([[1, 0], [0, 1], [1, 1]]),
        )
    if kind == "bifactor":
        return BifactorModel(3, [3, 7, 3])
    cls = {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "4pl": FourParameterLogistic,
        "grm": GradedResponseModel,
        "gpcm": GeneralizedPartialCredit,
        "pcm": PartialCreditModel,
        "nrm": NominalResponseModel,
    }[kind.split("_")[0]]
    kwargs = {}
    if kind.endswith("3d"):
        kwargs["n_factors"] = 3
    elif kind.endswith("2d"):
        kwargs["n_factors"] = 2
    if kind.split("_")[0] in ("grm", "gpcm", "pcm", "nrm"):
        kwargs["n_categories"] = [2, 4, 3]
    return cls(3, **kwargs)


def _problem(model):
    rng = np.random.default_rng(967)
    categories = model.n_categories if model.is_polytomous else [2] * model.n_items
    responses = np.column_stack([rng.integers(-1, k, 37) for k in categories])
    responses[0] = -1
    quadrature = GaussHermiteQuadrature(5, model.n_factors)
    quadrature.nodes.setflags(write=False)
    quadrature.weights.setflags(write=False)
    estimator = BLEstimator(n_quadpts=5, max_iter=5)
    estimator._quadrature = quadrature
    params, bounds, structure = estimator._flatten_parameters(model)
    params += rng.uniform(-0.02, 0.02, params.size)
    estimator._unflatten_parameters(model, params, structure)
    return estimator, responses, quadrature, params, bounds, structure


def _direct_loss(model, responses, quadrature, params, structure):
    local = deepcopy(model)
    BLEstimator()._unflatten_parameters(local, params, structure)
    likelihood = np.zeros((len(responses), len(quadrature.nodes)))
    for item in range(model.n_items):
        probabilities = np.clip(
            local.probability(quadrature.nodes, item),
            PROB_EPSILON,
            1 - PROB_EPSILON,
        )
        values = responses[:, item]
        valid = values >= 0
        if model.is_polytomous:
            terms = np.log(probabilities[:, values[valid]].T)
        else:
            y = values[valid, None]
            terms = y * np.log(probabilities) + (1 - y) * np.log1p(-probabilities)
        likelihood[valid] += terms
    return -float(logsumexp(likelihood + np.log(quadrature.weights), axis=1).sum())


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("missing", ["partial", "item", "all"])
def test_joint_gradient_matches_independent_marginal_likelihood(kind, missing):
    model = _model(kind)
    estimator, responses, quadrature, params, bounds, structure = _problem(model)
    if missing == "item":
        responses[:, 1] = -1
    elif missing == "all":
        responses[:] = -1
    responses.setflags(write=False)
    params.setflags(write=False)
    original = model.parameters
    with EMFitContext(responses) as context:
        objective = prepare_bl_objective(
            model,
            context,
            quadrature.nodes,
            np.log(quadrature.weights),
            structure,
            bounds,
            estimator._unflatten_parameters,
        )
        assert objective is not None
        loss, gradient = objective(params)
        expected = _direct_loss(model, responses, quadrature, params, structure)
        np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(objective.value(params), expected, atol=1e-12)
        for index in range(params.size):
            offset = np.zeros_like(params)
            offset[index] = 1e-5
            numeric = (
                _direct_loss(model, responses, quadrature, params + offset, structure)
                - _direct_loss(model, responses, quadrature, params - offset, structure)
            ) / 2e-5
            np.testing.assert_allclose(gradient[index], numeric, rtol=2e-6, atol=1e-7)
    for name in original:
        np.testing.assert_array_equal(model.parameters[name], original[name])


@pytest.mark.parametrize("poly", [False, True])
def test_clipped_tails_and_crossed_thresholds_retain_the_marginal_target(poly):
    model = GradedResponseModel(3, [2, 4, 3]) if poly else FourParameterLogistic(3)
    estimator, responses, quadrature, params, bounds, structure = _problem(model)
    if poly:
        params[structure["thresholds"]["start_idx"] :] *= -1
    else:
        params[
            structure["guessing"]["start_idx"] : structure["guessing"]["end_idx"]
        ] = 0
        params[structure["upper"]["start_idx"] :] = 1
    quadrature.nodes.setflags(write=True)
    quadrature.nodes[:] *= 20
    quadrature.nodes.setflags(write=False)
    with EMFitContext(responses) as context:
        objective = prepare_bl_objective(
            model,
            context,
            quadrature.nodes,
            np.log(quadrature.weights),
            structure,
            bounds,
            estimator._unflatten_parameters,
        )
        assert objective is not None
        loss, gradient = objective(params)
        assert np.all(np.isfinite(gradient))
        np.testing.assert_allclose(
            loss,
            _direct_loss(model, responses, quadrature, params, structure),
            rtol=1e-12,
        )
        for index in range(params.size):
            offset = np.zeros_like(params)
            offset[index] = 1e-6
            numeric = (
                _direct_loss(model, responses, quadrature, params + offset, structure)
                - _direct_loss(model, responses, quadrature, params - offset, structure)
            ) / 2e-6
            np.testing.assert_allclose(gradient[index], numeric, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize(
    "change", ["subclass", "likelihood", "private_curve", "layout", "pattern"]
)
def test_custom_models_keep_numerical_objectives(change, monkeypatch):
    class PowerModel(TwoParameterLogistic):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx) ** 2

    model = PowerModel(3) if change == "subclass" else TwoParameterLogistic(3)
    if change == "likelihood":
        original = model.log_likelihood_batch
        model.log_likelihood_batch = lambda data, theta: original(data, theta) * 1.3
    elif change == "private_curve":

        def transform(self, theta):
            return BaseItemModel._ensure_theta_2d(self, theta) * 1.2 + 0.8

        model._ensure_theta_2d = MethodType(transform, model)
    elif change == "layout":
        model._parameters = dict(reversed(list(model._parameters.items())))
    elif change == "pattern":
        model = MultidimensionalModel(
            3, 2, model_type="confirmatory", loading_pattern=np.full((3, 2), 0.5)
        )
    estimator, responses, quadrature, params, bounds, structure = _problem(model)
    with EMFitContext(responses) as context:
        assert (
            prepare_bl_objective(
                model,
                context,
                quadrature.nodes,
                np.log(quadrature.weights),
                structure,
                bounds,
                estimator._unflatten_parameters,
            )
            is None
        )

    def optimize(objective, x0, *, jac, **kwargs):
        assert not jac
        trial = x0 + 0.01
        actual = objective(trial)
        expected = -estimator._compute_marginal_log_likelihood(model, responses)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        return SimpleNamespace(x=trial, fun=actual, nit=1, success=True)

    monkeypatch.setattr(bl_module, "minimize", optimize)
    model._is_fitted = True
    estimator.fit(model, responses)


@pytest.mark.parametrize("method", ["Powell", None])
def test_custom_estimator_and_derivative_free_method_keep_their_objective(
    monkeypatch, method
):
    class ShiftedEstimator(BLEstimator):
        def _compute_marginal_log_likelihood(self, model, responses):
            return super()._compute_marginal_log_likelihood(model, responses) - 0.2

    calls = []

    def optimize(objective, x0, *, jac, **kwargs):
        calls.append(jac)
        return SimpleNamespace(x=x0, fun=objective(x0), nit=0, success=True)

    monkeypatch.setattr(bl_module, "minimize", optimize)
    model = TwoParameterLogistic(3)
    _, responses, _, _, _, _ = _problem(model)
    shifted = ShiftedEstimator(n_quadpts=5).fit(model, responses)
    ordinary = BLEstimator(n_quadpts=5, method=method).fit(model, responses)
    assert calls == [False, False]
    assert shifted.log_likelihood == pytest.approx(ordinary.log_likelihood - 0.2)


@pytest.mark.parametrize("kind", ["2pl", "grm", "pcm", "nrm_2d", "mirt", "bifactor"])
def test_joint_fit_matches_numerical_optimizer(kind, monkeypatch):
    evaluations = []
    original_minimize = bl_module.minimize

    def optimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        evaluations.append(result.nfev)
        return result

    monkeypatch.setattr(bl_module, "minimize", optimize)
    model = _model(kind)
    estimator, responses, _, _, _, _ = _problem(model)
    model._is_fitted = True
    analytic = estimator.fit(deepcopy(model), responses)
    monkeypatch.setattr(bl_module, "prepare_bl_objective", lambda *args: None)
    numeric = BLEstimator(n_quadpts=5, max_iter=5).fit(deepcopy(model), responses)
    assert evaluations[0] < evaluations[1] / 3
    assert analytic.n_parameters == numeric.n_parameters == model.n_parameters
    np.testing.assert_allclose(
        analytic.log_likelihood, numeric.log_likelihood, rtol=2e-7
    )
    for name in analytic.model.parameters:
        np.testing.assert_allclose(
            analytic.model.parameters[name],
            numeric.model.parameters[name],
            atol=3e-4,
            rtol=3e-4,
        )
        actual_se, expected_se = (
            analytic.standard_errors[name],
            numeric.standard_errors[name],
        )
        np.testing.assert_array_equal(np.isnan(actual_se), np.isnan(expected_se))
        actual_info = np.divide(
            1, actual_se**2, out=np.zeros_like(actual_se), where=actual_se > 0
        )
        expected_info = np.divide(
            1, expected_se**2, out=np.zeros_like(expected_se), where=expected_se > 0
        )
        # Compare curvature before its inverse magnifies cancellation in weak
        # directions. Built-in item models use the exact information on both
        # paths; others difference the gradient or the likelihood.
        np.testing.assert_allclose(actual_info, expected_info, rtol=1e-3, atol=1e-3)


def test_builtin_trials_do_not_mutate_original_or_repeat_response_preparation(
    monkeypatch,
):
    model = TwoParameterLogistic(3)
    _, responses, _, _, _, _ = _problem(model)
    original = model.parameters
    model._is_fitted = True
    preparations = []
    original_components = EMFitContext.response_components

    def components(self, *args):
        if self._components is None:
            preparations.append(True)
        return original_components(self, *args)

    monkeypatch.setattr(EMFitContext, "response_components", components)

    def optimize(objective, x0, *, jac, callback, **kwargs):
        assert jac
        trial = x0 + 0.02
        value, gradient = objective(trial)
        for name in original:
            np.testing.assert_array_equal(model.parameters[name], original[name])
        assert gradient.shape == trial.shape
        callback(trial)
        objective(x0)
        return SimpleNamespace(x=trial, fun=value, nit=1, success=True)

    monkeypatch.setattr(bl_module, "minimize", optimize)
    BLEstimator(n_quadpts=5, verbose=True).fit(model, responses)
    assert preparations == [True]


@pytest.mark.parametrize("stage", ["optimization", "standard_errors"])
def test_failed_numerical_trials_restore_parameters(monkeypatch, stage):
    model = TwoParameterLogistic(3)
    estimator, responses, _, params, _, structure = _problem(model)
    original = model.parameters
    model._is_fitted = True
    if stage == "optimization":
        monkeypatch.setattr(bl_module, "prepare_bl_objective", lambda *args: None)

        def optimize(objective, x0, **kwargs):
            objective(x0 + 0.02)
            raise RuntimeError("failed optimization")

        monkeypatch.setattr(bl_module, "minimize", optimize)
        with pytest.raises(RuntimeError, match="failed optimization"):
            estimator.fit(model, responses)
    else:

        def likelihood(self, theta, item_idx=None):
            if any(
                not np.array_equal(self._parameters[name], original[name])
                for name in original
            ):
                raise RuntimeError("failed curvature")
            return TwoParameterLogistic.probability(self, theta, item_idx)

        model.probability = MethodType(likelihood, model)
        with pytest.raises(RuntimeError, match="failed curvature"):
            estimator._compute_standard_errors(model, responses, params, structure)
    for name in original:
        np.testing.assert_array_equal(model.parameters[name], original[name])


def test_fixed_model_skips_optimization_and_counts_only_free_parameters(monkeypatch):
    class FixedModel(TwoParameterLogistic):
        @property
        def free_parameter_masks(self):
            return {
                name: np.zeros_like(value, dtype=bool)
                for name, value in self.parameters.items()
            }

    model = FixedModel(3)
    _, responses, _, _, _, _ = _problem(model)
    monkeypatch.setattr(
        bl_module,
        "minimize",
        lambda *args, **kwargs: pytest.fail("fixed model optimized"),
    )
    result = BLEstimator(n_quadpts=5).fit(model, responses)
    assert result.n_parameters == 0
    assert result.n_iterations == 0
    assert result.converged
    assert result.aic == pytest.approx(-2 * result.log_likelihood)
    for se in result.standard_errors.values():
        np.testing.assert_array_equal(se, 0)


def test_owned_posterior_and_bounded_likelihood_scratch():
    model = TwoParameterLogistic(8, n_factors=3)
    rng = np.random.default_rng(967)
    responses = rng.integers(-1, 2, (2000, 8))
    responses.setflags(write=False)
    estimator = BLEstimator(n_quadpts=7)
    quadrature = GaussHermiteQuadrature(7, 3)
    params, bounds, structure = estimator._flatten_parameters(model)
    with EMFitContext(responses) as context:
        objective = prepare_bl_objective(
            model,
            context,
            quadrature.nodes,
            np.log(quadrature.weights),
            structure,
            bounds,
            estimator._unflatten_parameters,
        )
        assert objective is not None
        tracemalloc.start()
        try:
            loss, gradient = objective(params)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert np.isfinite(loss)
        assert np.all(np.isfinite(gradient))
        assert peak < 2 * len(responses) * len(quadrature.nodes) * 8


@pytest.mark.parametrize("kind", ["2pl_3d", "grm_2d", "nrm_2d"])
def test_likelihood_row_blocks_preserve_values_and_gradients(kind, monkeypatch):
    model = _model(kind)
    estimator, responses, quadrature, params, bounds, structure = _problem(model)
    with EMFitContext(responses) as context:
        objective = prepare_bl_objective(
            model,
            context,
            quadrature.nodes,
            np.log(quadrature.weights),
            structure,
            bounds,
            estimator._unflatten_parameters,
        )
        assert objective is not None
        expected_loss, expected_gradient = objective(params)
        monkeypatch.setattr(objective_module, "_MAX_LIKELIHOOD_ENTRIES", 7)
        loss, gradient = objective(params)
        np.testing.assert_allclose(loss, expected_loss, atol=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)


@pytest.mark.parametrize("affine", [False, True])
def test_unbounded_trials_keep_stable_public_probability_kernels(affine):
    model = MultidimensionalModel(3, 2) if affine else TwoParameterLogistic(3)
    estimator, responses, quadrature, params, bounds, structure = _problem(model)
    if affine:
        params[:2] = 1e308
    else:
        params[0] = 1e300
        params[structure["difficulty"]["start_idx"]] = 6e300
    with EMFitContext(responses) as context:
        objective = prepare_bl_objective(
            model,
            context,
            quadrature.nodes,
            np.log(quadrature.weights),
            structure,
            bounds,
            estimator._unflatten_parameters,
        )
        assert objective is not None
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            loss, gradient = objective(params)
            reference = _direct_loss(model, responses, quadrature, params, structure)
        np.testing.assert_allclose(loss, reference, rtol=1e-12)
        assert np.all(np.isfinite(gradient))
