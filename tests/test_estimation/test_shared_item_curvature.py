"""Numerical uncertainty preserves custom curves and reuses prepared data."""

import tracemalloc
from copy import deepcopy
from types import MethodType

import numpy as np
import pytest

import mirt.estimation._em_context as context_module
from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.models.base import BaseItemModel
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


def _problem(model):
    rng = np.random.default_rng(917)
    categories = model.n_categories if model.is_polytomous else [2] * model.n_items
    responses = np.column_stack([rng.integers(-1, k, 29) for k in categories])
    quadrature = GaussHermiteQuadrature(5, model.n_factors)
    posterior = rng.uniform(0.1, 1.0, (29, len(quadrature.nodes)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior[::7] *= 3.0
    posterior.setflags(write=False)
    return responses, quadrature, posterior


def _personwise_reference(
    model, responses, quadrature, posterior, method, h, epsilon=PROB_EPSILON
):
    result = {name: np.zeros_like(values) for name, values in model.parameters.items()}
    for name, values in model.parameters.items():
        for index in np.ndindex(values.shape):
            if not model.free_parameter_masks[name][index]:
                continue
            item, coordinate = index[0], index[1:]
            valid = responses[:, item] >= 0

            def loss(offset):
                local = deepcopy(model)
                current = (
                    values[item].copy() if values.ndim > 1 else float(values[item])
                )
                if values.ndim > 1:
                    current[coordinate] += offset
                else:
                    current += offset
                local.set_item_parameter(item, name, current)
                p = np.clip(
                    local.probability(quadrature.nodes, item), epsilon, 1 - epsilon
                )
                if model.is_polytomous:
                    terms = np.log(p[:, responses[valid, item]].T)
                else:
                    y = responses[valid, item, None]
                    terms = y * np.log(p) + (1 - y) * np.log1p(-p)
                return float(np.sum(posterior[valid] * terms))

            center = loss(0.0)

            def se(step):
                numerator = (
                    loss(step) - 2 * center + loss(-step)
                    if method != "forward"
                    else loss(2 * step) - 2 * loss(step) + center
                )
                curvature = numerator / step**2
                return np.sqrt(-1 / curvature) if curvature < 0 else np.nan

            result[name][index] = (
                (4 * se(h / 2) - se(h)) / 3 if method == "richardson" else se(h)
            )
    return result


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TwoParameterLogistic(2),
        lambda: TwoParameterLogistic(2, n_factors=2),
        lambda: ThreeParameterLogistic(2),
        lambda: GradedResponseModel(2, n_categories=[2, 4], n_factors=2),
        lambda: GeneralizedPartialCredit(2, n_categories=[2, 4], n_factors=2),
        lambda: PartialCreditModel(2, n_categories=[2, 4]),
        lambda: NominalResponseModel(2, n_categories=[2, 4], n_factors=2),
    ],
)
@pytest.mark.parametrize("method", ["central", "forward", "richardson"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_numerical_methods_match_personwise_likelihood(factory, method, jobs):
    model = factory()
    responses, quadrature, posterior = _problem(model)
    original = model.parameters
    expected = _personwise_reference(
        model, responses, quadrature, posterior, method, 5e-3
    )
    actual = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method=method,
        step_size=5e-3,
        n_jobs=jobs,
    )
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=2e-6, atol=1e-7, equal_nan=True
        )
        np.testing.assert_array_equal(model.parameters[name], original[name])


@pytest.mark.parametrize("kind", ["bound_method", "constructor_state"])
@pytest.mark.parametrize("method", ["central", "forward", "richardson"])
def test_parallel_curvature_preserves_custom_model_behavior(kind, method):
    class CalibratedModel(TwoParameterLogistic):
        def __init__(self, n_items, *, calibration_scale):
            self.calibration_scale = calibration_scale
            super().__init__(n_items)

        def probability(self, theta, item_idx=None):
            return super().probability(theta * self.calibration_scale + 0.8, item_idx)

    if kind == "constructor_state":
        model = CalibratedModel(3, calibration_scale=1.4)
    else:
        model = TwoParameterLogistic(3)
        model.calibration_scale = 1.4

        def curve(self, theta, item_idx=None):
            return TwoParameterLogistic.probability(
                self, theta * self.calibration_scale + 0.8, item_idx
            )

        model.probability = MethodType(curve, model)
    responses, quadrature, posterior = _problem(model)
    original = model.parameters
    serial = compute_se(
        model, responses, quadrature, posterior, method=method, step_size=1e-4
    )
    parallel = compute_se(
        model, responses, quadrature, posterior, method=method, step_size=1e-4, n_jobs=2
    )
    for name in serial:
        np.testing.assert_array_equal(parallel[name], serial[name])
        np.testing.assert_array_equal(model.parameters[name], original[name])
    assert model.calibration_scale == 1.4


@pytest.mark.parametrize("poly", [False, True])
def test_core_uncertainty_respects_private_curve_overrides(poly):
    model = (
        GradedResponseModel(2, n_categories=[2, 4]) if poly else TwoParameterLogistic(2)
    )

    def transform(self, theta):
        return BaseItemModel._ensure_theta_2d(self, theta) * 1.2 + 0.7

    model._ensure_theta_2d = MethodType(transform, model)
    responses, quadrature, posterior = _problem(model)
    estimator = EMEstimator(n_quadpts=5)
    estimator._quadrature = quadrature
    actual = estimator._compute_standard_errors(model, responses, posterior)
    expected = compute_se(model, responses, quadrature, posterior)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_core_curvature_retains_configured_clipping():
    model = TwoParameterLogistic(2)
    responses, quadrature, posterior = _problem(model)
    estimator = EMEstimator(
        n_quadpts=5, prob_epsilon=0.1, se_step_size=5e-3, se_method="complete_data"
    )
    estimator._quadrature = quadrature
    actual = estimator._compute_standard_errors(model, responses, posterior)
    expected = _personwise_reference(
        model, responses, quadrature, posterior, "central", 5e-3, epsilon=0.1
    )
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("poly", [False, True])
@pytest.mark.parametrize("method", ["central", "richardson"])
def test_counts_and_worker_pool_are_shared_across_fields(monkeypatch, poly, method):
    model = (
        NominalResponseModel(3, n_categories=[2, 3, 4], n_factors=2)
        if poly
        else TwoParameterLogistic(3)
    )
    responses, quadrature, posterior = _problem(model)
    count_calls = []
    pool_exits = []
    original_counts = EMFitContext.expected_counts
    original_categories = EMFitContext.expected_category_counts
    original_pool = context_module.ThreadPoolExecutor

    def counts(self, *args, **kwargs):
        count_calls.append("binary")
        return original_counts(self, *args, **kwargs)

    def categories(self, item, *args, **kwargs):
        count_calls.append(item)
        return original_categories(self, item, *args, **kwargs)

    class Pool(original_pool):
        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                pool_exits.append(self._shutdown)

    monkeypatch.setattr(EMFitContext, "expected_counts", counts)
    monkeypatch.setattr(EMFitContext, "expected_category_counts", categories)
    monkeypatch.setattr(context_module, "ThreadPoolExecutor", Pool)
    compute_se(model, responses, quadrature, posterior, method=method, n_jobs=2)
    assert count_calls == ([0, 1, 2] if poly else ["binary"])
    assert pool_exits == [True]


def test_fixed_reference_and_padding_are_never_perturbed():
    model = NominalResponseModel(2, n_categories=[2, 4], n_factors=2)
    original_setter = model.set_item_parameter
    original = model.parameters

    def setter(item, name, value):
        free = model.free_parameter_masks[name][item]
        np.testing.assert_array_equal(
            np.asarray(value)[~free], original[name][item][~free]
        )
        return original_setter(item, name, value)

    model.set_item_parameter = setter
    responses, quadrature, posterior = _problem(model)
    compute_se(model, responses, quadrature, posterior, method="richardson")


def test_parallel_failure_closes_pool_and_leaves_original_parameters(monkeypatch):
    model = TwoParameterLogistic(3)
    model.initial = model.parameters

    def probability(self, theta, item_idx=None):
        if any(
            not np.array_equal(self.parameters[name], value)
            for name, value in self.initial.items()
        ):
            raise RuntimeError("failed worker trial")
        return TwoParameterLogistic.probability(self, theta, item_idx)

    model.probability = MethodType(probability, model)
    responses, quadrature, posterior = _problem(model)
    exits = []
    original_pool = context_module.ThreadPoolExecutor

    class Pool(original_pool):
        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                exits.append(self._shutdown)

    monkeypatch.setattr(context_module, "ThreadPoolExecutor", Pool)
    with pytest.raises(RuntimeError, match="failed worker trial"):
        compute_se(model, responses, quadrature, posterior, n_jobs=2)
    assert exits == [True]
    for name, values in model.initial.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("jobs", [0, -2, True, 1.5])
def test_numerical_uncertainty_validates_worker_count(jobs):
    model = TwoParameterLogistic(2)
    responses, quadrature, posterior = _problem(model)
    with pytest.raises(ValueError, match="n_jobs"):
        compute_se(model, responses, quadrature, posterior, n_jobs=jobs)


@pytest.mark.parametrize("jobs", [1, 2])
def test_uncertainty_scratch_does_not_scale_with_person_quadrature_size(jobs):
    model = TwoParameterLogistic(2, n_factors=3)
    quadrature = GaussHermiteQuadrature(7, 3)
    rng = np.random.default_rng(917)
    responses = rng.integers(-1, 2, (2000, 2))
    posterior = np.full((2000, 343), 1 / 343)
    posterior.setflags(write=False)
    tracemalloc.start()
    try:
        compute_se(
            model, responses, quadrature, posterior, method="richardson", n_jobs=jobs
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < posterior.nbytes / 4
