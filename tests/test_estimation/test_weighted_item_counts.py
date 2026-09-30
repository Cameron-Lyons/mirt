"""Weighted item objectives use bounded counts and preserve model state."""

import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

import mirt
import mirt.estimation._em_context as context_module
import mirt.estimation.em as em_module
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.weighted import WeightedEMEstimator


@pytest.mark.parametrize("limit", [12, 1_000_000])
@pytest.mark.parametrize("weighted", [False, True])
def test_count_blocks_match_direct_weighted_aggregation(monkeypatch, limit, weighted):
    monkeypatch.setattr(context_module, "_MAX_COUNT_ENTRIES", limit)
    rng = np.random.default_rng(42)
    responses = rng.integers(-2, 2, (23, 3))
    posterior = rng.uniform(0.1, 1.0, (23, 14))[:, ::2]
    person_weights = rng.uniform(0.1, 2.0, 23) if weighted else None
    if weighted:
        person_weights[::4] = 0.0
    posterior.setflags(write=False)
    context = EMFitContext(responses)
    unweighted = context.expected_counts(posterior)
    correct, observed = context.expected_counts(posterior, person_weights)
    reference = (
        posterior if person_weights is None else posterior * person_weights[:, None]
    )
    np.testing.assert_allclose(
        correct, np.where(responses >= 0, responses, 0).T @ reference
    )
    np.testing.assert_allclose(observed, (responses >= 0).T @ reference)
    for before, after in zip(
        unweighted, context.expected_counts(posterior), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    categories = context.expected_category_counts(1, 4, posterior, person_weights)
    expected = np.column_stack(
        [reference[responses[:, 1] == category].sum(axis=0) for category in range(4)]
    )
    np.testing.assert_allclose(categories, expected, atol=1e-14)


_FACTORIES = [
    lambda: mirt.OneParameterLogistic(3),
    lambda: mirt.TwoParameterLogistic(3),
    lambda: mirt.ThreeParameterLogistic(3),
    lambda: mirt.FourParameterLogistic(3),
    lambda: mirt.TwoParameterLogistic(3, n_factors=2),
    lambda: mirt.MultidimensionalModel(3, n_factors=2),
    lambda: mirt.GradedResponseModel(3, n_categories=[2, 3, 5]),
    lambda: mirt.GeneralizedPartialCredit(3, n_categories=[2, 3, 5]),
    lambda: mirt.PartialCreditModel(3, n_categories=[2, 3, 5]),
    lambda: mirt.NominalResponseModel(3, n_categories=[2, 3, 5]),
]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_weighted_objective_matches_personwise_likelihood(monkeypatch, factory):
    model = factory()
    if isinstance(model, mirt.FourParameterLogistic):
        model.set_parameters(upper=np.full(3, 0.9))
    rng = np.random.default_rng(713)
    categories = model.n_categories if model.is_polytomous else [2] * 3
    responses = np.column_stack([rng.integers(-1, k, 31) for k in categories])
    estimator = WeightedEMEstimator(n_quadpts=5)
    estimator._quadrature = GaussHermiteQuadrature(5, model.n_factors)
    points = estimator._quadrature.nodes
    posterior = rng.uniform(0.1, 1.0, (31, len(points)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    weights = rng.uniform(0.1, 2.0, 31)
    weights[::3] = 0.0
    calls = []
    original = model.parameters

    def minimize(objective, x0, *, jac=False, bounds, **kwargs):
        item = len(calls)
        calls.append(item)
        trial = np.array(
            [
                np.clip(value + 0.02, low + 0.01, high - 0.01)
                for value, (low, high) in zip(x0, bounds, strict=True)
            ]
        )

        def reference(params):
            estimator._set_item_params(model, item, params)
            try:
                probabilities = np.clip(
                    model.probability(points, item),
                    estimator.prob_epsilon,
                    1 - estimator.prob_epsilon,
                )
                valid = responses[:, item] >= 0
                if model.is_polytomous:
                    likelihoods = np.log(probabilities[:, responses[valid, item]].T)
                else:
                    y = responses[valid, item, None]
                    likelihoods = y * np.log(probabilities) + (1 - y) * np.log1p(
                        -probabilities
                    )
                return -np.sum(weights[valid, None] * posterior[valid] * likelihoods)
            finally:
                estimator._set_item_params(model, item, x0)

        evaluation = objective(trial)
        value, gradient = evaluation if jac else (evaluation, None)
        assert value == pytest.approx(reference(trial), rel=1e-12)
        if gradient is not None:
            numerical = np.zeros_like(trial)
            for coordinate in range(trial.size):
                delta = np.zeros_like(trial)
                delta[coordinate] = 1e-5
                numerical[coordinate] = (
                    reference(trial + delta) - reference(trial - delta)
                ) / 2e-5
            np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=1e-7)
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(em_module, "minimize", minimize)
    estimator._m_step_weighted(model, responses, posterior, weights)
    assert calls == [0, 1, 2]
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("poly", [False, True])
def test_zero_weight_and_fixed_items_do_not_optimize(monkeypatch, poly):
    parent = mirt.PartialCreditModel if poly else mirt.OneParameterLogistic

    class FixedItem(parent):
        @property
        def free_parameter_masks(self):
            masks = super().free_parameter_masks
            for mask in masks.values():
                mask[0] = False
            return masks

    model = FixedItem(4, n_categories=3) if poly else FixedItem(4)
    # Fixed parameters, zero survey mass, and missing observations leave their
    # respective items unchanged, even when other items have information.
    responses = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
    weights = np.array([1.0, 0.0])
    estimator = WeightedEMEstimator(n_quadpts=5)
    estimator._quadrature = GaussHermiteQuadrature(5)
    calls = []

    def minimize(objective, x0, **kwargs):
        calls.append(x0)
        return SimpleNamespace(x=x0)

    original = model.parameters
    monkeypatch.setattr(em_module, "minimize", minimize)
    estimator._m_step_weighted(model, responses, np.full((2, 5), 0.2), weights)
    assert len(calls) == 1
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize("poly", [False, True])
@pytest.mark.parametrize("operation", ["mstep", "curvature"])
def test_failed_custom_probability_restores_parameters(monkeypatch, poly, operation):
    model = (
        mirt.GradedResponseModel(2, n_categories=3)
        if poly
        else mirt.TwoParameterLogistic(2)
    )
    estimator = WeightedEMEstimator(n_quadpts=5)
    estimator._quadrature = GaussHermiteQuadrature(5)
    responses = np.array([[0, 1], [1, 0]])
    posterior = np.full((2, 5), 0.2)
    original = model.parameters
    probability = model.probability

    def fail(*args):
        if any(
            not np.array_equal(model.parameters[name], values)
            for name, values in original.items()
        ):
            raise RuntimeError("custom probability failed")
        return probability(*args)

    model.probability = fail
    if operation == "mstep":

        def minimize(objective, x0, **kwargs):
            objective(x0 + 0.05)

        monkeypatch.setattr(em_module, "minimize", minimize)
        run = estimator._m_step_weighted
    else:
        run = estimator._compute_weighted_standard_errors
    with pytest.raises(RuntimeError, match="custom probability failed"):
        run(model, responses, posterior, np.array([0.5, 2.0]))
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_weighted_fit_releases_response_context_on_failure(monkeypatch):
    estimator = WeightedEMEstimator(n_quadpts=5, max_iter=2)

    def fail(*args):
        assert estimator._fit_context is not None
        raise RuntimeError("failed E-step")

    monkeypatch.setattr(estimator, "_e_step_weighted", fail)
    with pytest.raises(RuntimeError, match="failed E-step"):
        estimator.fit(mirt.TwoParameterLogistic(2), np.array([[0, 1], [1, 0]]))
    assert estimator._fit_context is None


def test_weighted_statistics_scratch_does_not_scale_with_posterior(monkeypatch):
    estimator = WeightedEMEstimator(n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(7, 3)
    model = mirt.TwoParameterLogistic(2, n_factors=3)
    rng = np.random.default_rng(14)
    responses = rng.integers(-1, 2, (2000, 2))
    posterior = np.full((2000, 343), 1 / 343)
    posterior.setflags(write=False)
    weights = rng.uniform(0.1, 2.0, 2000)
    monkeypatch.setattr(
        em_module, "minimize", lambda objective, x0, **kwargs: SimpleNamespace(x=x0)
    )
    tracemalloc.start()
    try:
        estimator._m_step_weighted(model, responses, posterior, weights)
        estimator._compute_weighted_standard_errors(
            model, responses, posterior, weights
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < posterior.nbytes / 4
