"""Batched MCAT information matrices are exact and match per-item references."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.cat.mcat_selection as mcat_selection
from mirt.cat import (
    AOptimality,
    BayesianMCAT,
    COptimality,
    DOptimality,
    KullbackLeiblerMCAT,
    MCATEngine,
)
from mirt.cat.mcat_selection import (
    _compute_item_information_matrix,
    _item_information_matrices,
    create_mcat_selection_strategy,
)
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtModelError
from mirt.models import (
    GeneralizedPartialCredit,
    MultidimensionalModel,
    NoncompensatoryModel,
    TwoParameterLogistic,
)

THETA = np.array([0.3, -0.5, 0.8])
COVARIANCE = np.array([[0.7, -0.1, 0.2], [-0.1, 0.8, 0.1], [0.2, 0.1, 0.9]])


def _model(model_class, n_items=23, seed=8):
    rng = np.random.default_rng(seed)
    model = model_class(n_items=n_items, n_factors=3)
    slopes = rng.uniform(0.2, 2.2, size=(n_items, 3))
    if model_class is MultidimensionalModel:
        model.set_parameters(slopes=slopes, intercepts=rng.normal(size=n_items))
    elif model_class is TwoParameterLogistic:
        model.set_parameters(discrimination=slopes, difficulty=rng.normal(size=n_items))
    else:
        model.set_parameters(
            discrimination=slopes, difficulty=rng.normal(size=(n_items, 3))
        )
    model._is_fitted = True
    return model


def _reference_matrix(model, theta, item_idx):
    """Native matrix, or the compensatory p * q * a a^T fallback, per item."""
    if hasattr(model, "item_information_matrix"):
        return model.item_information_matrix(theta.reshape(1, -1), item_idx)[0]
    probability = float(
        np.clip(
            np.ravel(model.probability(theta.reshape(1, -1), item_idx=item_idx))[0],
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
    )
    slopes = np.asarray(model.get_item_parameters(item_idx)["discrimination"])
    return probability * (1.0 - probability) * np.outer(slopes, slopes)


def _fisher_matrix_by_differences(model, theta, item_idx, step=1e-5):
    """Return sum_k grad P_k grad P_k^T / P_k from central differences."""
    probabilities = np.asarray(model.probability(theta[None, :], item_idx)).reshape(-1)
    if probabilities.size == 1:
        probabilities = np.array([1.0 - probabilities[0], probabilities[0]])
    gradients = []
    for factor in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[factor] = step
        upper = np.asarray(model.probability((theta + shift)[None, :], item_idx))
        lower = np.asarray(model.probability((theta - shift)[None, :], item_idx))
        upper, lower = upper.reshape(-1), lower.reshape(-1)
        if upper.size == 1:
            upper = np.array([1.0 - upper[0], upper[0]])
            lower = np.array([1.0 - lower[0], lower[0]])
        gradients.append((upper - lower) / (2.0 * step))
    gradient = np.array(gradients)
    return (gradient / probabilities) @ gradient.T


@pytest.mark.parametrize(
    "model_class", [MultidimensionalModel, NoncompensatoryModel, TwoParameterLogistic]
)
def test_batched_matrices_match_per_item_reference(model_class):
    model = _model(model_class)
    items = [0, 3, 4, 9, 15, 22, 7]

    actual = _item_information_matrices(model, THETA, items)

    expected = np.stack([_reference_matrix(model, THETA, item) for item in items])
    assert actual.shape == (len(items), 3, 3)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("model_class", [MultidimensionalModel, TwoParameterLogistic])
def test_compensatory_matrices_equal_finite_difference_fisher(model_class):
    model = _model(model_class, n_items=6)
    for item in range(model.n_items):
        assert_allclose(
            _compute_item_information_matrix(model, THETA, item),
            _fisher_matrix_by_differences(model, THETA, item),
            rtol=1e-6,
            atol=1e-9,
        )


def test_multidimensional_partial_credit_uses_exact_fisher_matrices():
    model = GeneralizedPartialCredit(n_items=3, n_factors=2, n_categories=[3, 4, 2])
    model.set_parameters(
        discrimination=np.array([[0.8, 1.1], [1.2, -0.4], [1.5, 1.3]]),
        steps=np.array([[-0.5, 0.6, 0.0], [-1.0, 0.1, 0.9], [0.3, 0.0, 0.0]]),
    )
    model._is_fitted = True
    theta = np.array([0.3, -0.5])

    for item in range(model.n_items):
        assert_allclose(
            _compute_item_information_matrix(model, theta, item),
            _fisher_matrix_by_differences(model, theta, item),
            rtol=1e-6,
            atol=1e-9,
        )


class _PartialCreditWithoutMatrix(GeneralizedPartialCredit):
    """A polytomous model whose exact information matrix is unavailable."""

    item_information_matrix = None


def test_polytomous_models_without_information_matrix_are_rejected():
    model = _PartialCreditWithoutMatrix(n_items=4, n_factors=2, n_categories=3)
    model._is_fitted = True

    with pytest.raises(MirtModelError, match="does not define item_information_matrix"):
        _compute_item_information_matrix(model, np.zeros(2), 0)
    with pytest.raises(MirtModelError, match="_PartialCreditWithoutMatrix"):
        DOptimality().select_item(model, np.zeros(2), np.eye(2), {0, 1})
    with pytest.raises(MirtModelError):
        MCATEngine(model, max_items=2).run_simulation(np.zeros(2))


def test_multidimensional_model_evaluates_all_candidates_with_one_logit_pass(
    monkeypatch,
):
    model = _model(MultidimensionalModel, n_items=40)
    calls = []
    original = mcat_selection._sigmoid_derivative

    def capture(logits):
        calls.append(logits.shape)
        return original(logits)

    monkeypatch.setattr(mcat_selection, "_sigmoid_derivative", capture)
    DOptimality().get_item_criteria(model, THETA, COVARIANCE, set(range(40)))

    assert calls == [(40,)]


def test_customized_native_matrices_keep_the_per_item_path():
    model = _model(MultidimensionalModel, n_items=5)
    calls = []

    def scaled(theta, item_idx):
        calls.append(item_idx)
        return 2.0 * MultidimensionalModel.item_information_matrix(
            model, theta, item_idx
        )

    model.item_information_matrix = scaled

    actual = _item_information_matrices(model, THETA, [4, 1])

    assert calls == [4, 1]
    expected = [
        2.0 * MultidimensionalModel.item_information_matrix(model, THETA[None], item)[0]
        for item in (4, 1)
    ]
    assert_allclose(actual, expected)


def test_extreme_slopes_and_logits_reuse_native_log_space_recovery():
    model = MultidimensionalModel(n_items=4, n_factors=2)
    model.set_parameters(
        slopes=np.array([[1e-160, 1.0], [1e200, 0.5], [0.7, 0.4], [3.0, -2.0]]),
        intercepts=np.array([0.1, -0.2, 800.0, 0.3]),
    )
    model._is_fitted = True
    theta = np.array([0.4, -0.1])

    actual = _item_information_matrices(model, theta, range(4))

    for item in range(4):
        expected = model.item_information_matrix(theta[None], item)[0]
        assert_allclose(actual[item], expected, rtol=1e-13, atol=0.0)


def test_dichotomous_fallback_uses_one_probability_call():
    model = _model(TwoParameterLogistic)
    calls = []
    original = model.probability

    def counting(theta, item_idx=None):
        calls.append(item_idx)
        return original(theta, item_idx=item_idx)

    model.probability = counting
    AOptimality().get_item_criteria(model, THETA, COVARIANCE, set(range(23)))

    assert calls == [None]


@pytest.mark.parametrize(
    "model_class", [MultidimensionalModel, NoncompensatoryModel, TwoParameterLogistic]
)
def test_kl_criteria_equal_per_item_trace_reference(model_class):
    model = _model(model_class)
    available = {1, 5, 8, 13, 21}

    actual = KullbackLeiblerMCAT().get_item_criteria(
        model, THETA, COVARIANCE, available
    )

    assert list(actual) == sorted(available)
    for item in available:
        expected = np.trace(_reference_matrix(model, THETA, item) @ COVARIANCE)
        assert actual[item] == pytest.approx(expected, rel=1e-13)


def test_kl_subclass_per_item_criteria_are_still_used():
    class PreferHighIndex(KullbackLeiblerMCAT):
        def _compute_criterion(self, model, theta, covariance, item_idx):
            return float(item_idx)

    model = _model(MultidimensionalModel)
    strategy = PreferHighIndex()

    criteria = strategy.get_item_criteria(model, THETA, COVARIANCE, {2, 7, 4})

    assert criteria == {2: 2.0, 7: 7.0, 4: 4.0}
    assert strategy.select_item(model, THETA, COVARIANCE, {2, 7, 4}) == 7


def test_kl_single_item_criterion_ignores_overridden_batches():
    class Delegating(KullbackLeiblerMCAT):
        def get_item_criteria(self, model, theta, covariance, available_items):
            return mcat_selection.MCATSelectionStrategy.get_item_criteria(
                self, model, theta, covariance, available_items
            )

    model = _model(MultidimensionalModel)

    criteria = Delegating().get_item_criteria(model, THETA, COVARIANCE, {3})

    expected = np.trace(_reference_matrix(model, THETA, 3) @ COVARIANCE)
    assert criteria[3] == pytest.approx(expected, rel=1e-13)


def test_kl_integration_points_are_deprecated():
    with pytest.warns(DeprecationWarning, match="n_integration_points"):
        KullbackLeiblerMCAT(n_integration_points=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        create_mcat_selection_strategy("KL")


def test_bayesian_selection_is_the_a_optimality_criterion():
    model = _model(MultidimensionalModel)
    available = set(range(model.n_items))

    assert isinstance(BayesianMCAT(), AOptimality)
    assert BayesianMCAT().get_item_criteria(
        model, THETA, COVARIANCE, available
    ) == AOptimality().get_item_criteria(model, THETA, COVARIANCE, available)


@pytest.mark.parametrize(
    "strategy",
    [DOptimality(), AOptimality(), COptimality(np.array([0.5, -0.3, 1.0]))],
)
def test_stacked_criteria_equal_per_matrix_criteria(strategy):
    model = _model(MultidimensionalModel)
    available = set(range(model.n_items))

    actual = strategy.get_item_criteria(model, THETA, COVARIANCE, available)

    precision = np.linalg.inv(COVARIANCE + np.eye(3) * 1e-8)
    for item in available:
        information = _reference_matrix(model, THETA, item)
        post_cov = np.linalg.inv(precision + information + np.eye(3) * 1e-8)
        expected = strategy._criterion_from_post_cov(post_cov)
        assert actual[item] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_subclassed_per_matrix_criteria_are_still_used():
    class LargestVariance(COptimality):
        def _criterion_from_post_cov(self, post_cov):
            return float(-np.max(np.diag(post_cov)))

    model = _model(MultidimensionalModel)
    criteria = LargestVariance().get_item_criteria(model, THETA, COVARIANCE, {0, 1})

    precision = np.linalg.inv(COVARIANCE + np.eye(3) * 1e-8)
    for item in (0, 1):
        information = _reference_matrix(model, THETA, item)
        post_cov = np.linalg.inv(precision + information + np.eye(3) * 1e-8)
        assert criteria[item] == pytest.approx(-np.max(np.diag(post_cov)))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("d_optimality", DOptimality),
        (" A-OPTIMALITY ", AOptimality),
        ("c-optimality", COptimality),
        ("kl", KullbackLeiblerMCAT),
        ("BAYESIAN", BayesianMCAT),
    ],
)
def test_mcat_factory_normalizes_names(name, expected):
    assert type(create_mcat_selection_strategy(name)) is expected


def test_mcat_factory_reports_canonical_names():
    with pytest.raises(ValueError, match="D-optimality, A-optimality"):
        create_mcat_selection_strategy("E-optimality")
