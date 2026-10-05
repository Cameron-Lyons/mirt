"""Vectorized MEI criteria must equal the per-candidate reference algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.cat.selection as selection
from mirt.cat.selection import MaxExpectedInformation
from mirt.constants import PROB_EPSILON
from mirt.models import (
    GradedResponseModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


def _model(kind: str, n_items: int = 30) -> Any:
    rng = np.random.default_rng(11)
    discrimination = rng.lognormal(0.0, 0.35, n_items)
    if kind == "GRM":
        categories = [2 + item % 4 for item in range(n_items)]
        model = GradedResponseModel(n_items=n_items, n_categories=categories)
        model.set_parameters(
            discrimination=discrimination,
            thresholds=np.sort(rng.normal(0.0, 1.0, (n_items, 4)), axis=1),
        )
    else:
        model_class = TwoParameterLogistic if kind == "2PL" else ThreeParameterLogistic
        model = model_class(n_items=n_items)
        parameters = {
            "discrimination": discrimination,
            "difficulty": rng.normal(0.0, 1.0, n_items),
        }
        if kind == "3PL":
            parameters["guessing"] = rng.uniform(0.05, 0.25, n_items)
        model.set_parameters(**parameters)
    model._is_fitted = True
    return model


def _history(model: Any, length: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(length)
    items = rng.choice(model.n_items, size=length, replace=False).tolist()
    categories = model.n_categories if model.is_polytomous else [2] * model.n_items
    responses = [int(rng.integers(categories[item])) for item in items]
    return items, responses


def _reference_criteria(
    strategy: MaxExpectedInformation,
    model: Any,
    theta: float,
    available: set[int],
    administered: list[int],
    responses: list[int],
) -> dict[int, float]:
    """The original per-candidate, per-administered-item evaluation."""
    history_log_mass = strategy._history_log_mass(model, administered, responses)
    criteria = {}
    for item_idx in available:
        current = strategy._response_probabilities(model, theta, item_idx)
        nodes = strategy._node_response_probabilities(model, item_idx)
        clipped = np.clip(nodes, PROB_EPSILON, 1.0 - PROB_EPSILON)
        log_mass = history_log_mass[:, None] + np.log(clipped)
        log_mass -= np.max(log_mass, axis=0, keepdims=True)
        posterior_mass = np.exp(log_mass)
        posterior_mass /= posterior_mass.sum(axis=0, keepdims=True)
        hypothetical_theta = (posterior_mass.T @ strategy._theta_nodes)[:, None]
        test_information = np.zeros(len(current))
        for provisional_item in (*administered, item_idx):
            information = np.asarray(
                model.information(hypothetical_theta, item_idx=provisional_item)
            )
            test_information += information.reshape(len(current), -1).sum(axis=1)
        criteria[item_idx] = float(current @ test_information)
    return criteria


@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM"])
@pytest.mark.parametrize("history_length", [0, 5, 20])
@pytest.mark.parametrize("theta", [-1.3, 0.4])
def test_vectorized_criteria_match_per_candidate_reference(kind, history_length, theta):
    model = _model(kind)
    strategy = MaxExpectedInformation(n_quadpts=17, theta_bounds=(-3.5, 4.0))
    administered, responses = _history(model, history_length)
    available = set(range(model.n_items)) - set(administered)

    actual = strategy.get_item_criteria(
        model, theta, available, administered_items=administered, responses=responses
    )
    expected = _reference_criteria(
        strategy, model, theta, available, administered, responses
    )

    assert list(actual) == sorted(available)
    assert_allclose(
        [actual[item] for item in sorted(available)],
        [expected[item] for item in sorted(available)],
        rtol=1e-12,
        atol=0.0,
    )
    assert strategy.select_item(
        model, theta, available, administered, responses
    ) == max(sorted(expected), key=expected.__getitem__)


class _CountingModel:
    """Delegate to a model while counting curve and information calls."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.probability_calls = 0
        self.information_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        return self._model.probability(theta, item_idx=item_idx)

    def information(self, theta, item_idx=None):
        self.information_calls += 1
        return self._model.information(theta, item_idx=item_idx)


def test_dichotomous_selection_uses_a_fixed_number_of_model_calls():
    model = _CountingModel(_model("3PL", n_items=40))
    strategy = MaxExpectedInformation(n_quadpts=21)
    administered, responses = _history(model, 15)
    available = set(range(model.n_items)) - set(administered)

    strategy.get_item_criteria(model, 0.2, available, administered, responses)

    assert model.probability_calls == 2
    assert model.information_calls == 1


def test_dichotomous_information_is_evaluated_in_bounded_row_blocks(monkeypatch):
    model = _model("2PL", n_items=40)
    strategy = MaxExpectedInformation(n_quadpts=15)
    administered, responses = _history(model, 6)
    available = set(range(model.n_items)) - set(administered)
    expected = strategy.get_item_criteria(
        model, 0.3, available, administered, responses
    )

    monkeypatch.setattr(selection, "_MAX_INFORMATION_VALUES", 3 * model.n_items)
    counting = _CountingModel(model)
    actual = strategy.get_item_criteria(
        counting, 0.3, available, administered, responses
    )

    assert counting.information_calls == -(-2 * len(available) // 3)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_polytomous_selection_evaluates_each_administered_item_once():
    model = _CountingModel(_model("GRM", n_items=12))
    strategy = MaxExpectedInformation(n_quadpts=15)
    administered, responses = _history(model, 4)
    available = set(range(model.n_items)) - set(administered)

    strategy.get_item_criteria(model, -0.4, available, administered, responses)

    assert model.information_calls == len(administered) + len(available)
    assert model.probability_calls == 2 * len(available)


class _TotalInformation2PL(TwoParameterLogistic):
    """Return only test information from the bulk call, like polytomous models."""

    def information(self, theta, item_idx=None):
        values = super().information(theta, item_idx=item_idx)
        if item_idx is None:
            return np.asarray(values).sum(axis=1)
        return values


def test_bulk_information_without_item_columns_uses_itemwise_evaluation():
    reference = _model("2PL", n_items=10)
    model = _TotalInformation2PL(n_items=10)
    model.set_parameters(**reference.parameters)
    model._is_fitted = True
    strategy = MaxExpectedInformation(n_quadpts=15)
    administered, responses = [1, 4], [1, 0]
    available = {0, 2, 3, 7, 9}

    actual = strategy.get_item_criteria(model, 0.1, available, administered, responses)
    expected = _reference_criteria(
        strategy, reference, 0.1, available, administered, responses
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    "available, message",
    [
        ({-1, 2}, "out of range"),
        ({2, 10}, "out of range"),
        ({1.0, 2}, "must be integers"),
        ({True, 2}, "must be integers"),
    ],
)
def test_invalid_candidate_indices_are_rejected(available, message):
    model = _model("2PL", n_items=10)
    with pytest.raises(ValueError, match=message):
        MaxExpectedInformation().get_item_criteria(model, 0.0, available)


def test_empty_candidates_return_no_criteria():
    model = _model("2PL", n_items=10)
    assert MaxExpectedInformation().get_item_criteria(model, 0.0, set()) == {}


def test_single_item_criterion_ignores_overridden_batches():
    class Delegating(MaxExpectedInformation):
        def get_item_criteria(self, model, theta, available_items, *args, **kwargs):
            return selection.ItemSelectionStrategy.get_item_criteria(
                self, model, theta, available_items, *args, **kwargs
            )

    model = _model("2PL", n_items=10)
    strategy = Delegating(n_quadpts=15)

    actual = strategy.get_item_criteria(model, 0.2, {3, 6})

    expected = _reference_criteria(strategy, model, 0.2, {3, 6}, [], [])
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)
