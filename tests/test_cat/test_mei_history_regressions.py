"""Regression coverage for history-aware expected-information selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mirt.cat.engine import CATEngine
from mirt.cat.exposure import Randomesque
from mirt.cat.selection import (
    ItemSelectionStrategy,
    MaxExpectedInformation,
    MaxFisherInformation,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _fitted_2pl() -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=5).set_parameters(
        discrimination=np.array(
            [0.46002833, 0.30860493, 0.71829945, 2.93085020, 4.07184839]
        ),
        difficulty=np.array(
            [-0.03243128, -1.28356925, -2.36643547, 0.98954625, 0.16663349]
        ),
    )
    model._is_fitted = True
    return model


def _fitted_grm() -> GradedResponseModel:
    model = GradedResponseModel(n_items=4, n_categories=[2, 3, 5, 4])
    thresholds = np.zeros((4, 4))
    thresholds[0, :1] = [0.25]
    thresholds[1, :2] = [-1.1, 0.65]
    thresholds[2, :4] = [-1.7, -0.45, 0.55, 1.6]
    thresholds[3, :3] = [-1.25, 0.1, 1.15]
    model.set_parameters(
        discrimination=np.array([0.75, 1.2, 1.8, 0.65]),
        thresholds=thresholds,
    )
    model._is_fitted = True
    return model


def _bounded_eap(
    model: Any,
    item_indices: list[int],
    responses: list[int],
    n_quadpts: int,
    theta_bounds: tuple[float, float],
) -> float:
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_quadpts)
    lower, upper = theta_bounds
    half_width = (upper - lower) / 2.0
    nodes = (upper + lower) / 2.0 + half_width * raw_nodes
    integration_weights = half_width * raw_weights
    prior_density = np.exp(-0.5 * np.square(nodes)) / np.sqrt(2.0 * np.pi)

    response_matrix = np.full((1, model.n_items), -1, dtype=np.int_)
    response_matrix[0, item_indices] = responses
    log_likelihood = model.log_likelihood_batch(response_matrix, nodes[:, None]).ravel()
    log_mass = log_likelihood + np.log(integration_weights * prior_density)
    log_mass -= np.max(log_mass)
    posterior_mass = np.exp(log_mass)
    posterior_mass /= posterior_mass.sum()
    return float(posterior_mass @ nodes)


def _response_probabilities(model: Any, theta: float, item_idx: int) -> np.ndarray:
    probabilities = np.asarray(
        model.probability(np.array([[theta]]), item_idx=item_idx),
        dtype=np.float64,
    ).ravel()
    if model.is_polytomous:
        return probabilities
    return np.array([1.0 - probabilities[0], probabilities[0]])


def _reference_criterion(
    model: Any,
    theta: float,
    item_idx: int,
    administered_items: list[int],
    responses: list[int],
    n_quadpts: int,
    theta_bounds: tuple[float, float],
) -> float:
    provisional_items = [*administered_items, item_idx]
    expected_information = 0.0
    for category, probability in enumerate(
        _response_probabilities(model, theta, item_idx)
    ):
        hypothetical_theta = _bounded_eap(
            model,
            provisional_items,
            [*responses, category],
            n_quadpts,
            theta_bounds,
        )
        theta_array = np.array([[hypothetical_theta]])
        test_information = sum(
            float(model.information(theta_array, item_idx=index).sum())
            for index in provisional_items
        )
        expected_information += float(probability) * test_information
    return expected_information


@pytest.mark.parametrize(
    (
        "model",
        "theta",
        "administered",
        "responses",
        "available",
        "n_quadpts",
        "bounds",
    ),
    [
        (_fitted_2pl(), 0.9581138988376822, [0, 1], [0, 0], {2, 3, 4}, 31, (-1.5, 1.5)),
        (_fitted_grm(), 0.35, [0, 1], [1, 2], {2, 3}, 25, (-2.25, 1.75)),
    ],
)
def test_mei_matches_bounded_posterior_reference(
    model,
    theta,
    administered,
    responses,
    available,
    n_quadpts,
    bounds,
):
    strategy = MaxExpectedInformation(n_quadpts=n_quadpts, theta_bounds=bounds)

    actual = strategy.get_item_criteria(
        model,
        theta,
        available,
        administered_items=administered,
        responses=responses,
    )
    expected = {
        item_idx: _reference_criterion(
            model,
            theta,
            item_idx,
            administered,
            responses,
            n_quadpts,
            bounds,
        )
        for item_idx in available
    }

    assert actual == pytest.approx(expected, rel=2e-11, abs=2e-12)


def test_history_aware_mei_can_select_differently_from_mfi():
    model = _fitted_2pl()
    theta = 0.9581138988376822
    available = {2, 3, 4}

    mfi_item = MaxFisherInformation().select_item(model, theta, available)
    mei_item = MaxExpectedInformation(
        n_quadpts=31,
        theta_bounds=(-1.5, 1.5),
    ).select_item(
        model,
        theta,
        available,
        administered_items=[0, 1],
        responses=[0, 0],
    )

    assert mfi_item == 3
    assert mei_item == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_quadpts": 4},
        {"n_quadpts": 5.5},
        {"theta_bounds": (-np.inf, 2.0)},
        {"theta_bounds": (1.0, 1.0)},
    ],
)
def test_mei_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MaxExpectedInformation(**kwargs)


def test_mei_rejects_misaligned_or_invalid_history():
    strategy = MaxExpectedInformation()
    model = _fitted_grm()

    with pytest.raises(ValueError, match="equal length"):
        strategy.select_item(
            model,
            theta=0.0,
            available_items={2},
            administered_items=[0, 1],
            responses=[1],
        )
    with pytest.raises(ValueError, match="category range"):
        strategy.select_item(
            model,
            theta=0.0,
            available_items={2},
            administered_items=[0, 1],
            responses=[1, 3],
        )


class _HistoryRecorder(ItemSelectionStrategy):
    def __init__(self) -> None:
        self.history: tuple[list[int], list[int]] | None = None

    def select_item(
        self,
        model: Any,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        return min(available_items)

    def get_item_criteria(
        self,
        model: Any,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        self.history = (list(administered_items or []), list(responses or []))
        return {item_idx: -float(item_idx) for item_idx in available_items}


def test_cat_configures_mei_and_propagates_randomesque_history():
    engine = CATEngine(
        _fitted_2pl(),
        item_selection="MEI",
        n_quadpts=17,
        theta_bounds=(-1.25, 2.5),
    )
    assert isinstance(engine._selection, MaxExpectedInformation)
    assert engine._selection.n_quadpts == 17
    assert engine._selection.theta_bounds == (-1.25, 2.5)

    recorder = _HistoryRecorder()
    engine = CATEngine(
        _fitted_2pl(),
        item_selection=recorder,
        exposure_control=Randomesque(k=1, seed=19),
    )
    engine._items_administered = [0, 1]
    engine._responses = [1, 0]
    engine._available_items = {2, 3, 4}

    assert engine.select_next_item() == 2
    assert recorder.history == ([0, 1], [1, 0])
