"""Tests for aligned cognitive-diagnosis probability evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.cdm import DINA, DINO
from mirt.models.cdm_advanced import GDINA, HigherOrderCDM

Q_MATRIX = np.array(
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
)
ALPHA = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ]
)
ITEM_INDICES = np.array([5, 0, 2, 3, 1, 4, 2])


def _dina() -> DINA:
    return DINA(6, 3, Q_MATRIX).set_parameters(
        slip=np.array([0.08, 0.12, 0.16, 0.2, 0.1, 0.14]),
        guess=np.array([0.18, 0.2, 0.22, 0.15, 0.25, 0.19]),
    )


def _dino() -> DINO:
    return DINO(6, 3, Q_MATRIX).set_parameters(
        slip=np.array([0.09, 0.13, 0.17, 0.21, 0.11, 0.15]),
        guess=np.array([0.17, 0.19, 0.21, 0.14, 0.24, 0.18]),
    )


def _gdina() -> GDINA:
    model = GDINA(
        6,
        3,
        Q_MATRIX,
        reduced_models=["saturated", "DINA", "DINO", "ACDM", "LLM", "RRUM"],
    )
    parameters = [
        np.array([0.1, 0.3, 0.6, 0.9]),
        np.array([0.2, 0.8]),
        np.array([0.15, 0.75]),
        np.array([0.1, 0.2, 0.3]),
        np.array([-1.0, 0.7, 1.1]),
        np.array([0.9, 0.5, 0.7]),
    ]
    for item_idx, delta in enumerate(parameters):
        model.set_delta_parameters(item_idx, delta)
    return model


@pytest.mark.parametrize("factory", [_dina, _dino, _gdina])
def test_discrete_cdm_pairs_match_itemwise_probabilities_without_dispatch(
    factory: Callable[[], BaseItemModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = factory()
    expected = np.array(
        [
            model.probability(ALPHA[row : row + 1], int(item_idx))[0]
            for row, item_idx in enumerate(ITEM_INDICES)
        ]
    )

    def fail_item_dispatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("paired evaluation dispatched through probability()")

    monkeypatch.setattr(model, "probability", fail_item_dispatch)

    actual = model.probability_pairs(ALPHA, ITEM_INDICES)
    empty = model.probability_pairs(
        np.empty((0, model.n_factors), dtype=np.int_),
        np.array([], dtype=np.int_),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert empty.shape == (0,)


def test_higher_order_pairs_accept_scalar_ability_inputs_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HigherOrderCDM(6, 3, Q_MATRIX).set_higher_order_params(
        loadings=np.array([0.7, 1.1, 1.4]),
        thresholds=np.array([-0.5, 0.2, 0.8]),
    )
    theta = np.array([-1.2, -0.4, 0.0, 0.6, 1.1, 1.8, 0.3])
    expected = np.array(
        [
            model.probability(theta[row : row + 1], int(item_idx))[0]
            for row, item_idx in enumerate(ITEM_INDICES)
        ]
    )

    def fail_item_dispatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("paired evaluation dispatched through probability()")

    monkeypatch.setattr(model, "probability", fail_item_dispatch)

    actual = model.probability_pairs(theta, ITEM_INDICES)
    empty = model.probability_pairs(
        np.empty((0, 1)),
        np.array([], dtype=np.int_),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert empty.shape == (0,)


def test_higher_order_pairs_validate_item_alignment() -> None:
    model = HigherOrderCDM(6, 3, Q_MATRIX)

    with pytest.raises(MirtValidationError, match="one entry per theta row"):
        model.probability_pairs(np.array([-0.5, 0.5]), np.array([0]))
    with pytest.raises(MirtValidationError, match="valid model items"):
        model.probability_pairs(np.array([-0.5, 0.5]), np.array([0, 6]))
