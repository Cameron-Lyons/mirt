"""Regression coverage for native multidimensional information matrices."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat.mcat_selection import DOptimality, _compute_item_information_matrix
from mirt.models.bifactor import BifactorModel
from mirt.models.compensatory import (
    DisjunctiveModel,
    NoncompensatoryModel,
    PartiallyCompensatoryModel,
)
from mirt.models.multidimensional import MultidimensionalModel

THETA = np.array(
    [
        [0.2, -0.7, 1.1],
        [-1.0, 0.4, 0.3],
        [0.8, 1.2, -0.5],
        [0.0, 0.0, 0.0],
    ]
)
DISCRIMINATION = np.array(
    [
        [0.8, 1.3, 0.6],
        [1.2, 0.7, 1.5],
        [0.5, 1.1, 0.9],
    ]
)
DIFFICULTY = np.array(
    [
        [-0.4, 0.3, 0.8],
        [0.1, -0.5, 0.2],
        [0.7, 0.2, -0.6],
    ]
)


def _noncompensatory() -> NoncompensatoryModel:
    return NoncompensatoryModel(3, 3).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
    )


def _disjunctive() -> DisjunctiveModel:
    return DisjunctiveModel(3, 3).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
    )


def _partially_compensatory() -> PartiallyCompensatoryModel:
    return PartiallyCompensatoryModel(3, 3).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
        compensation=np.array(
            [
                [0.3, 0.8, 0.5],
                [0.9, 0.4, 0.7],
                [0.6, 0.2, 1.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "factory",
    [_noncompensatory, _disjunctive, _partially_compensatory],
)
def test_mcat_uses_model_native_information_matrix(
    factory: Callable[[], object],
) -> None:
    model = factory()
    expected = model.item_information_matrix(THETA[:1], item_idx=0)[0]

    actual = _compute_item_information_matrix(model, THETA[0], item_idx=0)

    assert_allclose(actual, expected)


def test_native_information_changes_d_optimal_item_choice() -> None:
    theta = np.array([-0.82876700, 0.15515576, -0.68981861])
    covariance = np.diag([0.87932990, 1.98609404, 0.67762722])
    model = NoncompensatoryModel(2, 3).set_parameters(
        discrimination=np.array(
            [
                [1.77357432, 1.43628674, 1.29205597],
                [1.13241693, 2.45901575, 0.43866336],
            ]
        ),
        difficulty=np.array(
            [
                [-0.67551457, 0.06745326, 0.01311421],
                [-0.08432634, -1.00626984, 1.09307522],
            ]
        ),
    )

    selected = DOptimality().select_item(
        model,
        theta,
        covariance,
        available_items={0, 1},
    )

    assert selected == 0


def _assert_information_contract(model, slopes: np.ndarray) -> None:
    expected_test = np.zeros((len(THETA), model.n_factors, model.n_factors))
    for item_idx in range(model.n_items):
        probability = model.probability(THETA, item_idx)
        expected = (
            probability[:, None, None]
            * (1.0 - probability[:, None, None])
            * np.outer(slopes[item_idx], slopes[item_idx])[None, :, :]
        )
        item_matrix = model.item_information_matrix(THETA, item_idx)

        assert_allclose(item_matrix, expected)
        assert_allclose(item_matrix, np.swapaxes(item_matrix, 1, 2))
        assert_allclose(
            np.trace(item_matrix, axis1=1, axis2=2),
            model.information(THETA, item_idx),
        )
        expected_test += expected

    assert_allclose(model.test_information_matrix(THETA), expected_test)
    assert_allclose(
        np.trace(expected_test, axis1=1, axis2=2),
        model.information(THETA).sum(axis=1),
    )


def test_multidimensional_model_exposes_item_and_test_fisher_matrices() -> None:
    slopes = np.array(
        [
            [0.5, 1.1, -0.2],
            [1.4, -0.3, 0.8],
            [0.7, 0.9, 1.2],
        ]
    )
    model = MultidimensionalModel(3, 3).set_parameters(
        slopes=slopes,
        intercepts=np.array([-0.4, 0.2, 0.7]),
    )

    _assert_information_contract(model, slopes)


def test_bifactor_model_exposes_item_and_test_fisher_matrices() -> None:
    model = BifactorModel(3, specific_factors=[4, 9, 4]).set_parameters(
        general_loadings=np.array([0.5, 1.4, 0.7]),
        specific_loadings=np.array([1.1, 0.8, 0.9]),
        intercepts=np.array([-0.4, 0.2, 0.7]),
    )
    slopes = model.get_loading_matrix()

    _assert_information_contract(model, slopes)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MultidimensionalModel(2, 3),
        lambda: BifactorModel(2, specific_factors=[0, 1]),
    ],
)
@pytest.mark.parametrize("item_idx", [-1, 2])
def test_information_matrix_rejects_invalid_item_index(
    factory: Callable[[], object],
    item_idx: int,
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        factory().item_information_matrix(THETA[:1], item_idx)


@pytest.mark.parametrize(
    "theta",
    [np.array([0.0, 1.0]), np.array([0.0, np.nan, 1.0])],
)
def test_mcat_information_rejects_invalid_theta(theta: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite factor values"):
        _compute_item_information_matrix(_noncompensatory(), theta, item_idx=0)
