"""Numerical and semantic contracts for logical multidimensional models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt._core import sigmoid
from mirt.models.compensatory import (
    DisjunctiveModel,
    NoncompensatoryModel,
    PartiallyCompensatoryModel,
)

N_ITEMS = 4
N_FACTORS = 3
DISCRIMINATION = np.array(
    [
        [0.7, 1.1, 1.4],
        [1.3, 0.8, 1.0],
        [0.9, 1.6, 0.6],
        [1.5, 1.2, 0.75],
    ]
)
DIFFICULTY = np.array(
    [
        [-0.8, 0.1, 0.6],
        [0.3, -0.5, 1.0],
        [-0.2, 0.7, -0.4],
        [0.9, -0.1, 0.2],
    ]
)
COMPENSATION = np.array(
    [
        [0.4, 0.8, 1.0],
        [0.7, 0.5, 0.9],
        [1.0, 0.3, 0.6],
        [0.2, 0.9, 0.75],
    ]
)
THETA = np.random.default_rng(21).normal(size=(37, N_FACTORS))


def _partial() -> PartiallyCompensatoryModel:
    return PartiallyCompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
        compensation=COMPENSATION,
    )


def _noncompensatory() -> NoncompensatoryModel:
    return NoncompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
    )


def _disjunctive() -> DisjunctiveModel:
    return DisjunctiveModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
    )


def _factor_probabilities() -> np.ndarray:
    linear = DISCRIMINATION[None, :, :] * (THETA[:, None, :] - DIFFICULTY[None, :, :])
    return sigmoid(linear)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            _partial,
            lambda factor_probability: np.prod(
                np.power(factor_probability, COMPENSATION),
                axis=2,
            ),
        ),
        (
            _noncompensatory,
            lambda factor_probability: np.prod(factor_probability, axis=2),
        ),
        (
            _disjunctive,
            lambda factor_probability: 1.0 - np.prod(1.0 - factor_probability, axis=2),
        ),
    ],
)
def test_probability_matches_logical_definition(
    factory: Callable[[], object],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    model = factory()
    actual = model.probability(THETA)

    assert_allclose(actual, expected(_factor_probabilities()))
    for item_idx in range(N_ITEMS):
        assert_allclose(model.probability(THETA, item_idx), actual[:, item_idx])


@pytest.mark.parametrize("factory", [_partial, _noncompensatory, _disjunctive])
def test_analytic_information_matches_probability_derivatives(
    factory: Callable[[], object],
) -> None:
    model = factory()
    probability = model.probability(THETA)
    numerical = np.zeros_like(probability)
    step = 1e-5

    for factor_idx in range(N_FACTORS):
        theta_plus = THETA.copy()
        theta_minus = THETA.copy()
        theta_plus[:, factor_idx] += step
        theta_minus[:, factor_idx] -= step
        derivative = (
            model.probability(theta_plus) - model.probability(theta_minus)
        ) / (2.0 * step)
        numerical += derivative**2 / (probability * (1.0 - probability))

    assert_allclose(model.information(THETA), numerical, rtol=2e-7, atol=1e-11)


@pytest.mark.parametrize("factory", [_partial, _noncompensatory, _disjunctive])
def test_fisher_matrices_are_symmetric_psd_and_match_scalar_information(
    factory: Callable[[], object],
) -> None:
    model = factory()
    expected_test_matrix = np.zeros((len(THETA), N_FACTORS, N_FACTORS))

    for item_idx in range(N_ITEMS):
        item_matrix = model.item_information_matrix(THETA, item_idx)
        expected_test_matrix += item_matrix

        assert_allclose(item_matrix, np.swapaxes(item_matrix, 1, 2))
        assert np.all(np.linalg.eigvalsh(item_matrix) >= -1e-12)
        assert_allclose(
            np.trace(item_matrix, axis1=1, axis2=2),
            model.information(THETA, item_idx),
        )

    assert_allclose(model.test_information_matrix(THETA), expected_test_matrix)
    assert_allclose(
        np.trace(expected_test_matrix, axis1=1, axis2=2),
        np.sum(model.information(THETA), axis=1),
    )


def test_all_one_compensation_weights_equal_noncompensatory_limit() -> None:
    partial = PartiallyCompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
        compensation=np.ones((N_ITEMS, N_FACTORS)),
    )
    noncompensatory = _noncompensatory()

    assert_allclose(partial.probability(THETA), noncompensatory.probability(THETA))
    assert_allclose(partial.information(THETA), noncompensatory.information(THETA))
    assert_allclose(
        partial.test_information_matrix(THETA),
        noncompensatory.test_information_matrix(THETA),
    )


def test_zero_weight_ignores_dimension_probability_and_information() -> None:
    compensation = np.ones((N_ITEMS, N_FACTORS))
    compensation[:, 1] = 0.0
    model = PartiallyCompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=DIFFICULTY,
        compensation=compensation,
    )
    shifted = THETA.copy()
    shifted[:, 1] += 100.0

    assert_allclose(model.probability(shifted), model.probability(THETA))
    matrices = model.test_information_matrix(THETA)
    assert_allclose(matrices[:, 1], 0.0, atol=1e-15)
    assert_allclose(matrices[:, :, 1], 0.0, atol=1e-15)


def test_disjunctive_is_probability_dual_of_conjunctive_model() -> None:
    disjunctive = _disjunctive()
    dual_conjunctive = NoncompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
        discrimination=DISCRIMINATION,
        difficulty=-DIFFICULTY,
    )

    assert_allclose(
        disjunctive.probability(THETA),
        1.0 - dual_conjunctive.probability(-THETA),
    )


@pytest.mark.parametrize(
    "compensation",
    [
        np.full((N_ITEMS, N_FACTORS), -0.1),
        np.full((N_ITEMS, N_FACTORS), 1.1),
        np.full((N_ITEMS, N_FACTORS), np.nan),
        np.full((N_ITEMS, N_FACTORS), np.inf),
    ],
)
def test_compensation_weights_require_finite_unit_interval(
    compensation: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        PartiallyCompensatoryModel(N_ITEMS, N_FACTORS).set_parameters(
            compensation=compensation
        )


@pytest.mark.parametrize(
    "model_type",
    [PartiallyCompensatoryModel, NoncompensatoryModel, DisjunctiveModel],
)
def test_models_require_multiple_factors(model_type: type[object]) -> None:
    with pytest.raises(ValueError, match="at least 2 factors"):
        model_type(n_items=3, n_factors=1)
