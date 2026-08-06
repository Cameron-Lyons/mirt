"""Structural, reliability, and interoperability contracts for bifactor models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt._core import sigmoid
from mirt.cat.mcat_selection import _compute_item_information_matrix
from mirt.models.bifactor import BifactorModel


def _noncontiguous_model() -> BifactorModel:
    return BifactorModel(5, specific_factors=[10, 10, 42, 42, 99]).set_parameters(
        general_loadings=np.array([0.5, 0.7, 0.4, 0.6, 0.9]),
        specific_loadings=np.array([0.3, 0.2, 0.8, 0.5, 0.4]),
        intercepts=np.array([-0.4, 0.2, 0.0, 0.5, -0.2]),
    )


def test_noncontiguous_labels_are_preserved_and_indexed_contiguously() -> None:
    model = _noncontiguous_model()

    assert model.n_factors == 4
    assert model.n_specific_factors == 3
    assert_array_equal(model.specific_factors, [10, 10, 42, 42, 99])
    assert_array_equal(model.specific_factor_labels, [10, 42, 99])
    assert model.get_factor_structure() == {10: [0, 1], 42: [2, 3], 99: [4]}


def test_specific_factor_labels_are_defensive_copies() -> None:
    model = _noncontiguous_model()
    factors = model.specific_factors
    labels = model.specific_factor_labels

    factors[:] = 0
    labels[:] = 0

    assert_array_equal(model.specific_factors, [10, 10, 42, 42, 99])
    assert_array_equal(model.specific_factor_labels, [10, 42, 99])


def test_bulk_probability_matches_loading_matrix_definition() -> None:
    model = _noncontiguous_model()
    theta = np.random.default_rng(8).normal(size=(41, model.n_factors))

    actual = model.probability(theta)
    expected = sigmoid(theta @ model.get_loading_matrix().T + model.intercepts)

    assert_allclose(actual, expected)
    for item_idx in range(model.n_items):
        assert_allclose(model.probability(theta, item_idx), actual[:, item_idx])


def test_loading_matrix_places_each_specific_loading_once() -> None:
    model = _noncontiguous_model()
    loadings = model.get_loading_matrix()

    assert loadings.shape == (5, 4)
    assert_array_equal(loadings[:, 0], model.general_loadings)
    assert_array_equal(np.count_nonzero(loadings[:, 1:], axis=1), np.ones(5))
    assert_array_equal(loadings[[0, 1], 1], model.specific_loadings[[0, 1]])
    assert_array_equal(loadings[[2, 3], 2], model.specific_loadings[[2, 3]])
    assert loadings[4, 3] == model.specific_loadings[4]


def test_item_parameters_expose_full_slope_vector_for_multidimensional_cat() -> None:
    model = _noncontiguous_model()
    theta = np.array([0.2, -0.4, 0.7, 0.1])
    item_idx = 2

    parameters = model.get_item_parameters(item_idx)
    slopes = model.get_loading_matrix()[item_idx]
    probability = float(model.probability(theta[None, :], item_idx)[0])
    information = _compute_item_information_matrix(model, theta, item_idx)

    assert_array_equal(parameters["slopes"], slopes)
    assert_allclose(
        information,
        probability * (1.0 - probability) * np.outer(slopes, slopes),
    )


def test_copy_preserves_structure_state_and_parameter_independence() -> None:
    model = _noncontiguous_model()
    model._is_fitted = True

    copied = model.copy()
    copied.set_item_parameter(0, "general_loadings", 3.0)

    assert copied.is_fitted
    assert copied.item_names == model.item_names
    assert_array_equal(copied.specific_factors, model.specific_factors)
    assert copied.general_loadings[0] == 3.0
    assert model.general_loadings[0] == 0.5


def test_reliability_coefficients_use_factor_level_score_variance() -> None:
    model = BifactorModel(4, specific_factors=[2, 2, 5, 5]).set_parameters(
        general_loadings=np.array([0.5, 0.7, 0.4, 0.6]),
        specific_loadings=np.array([0.3, 0.2, 0.8, 0.5]),
        intercepts=np.zeros(4),
    )
    general_variance = (0.5 + 0.7 + 0.4 + 0.6) ** 2
    specific_variance = (0.3 + 0.2) ** 2 + (0.8 + 0.5) ** 2
    total_variance = general_variance + specific_variance + 4
    subscale_common = (0.4 + 0.6) ** 2 + (0.8 + 0.5) ** 2

    assert_allclose(model.omega_hierarchical(), general_variance / total_variance)
    assert_allclose(
        model.omega_total(),
        (general_variance + specific_variance) / total_variance,
    )
    assert_allclose(model.omega_subscale(5), subscale_common / (subscale_common + 2))
    assert np.isnan(model.omega_subscale(0))


def test_explained_common_variance_uses_external_factor_labels() -> None:
    model = _noncontiguous_model()
    result = model.explained_common_variance()

    assert set(result) == {"general", "specific_10", "specific_42", "specific_99"}
    assert_allclose(sum(result.values()), 1.0)


def test_zero_common_variance_returns_explicit_nan_without_warnings() -> None:
    model = BifactorModel(4, specific_factors=[3, 3, 8, 8]).set_parameters(
        general_loadings=np.zeros(4),
        specific_loadings=np.zeros(4),
        intercepts=np.zeros(4),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = model.explained_common_variance()

    assert all(np.isnan(value) for value in result.values())
    assert model.omega_hierarchical() == 0.0
    assert model.omega_total() == 0.0


@pytest.mark.parametrize(
    "specific_factors",
    [
        [0.5, 1.0],
        [0.0, np.nan],
        ["general", "specific"],
        [False, True],
    ],
)
def test_noninteger_specific_factor_labels_are_rejected(
    specific_factors: list[object],
) -> None:
    with pytest.raises(ValueError, match="integer labels"):
        BifactorModel(2, specific_factors=specific_factors)  # type: ignore[arg-type]


def test_multidimensional_specific_factor_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match n_items"):
        BifactorModel(4, specific_factors=np.array([[0, 0], [1, 1]]))
