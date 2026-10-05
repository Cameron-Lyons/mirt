"""Exact Fisher information matrices for multidimensional GRM and NRM items."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from mirt.cat.mcat_selection import _compute_item_information_matrix
from mirt.models.polytomous import GradedResponseModel, NominalResponseModel

PolytomousMatrixModel = GradedResponseModel | NominalResponseModel


def _graded(n_factors: int) -> GradedResponseModel:
    counts = [2, 4, 5, 3]
    model = GradedResponseModel(4, counts, n_factors=n_factors)
    rng = np.random.default_rng(10 + n_factors)
    thresholds = np.zeros((4, 4))
    for item, count in enumerate(counts):
        thresholds[item, : count - 1] = np.sort(rng.normal(size=count - 1))
    discrimination = rng.uniform(0.3, 1.8, size=(4, n_factors))
    if n_factors > 1:
        # A negative loading is valid while the summed slope stays positive.
        discrimination[1, 0] = -0.4
    return model.set_parameters(
        discrimination=discrimination.reshape(model.discrimination.shape),
        thresholds=thresholds,
    )


def _nominal(n_factors: int) -> NominalResponseModel:
    counts = [3, 5, 2, 4]
    model = NominalResponseModel(4, counts, n_factors=n_factors)
    rng = np.random.default_rng(20 + n_factors)
    return model.set_parameters(
        slopes=rng.normal(scale=0.9, size=model.slopes.shape),
        intercepts=rng.normal(size=model.intercepts.shape),
    )


FACTORIES: list[Callable[[], PolytomousMatrixModel]] = [
    lambda: _graded(1),
    lambda: _graded(2),
    lambda: _graded(3),
    lambda: _nominal(1),
    lambda: _nominal(2),
    lambda: _nominal(3),
]


def _theta(n_factors: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(5)
    return rng.uniform(-2.0, 2.0, size=(6, n_factors))


def _finite_difference_fisher(
    model: PolytomousMatrixModel, theta: NDArray[np.float64], item_idx: int
) -> NDArray[np.float64]:
    step = 1e-5
    gradients = []
    for factor in range(model.n_factors):
        shift = np.zeros(model.n_factors)
        shift[factor] = step
        gradients.append(
            (
                model.probability(theta + shift, item_idx)
                - model.probability(theta - shift, item_idx)
            )
            / (2.0 * step)
        )
    gradient = np.stack(gradients, axis=-1)
    probability = model.probability(theta, item_idx)
    return np.einsum("ncf,ncg,nc->nfg", gradient, gradient, 1.0 / probability)


@pytest.mark.parametrize("factory", FACTORIES)
def test_item_matrix_matches_finite_difference_fisher_and_scalar_trace(
    factory: Callable[[], PolytomousMatrixModel],
) -> None:
    model = factory()
    theta = _theta(model.n_factors)

    for item_idx in range(model.n_items):
        matrix = model.item_information_matrix(theta, item_idx)

        assert matrix.shape == (len(theta), model.n_factors, model.n_factors)
        np.testing.assert_allclose(
            matrix, _finite_difference_fisher(model, theta, item_idx), atol=1e-7
        )
        np.testing.assert_allclose(matrix, np.swapaxes(matrix, 1, 2), atol=1e-15)
        assert np.linalg.eigvalsh(matrix).min() > -1e-12
        np.testing.assert_allclose(
            np.trace(matrix, axis1=1, axis2=2),
            model.information(theta, item_idx),
            rtol=1e-10,
            atol=1e-13,
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_test_matrix_sums_items_and_traces_to_test_information(
    factory: Callable[[], PolytomousMatrixModel],
) -> None:
    model = factory()
    theta = _theta(model.n_factors)
    items = [model.item_information_matrix(theta, item) for item in range(4)]

    total = model.test_information_matrix(theta)

    np.testing.assert_allclose(total, sum(items), rtol=1e-14)
    np.testing.assert_allclose(
        np.trace(total, axis1=1, axis2=2), model.information(theta), rtol=1e-12
    )
    empty = theta[:0]
    assert model.item_information_matrix(empty, 0).shape == (
        0,
        model.n_factors,
        model.n_factors,
    )
    assert model.test_information_matrix(empty).shape == (
        0,
        model.n_factors,
        model.n_factors,
    )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("item_idx", [-1, 4, True, 1.0])
def test_item_matrix_rejects_invalid_item_indices(
    factory: Callable[[], PolytomousMatrixModel], item_idx: object
) -> None:
    model = factory()

    with pytest.raises(IndexError):
        model.item_information_matrix(_theta(model.n_factors), item_idx)  # type: ignore[arg-type]


def test_mcat_uses_exact_graded_matrix_instead_of_first_category_fallback() -> None:
    model = GradedResponseModel(5, 4, n_factors=2)
    discrimination = model.discrimination.copy()
    discrimination[0] = [1.5, 0.5]
    model.set_parameters(discrimination=discrimination)
    theta = np.array([0.3, -0.2])

    matrix = _compute_item_information_matrix(model, theta, item_idx=0)

    expected = _finite_difference_fisher(model, theta[None, :], 0)[0]
    np.testing.assert_allclose(matrix, expected, atol=1e-7)
    np.testing.assert_allclose(matrix[0, 0], 0.5691, atol=5e-5)


@pytest.mark.parametrize("factory", FACTORIES[1:3] + FACTORIES[4:])
def test_mcat_uses_native_polytomous_matrices(
    factory: Callable[[], PolytomousMatrixModel],
) -> None:
    model = factory()
    theta = _theta(model.n_factors)

    for item_idx in range(model.n_items):
        np.testing.assert_allclose(
            _compute_item_information_matrix(model, theta[0], item_idx),
            model.item_information_matrix(theta[:1], item_idx)[0],
        )
