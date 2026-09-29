"""Independent numeric and storage contracts for affine logistic information."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt import _logistic
from mirt.models.bifactor import BifactorModel
from mirt.models.multidimensional import MultidimensionalModel


def _model(kind, slopes, intercepts):
    slopes = np.asarray(slopes, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    if kind == "multidimensional":
        return MultidimensionalModel(len(slopes), slopes.shape[1]).set_parameters(
            slopes=slopes, intercepts=intercepts
        )
    return BifactorModel(len(slopes), [7] * len(slopes)).set_parameters(
        general_loadings=slopes[:, 0],
        specific_loadings=slopes[:, 1],
        intercepts=intercepts,
    )


def _decimal_matrices(theta, slopes, intercepts):
    with localcontext() as context:
        context.prec = 100
        items = []
        totals = []
        traces = []
        for point in theta:
            matrices = []
            for slope, intercept in zip(slopes, intercepts, strict=True):
                a = [Decimal.from_float(float(value)) for value in slope]
                t = [Decimal.from_float(float(value)) for value in point]
                z = sum(x * y for x, y in zip(a, t, strict=True))
                z += Decimal.from_float(float(intercept))
                tail = (-abs(z)).exp() if abs(z) < 10000 else Decimal(0)
                variance = tail / (1 + tail) ** 2
                matrices.append([[variance * x * y for y in a] for x in a])
            items.append(matrices)
            traces.append(
                [sum(matrix[k][k] for k in range(len(point))) for matrix in matrices]
            )
            totals.append(
                [
                    [
                        sum(matrix[k][column] for matrix in matrices)
                        for column in range(len(point))
                    ]
                    for k in range(len(point))
                ]
            )
        return (
            np.array(items, dtype=float),
            np.array(totals, dtype=float),
            np.array(traces, dtype=float),
        )


@pytest.mark.parametrize("kind", ["multidimensional", "bifactor"])
@pytest.mark.parametrize(
    "slope",
    [
        [1.0, -2.0],
        [1e200, -2e200],
        [1e308, -1e308],
        [1e308, 1e-308],
        [1e-160, -2e-160],
        [1e-200, 1e-200],
        [0.0, 0.0],
    ],
)
def test_information_retains_tails_and_slope_scale(kind, slope):
    intercepts = np.array(
        [-1000.0, -745.0, -710.0, -40.0, 0.0, 40.0, 710.0, 745.0, 1000.0]
    )
    slopes = np.tile(slope, (len(intercepts), 1))
    model = _model(kind, slopes, intercepts)
    theta = np.zeros((1, 2))
    items, total, traces = _decimal_matrices(theta, slopes, intercepts)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information(theta)
        test_matrix = model.test_information_matrix(theta)
        for item in range(model.n_items):
            single = model.information(theta, item)
            matrix = model.item_information_matrix(theta, item)
            np.testing.assert_allclose(single, traces[:, item], rtol=5e-12, atol=0.0)
            np.testing.assert_allclose(matrix, items[:, item], rtol=5e-12, atol=0.0)
            np.testing.assert_array_equal(matrix, matrix.swapaxes(1, 2))
    np.testing.assert_allclose(actual, traces, rtol=5e-12, atol=0.0)
    np.testing.assert_allclose(test_matrix, total, rtol=5e-12, atol=0.0)


@pytest.mark.parametrize("kind", ["multidimensional", "bifactor"])
@pytest.mark.parametrize("slope", [1.0, 1e200, 1e308])
def test_matrix_sum_recovers_information_after_individual_tail_underflow(kind, slope):
    slopes = np.tile([slope, -slope / 2.0], (12, 1))
    intercepts = np.full(12, 747.0 if slope == 1.0 else 1200.0)
    theta = np.zeros((1, 2))
    model = _model(kind, slopes, intercepts)
    _, expected, _ = _decimal_matrices(theta, slopes, intercepts)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.test_information_matrix(theta)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=0.0)
    assert actual[0, 0, 0] > 0.0


@pytest.mark.parametrize("kind", ["multidimensional", "bifactor"])
def test_opposing_cross_factor_terms_cancel_without_nan(kind):
    slopes = np.array([[1e200, -1e200], [1e200, 1e200]])
    intercepts = np.full(2, 400.0)
    theta = np.zeros((1, 2))
    model = _model(kind, slopes, intercepts)
    _, expected, _ = _decimal_matrices(theta, slopes, intercepts)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.test_information_matrix(theta)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=0.0)
    np.testing.assert_array_equal(actual[:, 0, 1], 0.0)


@pytest.mark.parametrize("kind", ["multidimensional", "bifactor"])
def test_information_batches_preserve_strided_readonly_theta(kind, monkeypatch):
    rng = np.random.default_rng(725)
    if kind == "multidimensional":
        slopes = rng.normal(size=(7, 6))
        model = _model(kind, slopes, rng.normal(size=7))
    else:
        model = BifactorModel(7, [2, 5, 2, 8, 15, 22, 8]).set_parameters(
            general_loadings=rng.normal(size=7),
            specific_loadings=rng.normal(size=7),
            intercepts=rng.normal(size=7),
        )
        slopes = model.get_loading_matrix()
    theta = rng.normal(size=(22, 6))[::2, ::-1]
    original_theta = theta.copy()
    theta.flags.writeable = False
    items, total, traces = _decimal_matrices(theta, slopes, model.intercepts)
    original = _logistic._sigmoid_derivative
    sizes = []

    def tracked(logits):
        assert logits.size <= 17
        sizes.append(logits.size)
        return original(logits)

    monkeypatch.setattr(_logistic, "_INFORMATION_CHUNK_ELEMENTS", 17)
    monkeypatch.setattr(_logistic, "_sigmoid_derivative", tracked)
    np.testing.assert_allclose(model.information(theta), traces, rtol=2e-14)
    np.testing.assert_allclose(
        model.test_information_matrix(theta), total, rtol=2e-14, atol=1e-15
    )
    np.testing.assert_allclose(model.information(theta, 3), traces[:, 3], rtol=2e-14)
    np.testing.assert_allclose(
        model.item_information_matrix(theta, 3), items[:, 3], rtol=2e-14
    )
    np.testing.assert_array_equal(theta, original_theta)
    assert len(sizes) > 10


@pytest.mark.parametrize("kind", ["multidimensional", "bifactor"])
def test_information_empty_cohort_and_undefined_theta(kind):
    model = _model(kind, [[1.0, 2.0]], [0.0])
    empty = np.empty((0, 2))
    assert model.information(empty).shape == (0, 1)
    assert model.information(empty, 0).shape == (0,)
    assert model.item_information_matrix(empty, 0).shape == (0, 2, 2)
    assert model.test_information_matrix(empty).shape == (0, 2, 2)
    theta = np.array([[-np.inf, 0.0], [np.inf, 0.0], [np.nan, 0.0]])
    for actual in (
        model.information(theta),
        model.item_information_matrix(theta, 0),
        model.test_information_matrix(theta),
    ):
        np.testing.assert_array_equal(actual[:2], 0.0)
        assert np.isnan(actual[2]).all()


@pytest.mark.parametrize("scale", [0.0, 1e-160, 1.0, 1e200, 1e308])
def test_standardized_loadings_and_communalities_preserve_large_slopes(scale):
    slopes = np.array([[scale, -scale, scale / 2.0]])
    model = MultidimensionalModel(1, 3).set_parameters(slopes=slopes)
    with localcontext() as context:
        context.prec = 100
        a = [Decimal.from_float(float(value)) for value in slopes[0]]
        norm = sum(value * value for value in a)
        expected_loadings = np.array([float(value / (1 + norm).sqrt()) for value in a])
        expected_communality = float(norm / (1 + norm))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.get_factor_loadings()
        communality = model.communalities()
    np.testing.assert_allclose(actual[0], expected_loadings, rtol=5e-15, atol=0.0)
    np.testing.assert_allclose(
        communality, [expected_communality], rtol=5e-15, atol=0.0
    )
    assert 0.0 <= communality[0] <= 1.0
    unstandardized = model.get_factor_loadings(standardized=False)
    unstandardized[:] = 0.0
    np.testing.assert_array_equal(model.slopes, slopes)
