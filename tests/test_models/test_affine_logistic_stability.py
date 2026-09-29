"""Finite dot-product cancellation and bounded affine logistic queries."""

from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest

from mirt import _logistic
from mirt.models.bifactor import BifactorModel
from mirt.models.multidimensional import MultidimensionalModel

_MAX_FLOAT = np.finfo(float).max


def _model(kind, slopes, intercepts):
    slopes = np.asarray(slopes, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    if kind == "mirt":
        return MultidimensionalModel(len(slopes), slopes.shape[1]).set_parameters(
            slopes=slopes, intercepts=intercepts
        )
    return BifactorModel(len(slopes), [7] * len(slopes)).set_parameters(
        general_loadings=slopes[:, 0],
        specific_loadings=slopes[:, 1],
        intercepts=intercepts,
    )


def _decimal_curve(point, slope, intercept):
    with localcontext() as context:
        # Resolve products of the original binary floats before reducing
        # precision for the nonlinear part of the reference calculation.
        context.prec = 3000
        points = [Decimal.from_float(float(value)) for value in point]
        slopes = [Decimal.from_float(float(value)) for value in slope]
        logit = sum(a * t for a, t in zip(slopes, points, strict=True))
        logit += Decimal.from_float(float(intercept))
        norm = sum(a * a for a in slopes)
        context.prec = 100
        tail = (-abs(logit)).exp() if abs(logit) < 10000 else Decimal(0)
        probability = (1 if logit >= 0 else tail) / (1 + tail)
        variance = tail / (1 + tail) ** 2
        matrix = [[variance * a * b for b in slopes] for a in slopes]
        return float(probability), float(variance * norm), np.array(matrix, dtype=float)


@pytest.mark.parametrize("kind", ["mirt", "bifactor"])
@pytest.mark.parametrize(
    "slopes,point,intercept",
    [
        ([1e308, -1e308], [2.0, 2.0], 2.0),
        ([1e200, -1e200], [1e200, 1e200], 1000.0),
        ([1e200, -1e200], [1e200, 1e200], -1000.0),
        ([1e16 + 2.0, -1e16], [np.nextafter(1.0, 2.0), 1.0], 0.0),
        ([1.0, 1.0], [_MAX_FLOAT, -_MAX_FLOAT], 0.2),
        ([1e308, 1e308], [2.0, 2.0], -_MAX_FLOAT),
        ([-1e308, -1e308], [2.0, 2.0], _MAX_FLOAT),
        ([1.5, 0.0], [1e308, 0.0], -1.5e308),
        ([0.5, 0.5], [_MAX_FLOAT, _MAX_FLOAT], -_MAX_FLOAT),
        ([0.0, 0.0], [_MAX_FLOAT, _MAX_FLOAT], 2.0),
        ([1e-308, -1e-308], [_MAX_FLOAT, _MAX_FLOAT], 2.0),
    ],
)
def test_finite_affine_queries_match_exact_dot_product(kind, slopes, point, intercept):
    model = _model(kind, [slopes], [intercept])
    probability, information, matrix = _decimal_curve(point, slopes, intercept)
    for count in (1, 3):
        theta = np.tile(point, (count, 1))
        original = theta.copy()
        theta.flags.writeable = False
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for item_idx in (None, 0):
                np.testing.assert_allclose(
                    model.probability(theta, item_idx),
                    probability,
                    rtol=5e-12,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    model.information(theta, item_idx),
                    information,
                    rtol=5e-12,
                    atol=0.0,
                )
            np.testing.assert_allclose(
                model.probability_pairs(theta, np.zeros(count, dtype=int)),
                probability,
                rtol=5e-12,
                atol=0.0,
            )
            for actual in (
                model.item_information_matrix(theta, 0),
                model.test_information_matrix(theta),
            ):
                np.testing.assert_allclose(
                    actual, np.broadcast_to(matrix, actual.shape), rtol=5e-12, atol=0.0
                )
        np.testing.assert_array_equal(theta, original)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_dense_cancellation_retains_small_terms_in_any_factor_order(order):
    slopes = np.array([1e150, 1.0, -1e150])[list(order)]
    point = np.array([1e200, 2.0, 1e200])[list(order)]
    model = _model("mirt", [slopes], [0.0])
    probability, information, matrix = _decimal_curve(point, slopes, 0.0)
    theta = np.tile(point, (2, 1))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        np.testing.assert_allclose(model.probability(theta), probability, rtol=5e-12)
        np.testing.assert_allclose(model.information(theta), information, rtol=5e-12)
        np.testing.assert_allclose(
            model.item_information_matrix(theta, 0),
            np.broadcast_to(matrix, (2, 3, 3)),
            rtol=5e-12,
        )
        np.testing.assert_allclose(
            model.probability_pairs(theta, [0, 0]), probability, rtol=5e-12
        )


@pytest.mark.parametrize("kind", ["mirt", "bifactor"])
@pytest.mark.parametrize(
    "method,item_idx",
    [
        ("probability", None),
        ("probability", 3),
        ("probability_pairs", None),
        ("information", None),
        ("information", 3),
        ("item_information_matrix", 3),
        ("test_information_matrix", None),
    ],
)
def test_affine_recovery_and_query_buffers_are_bounded(
    kind, method, item_idx, monkeypatch
):
    slopes = np.array(
        [[1e200, -1e200], [1.5, 0.0], [0.0, 0.0], [1e16 + 2.0, -1e16], [0.5, 0.5]]
    )
    intercepts = np.array([1000.0, -1.5e308, 2.0, 0.0, -_MAX_FLOAT])
    model = _model(kind, slopes, intercepts)
    storage = np.full((78, 2), np.nan)
    storage[::2] = np.resize(
        [
            [1e200, 1e200],
            [1e308, 0.0],
            [np.nextafter(1.0, 2.0), 1.0],
            [_MAX_FLOAT, _MAX_FLOAT],
        ],
        (39, 2),
    )
    theta = storage[::2]
    original = theta.copy()
    theta.flags.writeable = False
    indices = np.resize([4, 3, 0, 1, 2], 78)[::2]
    indices.flags.writeable = False
    references = [
        [
            _decimal_curve(point, slope, intercept)
            for slope, intercept in zip(slopes, intercepts, strict=True)
        ]
        for point in theta
    ]
    field = 0 if method.startswith("probability") else 1
    if method in ("item_information_matrix", "test_information_matrix"):
        matrices = np.array([[entry[2] for entry in row] for row in references])
        with np.errstate(over="ignore"):
            expected = (
                matrices[:, item_idx] if item_idx is not None else matrices.sum(axis=1)
            )
    else:
        expected = np.array([[entry[field] for entry in row] for row in references])
        if method == "probability_pairs":
            expected = expected[np.arange(len(theta)), indices]
        elif item_idx is not None:
            expected = expected[:, item_idx]
    original_exact = _logistic._exact_affine_pairs
    original_probability = _logistic._logistic_probability
    sizes = []
    probability_sizes = []

    def bounded(points, coefficients, offsets):
        assert points.size <= 17
        assert coefficients.size <= 17
        sizes.append(points.size)
        return original_exact(points, coefficients, offsets)

    def bounded_probability(logits, *args):
        assert logits.size <= 17
        probability_sizes.append(logits.size)
        return original_probability(logits, *args)

    monkeypatch.setattr(_logistic, "_AFFINE_CHUNK_ELEMENTS", 17)
    monkeypatch.setattr(_logistic, "_INFORMATION_CHUNK_ELEMENTS", 17)
    monkeypatch.setattr(_logistic, "_exact_affine_pairs", bounded)
    monkeypatch.setattr(_logistic, "_logistic_probability", bounded_probability)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        if method == "probability_pairs":
            actual = model.probability_pairs(theta, indices)
        elif method == "test_information_matrix":
            actual = model.test_information_matrix(theta)
        else:
            actual = getattr(model, method)(theta, item_idx)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=0.0)
    np.testing.assert_array_equal(theta, original)
    assert sizes and len(sizes) > 1
    if method.startswith("probability"):
        assert sum(probability_sizes) == expected.size


@pytest.mark.parametrize("kind", ["mirt", "bifactor"])
def test_nonfinite_inputs_do_not_enter_exact_recovery(kind, monkeypatch):
    model = _model(kind, [[1.0, 2.0]], [0.0])
    theta = np.array([[np.inf, 0.0], [-np.inf, 0.0], [np.nan, 0.0]])

    def unexpected(*args):
        raise AssertionError("exact recovery received nonfinite inputs")

    monkeypatch.setattr(_logistic, "_exact_affine_pairs", unexpected)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for values in (
            model.probability(theta)[:, 0],
            model.probability(theta, 0),
            model.probability_pairs(theta, [0, 0, 0]),
        ):
            np.testing.assert_array_equal(values[:2], [1.0, 0.0])
            assert np.isnan(values[2])
        for values in (
            model.information(theta),
            model.item_information_matrix(theta, 0),
            model.test_information_matrix(theta),
        ):
            np.testing.assert_array_equal(values[:2], 0.0)
            assert np.isnan(values[2]).all()


def test_bifactor_recovery_ignores_unrelated_nonfinite_factor_columns():
    model = BifactorModel(3, [7, 20, 7]).set_parameters(
        general_loadings=np.array([1e200, 0.5, 1e16 + 2.0]),
        specific_loadings=np.array([-1e200, 0.5, -1e16]),
        intercepts=np.array([1000.0, 0.0, 0.0]),
    )
    theta = np.array([[1e200, 1e200, np.nan], [np.nextafter(1.0, 2.0), 1.0, np.inf]])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probabilities = model.probability(theta)
        pairs = model.probability_pairs(theta, [0, 2])
        for row, item in ((0, 0), (1, 2)):
            expected = _decimal_curve(
                theta[row, :2],
                [model.general_loadings[item], model.specific_loadings[item]],
                model.intercepts[item],
            )[0]
            np.testing.assert_allclose(probabilities[row, item], expected, rtol=5e-12)
            np.testing.assert_allclose(pairs[row], expected, rtol=5e-12)
            np.testing.assert_allclose(
                model.probability(theta[row : row + 1], item), expected, rtol=5e-12
            )
    assert np.isnan(probabilities[0, 1])
    assert probabilities[1, 1] == 1.0


@pytest.mark.parametrize("kind", ["mirt", "bifactor"])
def test_affine_probabilities_preserve_smallest_tails_across_fast_path_boundary(kind):
    slopes = [1.0, -2.0]
    model = _model(kind, [slopes], [0.0])
    z = np.array(
        [
            -1000.0,
            -745.0,
            -710.0,
            -700.1,
            -700.0,
            -699.9,
            -40.0,
            0.0,
            40.0,
            700.0,
            1000.0,
        ]
    )
    theta = np.column_stack((z, np.zeros_like(z)))
    expected = np.array([_decimal_curve(point, slopes, 0.0)[0] for point in theta])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        np.testing.assert_allclose(
            model.probability(theta)[:, 0], expected, rtol=5e-12, atol=0.0
        )
        np.testing.assert_allclose(
            model.probability(theta, 0), expected, rtol=5e-12, atol=0.0
        )
        np.testing.assert_allclose(
            model.probability_pairs(theta, np.zeros(len(theta), dtype=int)),
            expected,
            rtol=5e-12,
            atol=0.0,
        )
        for point, value in zip(theta, expected, strict=True):
            np.testing.assert_allclose(
                model.probability(point[None, :], 0), value, rtol=5e-12, atol=0.0
            )


def test_sparse_bifactor_pairs_do_not_copy_unused_item_bank():
    import tracemalloc

    model = BifactorModel(100_000, np.zeros(100_000, dtype=int))
    theta = np.array([[0.2, -0.1], [-0.4, 0.5]])
    indices = np.array([5, 99_999])
    expected = model.probability_pairs(theta, indices)
    tracemalloc.start()
    try:
        actual = model.probability_pairs(theta, indices)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(actual, expected)
    # A copied factor-index vector alone would require 800 KB.
    assert peak < 64_000
