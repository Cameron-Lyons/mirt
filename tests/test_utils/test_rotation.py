"""Tests for orthogonal and oblique factor rotations."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel
from mirt.utils.rotation import (
    apply_rotation_to_model,
    get_rotated_loadings,
    oblimin,
    promax,
    rotate_loadings,
    varimax,
)

SIMPLE_PATTERN = np.array(
    [
        [0.85, 0.02],
        [0.78, -0.04],
        [0.72, 0.06],
        [0.90, 0.00],
        [0.01, 0.82],
        [-0.05, 0.76],
        [0.04, 0.71],
        [0.00, 0.88],
    ]
)
TRUE_FACTOR_CORRELATION = np.array([[1.0, 0.55], [0.55, 1.0]])
CORRELATED_UNROTATED = SIMPLE_PATTERN @ np.linalg.cholesky(TRUE_FACTOR_CORRELATION)


def _orthomax_score(loadings: np.ndarray, gamma: float) -> float:
    squared = loadings**2
    return float(
        np.sum(squared**2)
        - (gamma / loadings.shape[0]) * np.sum(squared.sum(axis=0) ** 2)
    )


def _assert_oblique_invariants(
    unrotated: np.ndarray,
    rotated: np.ndarray,
    rotation: np.ndarray,
    correlation: np.ndarray,
) -> None:
    assert_allclose(rotated, unrotated @ rotation, atol=1e-10)
    assert_allclose(np.diag(correlation), 1.0, atol=1e-10)
    assert_allclose(correlation, correlation.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(correlation) > 0)
    assert_allclose(
        rotated @ correlation @ rotated.T,
        unrotated @ unrotated.T,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("method", "gamma"),
    [("varimax", 1.0), ("quartimax", 0.0), ("equamax", 1.5)],
)
def test_orthogonal_rotations_preserve_model_and_improve_objective(
    method: str, gamma: float
) -> None:
    rng = np.random.default_rng(20260803)
    loadings = rng.normal(size=(30, 3))

    rotated, rotation, correlation = rotate_loadings(
        loadings, method=method, normalize=False
    )

    assert correlation is None
    assert_allclose(rotated, loadings @ rotation, atol=1e-11)
    assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert_allclose(rotated @ rotated.T, loadings @ loadings.T, atol=1e-9)
    assert_allclose(np.sum(rotated**2, axis=1), np.sum(loadings**2, axis=1), atol=1e-10)
    assert _orthomax_score(rotated, gamma) >= _orthomax_score(loadings, gamma)


def test_equamax_no_longer_collapses_to_quartimax() -> None:
    rng = np.random.default_rng(20260803)
    loadings = rng.normal(size=(40, 3))

    quartimax, _, _ = rotate_loadings(loadings, method="quartimax", normalize=False)
    equamax, _, _ = rotate_loadings(loadings, method="equamax", normalize=False)

    assert np.max(np.abs(equamax - quartimax)) > 0.01


@pytest.mark.parametrize("method", ["oblimin", "geomin", "promax"])
def test_oblique_rotations_recover_correlated_simple_structure(method: str) -> None:
    rotated, rotation, correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method=method,
        normalize=False,
        max_iter=2000,
        tol=1e-9,
    )

    assert correlation is not None
    _assert_oblique_invariants(CORRELATED_UNROTATED, rotated, rotation, correlation)
    assert_allclose(rotated, SIMPLE_PATTERN, atol=0.02)
    assert_allclose(correlation, TRUE_FACTOR_CORRELATION, atol=0.02)


def test_oblimin_gamma_changes_the_solution_without_breaking_invariants() -> None:
    quartimin, quartimin_rotation, quartimin_correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method="oblimin",
        gamma=0.0,
        normalize=False,
    )
    biquartimin, biquartimin_rotation, biquartimin_correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method="oblimin",
        gamma=0.5,
        normalize=False,
    )

    assert quartimin_correlation is not None
    assert biquartimin_correlation is not None
    _assert_oblique_invariants(
        CORRELATED_UNROTATED,
        quartimin,
        quartimin_rotation,
        quartimin_correlation,
    )
    _assert_oblique_invariants(
        CORRELATED_UNROTATED,
        biquartimin,
        biquartimin_rotation,
        biquartimin_correlation,
    )
    assert not np.allclose(quartimin, biquartimin)


def test_geomin_epsilon_changes_the_solution() -> None:
    small, _, small_correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method="geomin",
        geomin_epsilon=0.001,
        normalize=False,
    )
    large, _, large_correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method="geomin",
        geomin_epsilon=0.1,
        normalize=False,
    )

    assert small_correlation is not None
    assert large_correlation is not None
    assert not np.allclose(small, large)


@pytest.mark.parametrize(
    "method", ["varimax", "quartimax", "equamax", "oblimin", "geomin", "promax"]
)
def test_kaiser_normalization_keeps_public_rotation_contract(method: str) -> None:
    loadings = CORRELATED_UNROTATED.copy()
    loadings[0] *= 2.5
    loadings[1] *= 0.4

    rotated, rotation, correlation = rotate_loadings(
        loadings, method=method, normalize=True
    )

    assert_allclose(rotated, loadings @ rotation, atol=1e-9)
    if correlation is None:
        assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-9)
    else:
        _assert_oblique_invariants(loadings, rotated, rotation, correlation)


def test_zero_loading_rows_remain_finite_with_kaiser_normalization() -> None:
    loadings = np.vstack([CORRELATED_UNROTATED, np.zeros((1, 2))])

    for method in ("varimax", "oblimin", "geomin", "promax"):
        rotated, rotation, correlation = rotate_loadings(loadings, method=method)
        assert np.all(np.isfinite(rotated))
        assert np.all(np.isfinite(rotation))
        assert_allclose(rotated[-1], 0.0)
        if correlation is not None:
            assert np.all(np.isfinite(correlation))


@pytest.mark.parametrize("scale", [1e-300, 1e300])
@pytest.mark.parametrize(
    "method", ["varimax", "quartimax", "equamax", "oblimin", "geomin", "promax"]
)
def test_kaiser_normalization_is_stable_across_finite_scales(
    method: str, scale: float
) -> None:
    expected, expected_rotation, expected_correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method=method,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rotated, rotation, correlation = rotate_loadings(
            CORRELATED_UNROTATED * scale,
            method=method,
        )

    assert np.all(np.isfinite(rotated))
    assert np.all(np.isfinite(rotation))
    assert_allclose(rotated / scale, expected, atol=1e-10)
    assert_allclose(rotation, expected_rotation, atol=1e-10)
    if expected_correlation is None:
        assert correlation is None
    else:
        assert_allclose(correlation, expected_correlation, atol=1e-10)


@pytest.mark.parametrize(
    "loadings",
    [
        np.zeros((6, 2)),
        np.column_stack((np.linspace(-1.0, 1.0, 6), np.zeros(6))),
        np.column_stack((np.linspace(-1.0, 1.0, 6),) * 2),
    ],
)
def test_promax_falls_back_safely_for_rank_deficient_loadings(
    loadings: np.ndarray,
) -> None:
    rotated, rotation, correlation = rotate_loadings(loadings, method="promax")

    assert correlation is not None
    _assert_oblique_invariants(loadings, rotated, rotation, correlation)
    assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-10)
    assert_allclose(correlation, np.eye(2), atol=1e-10)


def test_none_and_single_factor_are_noops() -> None:
    loadings = np.array([[0.2], [0.5], [0.8]])

    single, single_rotation, single_correlation = rotate_loadings(
        loadings, method="promax"
    )
    none, none_rotation, none_correlation = rotate_loadings(
        CORRELATED_UNROTATED, method="NONE"
    )

    assert_allclose(single, loadings)
    assert_allclose(single_rotation, np.eye(1))
    assert single_correlation is None
    assert_allclose(none, CORRELATED_UNROTATED)
    assert_allclose(none_rotation, np.eye(2))
    assert none_correlation is None


def test_convenience_functions_match_general_dispatch() -> None:
    expected_varimax, _, _ = rotate_loadings(CORRELATED_UNROTATED, method="varimax")
    expected_promax, _, _ = rotate_loadings(CORRELATED_UNROTATED, method="promax")
    expected_oblimin, _, _ = rotate_loadings(CORRELATED_UNROTATED, method="oblimin")

    assert_allclose(varimax(CORRELATED_UNROTATED), expected_varimax)
    assert_allclose(promax(CORRELATED_UNROTATED), expected_promax)
    assert_allclose(oblimin(CORRELATED_UNROTATED), expected_oblimin)


@pytest.mark.parametrize(
    ("loadings", "kwargs", "message"),
    [
        (np.array([0.2, 0.4]), {}, "2D matrix"),
        (np.empty((0, 2)), {}, "at least one item"),
        (np.ones((2, 3)), {}, "at least as many items"),
        (np.array([[0.2, np.nan], [0.3, 0.4]]), {}, "finite values"),
        (np.ones((3, 2)), {"method": "unknown"}, "Unknown rotation"),
        (np.ones((3, 2)), {"max_iter": 0}, "at least 1"),
        (np.ones((3, 2)), {"tol": 0.0}, "positive value"),
        (np.ones((3, 2)), {"normalize": 1}, "boolean"),
        (
            np.ones((3, 2)),
            {"method": "promax", "kappa": 1.0},
            "greater than 1",
        ),
        (
            np.ones((3, 2)),
            {"method": "oblimin", "gamma": np.inf},
            "finite value",
        ),
        (
            np.ones((3, 2)),
            {"method": "geomin", "geomin_epsilon": 0.0},
            "positive value",
        ),
    ],
)
def test_rotate_loadings_validates_inputs(
    loadings: np.ndarray, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rotate_loadings(loadings, **kwargs)


def test_standard_multidimensional_model_loadings_can_be_read_and_applied() -> None:
    model = MultidimensionalModel(8, n_factors=2)
    model.set_parameters(
        slopes=CORRELATED_UNROTATED,
        intercepts=np.zeros(8),
    )

    rotated, rotation, correlation = rotate_loadings(
        model.slopes, method="oblimin", normalize=False
    )
    extracted, extracted_correlation = get_rotated_loadings(
        model, method="oblimin", normalize=False
    )
    apply_rotation_to_model(model, rotation, correlation)

    assert_allclose(extracted, rotated)
    assert_allclose(extracted_correlation, correlation)
    assert_allclose(model.slopes, rotated)
    assert_allclose(model._rotation_matrix, rotation)
    assert_allclose(model._factor_correlation, correlation)


def test_get_rotated_loadings_supports_bifactor_loading_matrix() -> None:
    model = BifactorModel(8, specific_factors=[0, 0, 0, 0, 1, 1, 1, 1])

    rotated, correlation = get_rotated_loadings(model, method="varimax")

    assert rotated.shape == (8, 3)
    assert correlation is None
    with pytest.raises(ValueError, match="structured bifactor"):
        apply_rotation_to_model(model, np.eye(3))


def test_apply_rotation_rejects_confirmatory_and_inconsistent_inputs() -> None:
    confirmatory = MultidimensionalModel(
        4,
        n_factors=2,
        model_type="confirmatory",
        loading_pattern=np.ones((4, 2)),
    )
    exploratory = MultidimensionalModel(4, n_factors=2)

    with pytest.raises(ValueError, match="exploratory"):
        apply_rotation_to_model(confirmatory, np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        apply_rotation_to_model(exploratory, np.eye(3))
    with pytest.raises(ValueError, match="nonsingular"):
        apply_rotation_to_model(exploratory, np.ones((2, 2)))
    with pytest.raises(ValueError, match="well-conditioned"):
        apply_rotation_to_model(exploratory, np.diag([1.0, 1e-9]))
    with pytest.raises(ValueError, match="inconsistent"):
        apply_rotation_to_model(
            exploratory,
            np.array([[1.0, 0.2], [0.0, 1.0]]),
            np.eye(2),
        )


def test_models_without_factor_loadings_are_rejected() -> None:
    model = TwoParameterLogistic(4)
    with pytest.raises(ValueError, match="does not have loadings"):
        get_rotated_loadings(model)
    with pytest.raises(ValueError, match="does not have freely rotatable"):
        apply_rotation_to_model(model, np.eye(1))
