"""Tests for orthogonal and oblique factor rotations."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

import mirt
from mirt.equating import target_rotation
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel, standardized_loadings
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)
from mirt.results.fit_result import FitResult
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

# A three-factor simple structure mixed by a random orthogonal matrix.
THREE_FACTOR_UNROTATED = np.array(
    [
        [-0.422327454696697, 0.033005391476899, 0.315778555562783],
        [-0.565002307417890, -0.128956823038007, 0.448626060469185],
        [-0.391931140922198, -0.038679104392955, 0.284638910121828],
        [-0.601222764751687, 0.026422344955997, 0.323848851322985],
        [0.350691409632322, -0.716994427812227, 0.120768370910261],
        [0.351047973418208, -0.448535059335859, 0.299444770163117],
        [0.278035630235763, -0.410334802870209, 0.307950786265218],
        [0.426665979112948, -0.462688447642095, 0.574547923636424],
        [0.118846767388107, 0.265108635588310, 0.516948219095298],
        [0.365792175504690, 0.409424059396781, 0.504023350655547],
        [-0.061195985612479, 0.287831255363941, 0.503146493230576],
        [0.379792409496607, 0.343266504368837, 0.379928160743180],
    ]
)
# Unspecified (NaN) entries of a partially specified simple-structure target.
THREE_FACTOR_TARGET = np.full((12, 3), np.nan)
for _factor in range(3):
    THREE_FACTOR_TARGET[4 * _factor : 4 * _factor + 4, np.arange(3) != _factor] = 0.0
# GPArotation 2026.8.2 criterion values of R mirt's Rotate(A, method) with
# eps = 1e-12, which calls GPArotation with normalize = FALSE.
R_ROTATE_CRITERIA = {
    "oblimin": 0.068228394743,
    "varimax": -0.373970749715,
    "quartimax": -0.571531581379,
    "geominT": 0.646488037748,
    "geominQ": 0.615732612265,
    "bifactorT": 0.029679248904,
    "bifactorQ": 0.018973938476,
    "targetT": 0.341932956024,
    "targetQ": 0.295592470228,
    "pstQ": 0.295592470228,
}
# Slopes of a two-factor exploratory model, rotated away from simple structure.
_ANGLE = np.deg2rad(35.0)
EXPLORATORY_SLOPES = np.column_stack(
    (
        [1.2, 1.5, 0.9, 1.1, 0.3, 0.2, 0.6, 1.4, 0.8, 0.1],
        [0.2, 0.4, 0.1, 0.5, 1.3, 1.6, 1.0, 0.9, 0.7, 1.2],
    )
) @ np.array([[np.cos(_ANGLE), -np.sin(_ANGLE)], [np.sin(_ANGLE), np.cos(_ANGLE)]])
# R mirt 1.48 summary(mod, rotate = "oblimin", eps = 1e-12)$rotF and $fcor
# for 2PL and four-category GPCM items with these slopes.
R_SUMMARY_OBLIMIN = {
    "2PL": (
        [
            [0.6191459290, -0.0979287035],
            [0.6917443181, -0.0417940961],
            [0.5083291791, -0.1073330873],
            [0.5429670235, 0.0716641710],
            [0.0511211927, 0.5921454810],
            [-0.0219304402, 0.6973810136],
            [0.2394802569, 0.4151070194],
            [0.5870362262, 0.1979690028],
            [0.3821480491, 0.2328015910],
            [-0.0450646118, 0.5965210573],
        ],
        0.4517852572,
    ),
    "GPCM": (
        [
            [0.8961779976, -0.1196580525],
            [0.9130440661, -0.0312719821],
            [0.8242653612, -0.1543163636],
            [0.7848865391, 0.1270866632],
            [0.0601079503, 0.8426335187],
            [-0.0408208689, 0.9232813066],
            [0.3433308377, 0.6300810456],
            [0.7522047541, 0.2800541327],
            [0.5766622760, 0.3765035628],
            [-0.0770518437, 0.8789507812],
        ],
        0.4409081110,
    ),
}


def _fit_result(model: BaseItemModel, **kwargs: object) -> FitResult:
    return FitResult(
        model=model,
        log_likelihood=0.0,
        n_iterations=0,
        converged=True,
        standard_errors={},
        aic=0.0,
        bic=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


def _criterion(method: str, loadings: np.ndarray) -> float:
    """Return GPArotation's criterion value, written out independently."""
    squared = loadings**2
    if method in {"varimax", "quartimax"}:
        if method == "quartimax":
            return -float(np.sum(squared**2)) / 4
        centered = squared - squared.mean(axis=0)
        return -float(np.sum(centered**2)) / 4
    if method == "oblimin":
        return float(np.sum(squared * (squared.sum(1, keepdims=True) - squared))) / 4
    if method.startswith("geomin"):
        return float(np.sum(np.exp(np.mean(np.log(squared + 0.01), axis=1))))
    if method.startswith("bifactor"):
        group = squared[:, 1:]
        return float(np.sum(group.sum(1) ** 2) - np.sum(group**2))
    specified = ~np.isnan(THREE_FACTOR_TARGET)
    return float(np.sum((loadings - THREE_FACTOR_TARGET)[specified] ** 2))


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
        normalize=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rotated, rotation, correlation = rotate_loadings(
            CORRELATED_UNROTATED * scale,
            method=method,
            normalize=True,
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
    result = _fit_result(model)

    standardized = standardized_loadings(model.slopes)[0]
    rotated, rotation, correlation = rotate_loadings(standardized, method="oblimin")
    extracted, extracted_correlation = get_rotated_loadings(
        model, method="oblimin", standardized=True
    )
    slopes, _ = get_rotated_loadings(model, method="oblimin")
    apply_rotation_to_model(result, rotation, correlation)

    assert_allclose(extracted, rotated)
    assert_allclose(extracted_correlation, correlation)
    assert_allclose(slopes, CORRELATED_UNROTATED @ rotation)
    assert_allclose(model.slopes, slopes)
    assert_allclose(result.latent_covariance, correlation, atol=1e-12)
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


def _kaiser_rows(loadings: np.ndarray) -> np.ndarray:
    return loadings / np.linalg.norm(loadings, axis=1, keepdims=True)


def _marginal_log_likelihood(result: FitResult, responses: np.ndarray) -> float:
    quadrature = GaussHermiteQuadrature(
        n_points=61,
        n_dimensions=result.model.n_factors,
        mean=result.latent_mean,
        cov=result.latent_covariance,
    )
    joint = result.model.log_likelihood_batch(responses, quadrature.nodes) + np.log(
        quadrature.weights
    )
    return float(np.sum(logsumexp(joint, axis=1)))


def _exploratory_model(family: str) -> BaseItemModel:
    slopes = EXPLORATORY_SLOPES
    offsets = np.linspace(-0.8, 0.8, 10)
    model: BaseItemModel
    if family == "MIRT":
        model = MultidimensionalModel(10, n_factors=2)
        model.set_parameters(slopes=slopes, intercepts=offsets)
    elif family == "2PL":
        model = TwoParameterLogistic(10, n_factors=2)
        model.set_parameters(discrimination=slopes, difficulty=offsets)
    elif family == "GRM":
        model = GradedResponseModel(10, n_categories=4, n_factors=2)
        model.set_parameters(
            discrimination=slopes,
            thresholds=offsets[:, None] + np.array([-1.0, 0.1, 1.2]),
        )
    elif family == "GPCM":
        model = GeneralizedPartialCredit(10, n_categories=4, n_factors=2)
        model.set_parameters(
            discrimination=slopes,
            steps=offsets[:, None] + np.array([-0.9, 0.3, 0.8]),
        )
    else:
        model = NominalResponseModel(10, n_categories=4, n_factors=2)
        category_slopes = (
            np.array([0.0, 0.4, 1.0, 1.5])[None, :, None] * slopes[:, None, :]
        )
        category_slopes[:, 1, 0] += 0.3
        model.set_parameters(
            slopes=category_slopes,
            intercepts=np.column_stack(
                (np.zeros(10), offsets, offsets + 0.4, offsets - 0.5)
            ),
        )
    return model


def test_equamax_reaches_the_gradient_projection_optimum_with_three_factors() -> None:
    # GPArotation's cfT(kappa = 3 / (2 * 12)) criterion values; the SVD
    # iteration used to stop at 2.662 with Kaiser normalization.
    for normalize, expected in ((True, 4.67265156052208), (False, 1.10088541584650)):
        rotated, rotation, correlation = rotate_loadings(
            THREE_FACTOR_UNROTATED,
            method="equamax",
            normalize=normalize,
            tol=1e-10,
        )
        scored = _kaiser_rows(rotated) if normalize else rotated

        assert correlation is None
        assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert _orthomax_score(scored, 1.5) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("n_factors", [3, 4, 5])
def test_equamax_agrees_with_its_best_random_start(n_factors: int) -> None:
    rng = np.random.default_rng(100 * n_factors)
    n_items = 4 * n_factors
    pattern = np.zeros((n_items, n_factors))
    pattern[np.arange(n_items), np.arange(n_items) % n_factors] = rng.uniform(
        0.4, 0.9, n_items
    )
    pattern += rng.normal(0.0, 0.15, pattern.shape)
    mixing, _ = np.linalg.qr(rng.normal(size=(n_factors, n_factors)))
    loadings = pattern @ mixing

    default, _, _ = rotate_loadings(loadings, method="equamax", normalize=False)
    best, _, _ = rotate_loadings(
        loadings, method="equamax", normalize=False, random_starts=20, seed=1
    )

    assert _orthomax_score(default, n_factors / 2) == pytest.approx(
        _orthomax_score(best, n_factors / 2), abs=1e-8
    )


@pytest.mark.parametrize("method", sorted(R_ROTATE_CRITERIA))
def test_rotations_reach_r_mirt_criterion_values(method: str) -> None:
    kwargs: dict[str, object] = {}
    if method.startswith("target"):
        kwargs["target"] = THREE_FACTOR_TARGET
    elif method.startswith("pst"):
        specified = ~np.isnan(THREE_FACTOR_TARGET)
        kwargs["target"] = np.where(specified, THREE_FACTOR_TARGET, 0.0)
        kwargs["target_weights"] = specified.astype(float)

    rotated, rotation, correlation = rotate_loadings(
        THREE_FACTOR_UNROTATED, method=method, tol=1e-10, max_iter=20000, **kwargs
    )

    assert _criterion(method, rotated) == pytest.approx(
        R_ROTATE_CRITERIA[method], abs=1e-9
    )
    assert_allclose(rotated, THREE_FACTOR_UNROTATED @ rotation, atol=1e-12)
    # R mirt reflects each column so that its largest loading is positive.
    largest = rotated[np.argmax(np.abs(rotated), axis=0), np.arange(3)]
    assert np.all(largest > 0.0)
    if correlation is None:
        assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    else:
        _assert_oblique_invariants(
            THREE_FACTOR_UNROTATED, rotated, rotation, correlation
        )


def test_promax_forms_its_target_from_denormalized_varimax_loadings() -> None:
    # stats::promax(A, m = 4) with a converged Kaiser-normalized varimax,
    # its third column reflected as R mirt's Rotate() does.
    expected = np.array(
        [
            [1.0, -0.128297719001, 0.225614856670],
            [-0.128297719001, 1.0, 0.108490597982],
            [0.225614856670, 0.108490597982, 1.0],
        ]
    )

    _, _, correlation = rotate_loadings(
        THREE_FACTOR_UNROTATED, method="promax", tol=1e-12
    )

    assert correlation is not None
    assert_allclose(correlation, expected, atol=1e-9)


def test_default_normalization_follows_r_mirt() -> None:
    for method in ("varimax", "oblimin", "geominQ"):
        default = rotate_loadings(THREE_FACTOR_UNROTATED, method=method)
        unnormalized = rotate_loadings(
            THREE_FACTOR_UNROTATED, method=method, normalize=False
        )
        assert_allclose(default[0], unnormalized[0])
    promax_default = rotate_loadings(THREE_FACTOR_UNROTATED, method="promax")
    promax_kaiser = rotate_loadings(
        THREE_FACTOR_UNROTATED, method="promax", normalize=True
    )
    assert_allclose(promax_default[0], promax_kaiser[0])


def test_fully_specified_target_rotation_is_orthogonal_procrustes() -> None:
    rotated, _, correlation = rotate_loadings(
        CORRELATED_UNROTATED,
        method="targetT",
        target=SIMPLE_PATTERN,
        tol=1e-12,
        reflect=False,
    )
    expected, _ = target_rotation(CORRELATED_UNROTATED, SIMPLE_PATTERN)

    assert correlation is None
    assert_allclose(rotated, expected, atol=1e-9)


def test_partially_specified_target_matches_target_with_missing_entries() -> None:
    specified = ~np.isnan(THREE_FACTOR_TARGET)
    for orthogonal in (True, False):
        suffix = "T" if orthogonal else "Q"
        target_result = rotate_loadings(
            THREE_FACTOR_UNROTATED,
            method=f"target{suffix}",
            target=THREE_FACTOR_TARGET,
        )
        pst_result = rotate_loadings(
            THREE_FACTOR_UNROTATED,
            method=f"pst{suffix}",
            target=np.where(specified, THREE_FACTOR_TARGET, 0.0),
            target_weights=specified.astype(float),
        )
        assert_allclose(pst_result[0], target_result[0], atol=1e-12)


def test_reflection_only_flips_column_signs() -> None:
    for method in ("varimax", "oblimin", "bifactorQ"):
        reflected, rotation, correlation = rotate_loadings(
            THREE_FACTOR_UNROTATED, method=method
        )
        raw, raw_rotation, raw_correlation = rotate_loadings(
            THREE_FACTOR_UNROTATED, method=method, reflect=False
        )
        signs = np.sign(np.sum(reflected * raw, axis=0))

        assert_allclose(reflected, raw * signs, atol=1e-12)
        assert_allclose(rotation, raw_rotation * signs, atol=1e-12)
        if correlation is not None:
            assert raw_correlation is not None
            assert_allclose(
                correlation, raw_correlation * np.outer(signs, signs), atol=1e-12
            )


def test_random_starts_are_reproducible_and_never_worse() -> None:
    first = rotate_loadings(
        THREE_FACTOR_UNROTATED, method="geominQ", random_starts=5, seed=3
    )
    second = rotate_loadings(
        THREE_FACTOR_UNROTATED, method="geominQ", random_starts=5, seed=3
    )
    identity_only = rotate_loadings(THREE_FACTOR_UNROTATED, method="geominQ")

    assert_allclose(first[0], second[0])
    assert (
        _criterion("geominQ", first[0])
        <= _criterion("geominQ", identity_only[0]) + 1e-9
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "targetT"}, "requires a target"),
        ({"method": "pstQ", "target": np.zeros((12, 3))}, "requires target_weights"),
        ({"method": "targetQ", "target": np.zeros((3, 3))}, "shape"),
        ({"method": "targetT", "target": np.full((12, 3), np.nan)}, "at least one"),
        ({"method": "varimax", "target": np.zeros((12, 3))}, "apply only"),
        (
            {
                "method": "pstT",
                "target": np.zeros((12, 3)),
                "target_weights": -np.ones((12, 3)),
            },
            "non-negative",
        ),
        ({"random_starts": -1}, "random_starts"),
        ({"seed": 1.5}, "seed"),
        ({"reflect": 1}, "reflect"),
    ],
)
def test_new_rotation_options_are_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(MirtValidationError, match=message):
        rotate_loadings(THREE_FACTOR_UNROTATED, **kwargs)


@pytest.mark.parametrize("family", ["2PL", "GRM", "GPCM", "NRM"])
def test_get_rotated_loadings_matches_r_mirt_summary(family: str) -> None:
    model = _exploratory_model(family)
    if family == "NRM":
        # Keep R's rank-one nominal slopes ak * a.
        model.set_parameters(
            slopes=np.array([0.0, 0.7, 2.2, 3.0])[None, :, None]
            * EXPLORATORY_SLOPES[:, None, :]
        )
    expected, expected_correlation = R_SUMMARY_OBLIMIN[
        "GPCM" if family in {"GPCM", "NRM"} else "2PL"
    ]

    loadings, correlation = get_rotated_loadings(model, standardized=True, tol=1e-10)
    slopes, slope_correlation = get_rotated_loadings(model, tol=1e-10)

    assert correlation is not None and slope_correlation is not None
    assert_allclose(loadings, expected, atol=1e-8)
    assert correlation[0, 1] == pytest.approx(expected_correlation, abs=1e-8)
    assert_allclose(slope_correlation, correlation)
    # Standardizing scales rows, so the slopes rotate by the same matrix.
    unrotated, _ = get_rotated_loadings(model, method="none")
    _, rotation, _ = rotate_loadings(
        get_rotated_loadings(model, method="none", standardized=True)[0],
        method="oblimin",
        tol=1e-10,
    )
    assert_allclose(slopes, unrotated @ rotation, atol=1e-10)


@pytest.mark.parametrize("family", ["MIRT", "2PL", "GRM", "GPCM", "NRM"])
@pytest.mark.parametrize(
    "method",
    [
        "varimax",
        "quartimax",
        "equamax",
        "geominT",
        "oblimin",
        "quartimin",
        "promax",
        "geominQ",
    ],
)
def test_applied_rotations_keep_the_model_equivalent(family: str, method: str) -> None:
    model = _exploratory_model(family)
    original = model.copy()
    result = _fit_result(model)
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(60, 2))
    responses = original.simulate(theta, seed=3)
    marginal = _marginal_log_likelihood(_fit_result(original), responses)

    apply_rotation_to_model(result, method)

    rotation = model._rotation_matrix
    inverse = np.linalg.inv(rotation)
    if result.latent_covariance is None:
        assert_allclose(inverse @ inverse.T, np.eye(2), atol=1e-10)
    else:
        assert_allclose(result.latent_covariance, inverse @ inverse.T, atol=1e-12)
        assert_allclose(np.diag(result.latent_covariance), 1.0, atol=1e-12)
    # The rotated factors are inv(T) @ theta, which leaves every response
    # probability, and hence the marginal likelihood, unchanged.
    assert_allclose(
        model.log_likelihood_batch(responses, theta @ inverse.T),
        original.log_likelihood_batch(responses, theta),
        atol=1e-9,
    )
    assert _marginal_log_likelihood(result, responses) == pytest.approx(
        marginal, abs=1e-3
    )


def test_rotation_helpers_accept_an_exploratory_fit_mirt_result() -> None:
    rng = np.random.default_rng(0)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], 500)
    responses = mirt.simdata(
        model="2PL",
        theta=theta,
        discrimination=2.0 * SIMPLE_PATTERN,
        difficulty=np.linspace(-1.0, 1.0, 8),
        seed=2,
    )
    result = mirt.fit_mirt(responses, model="2PL", n_factors=2)
    log_likelihood = _marginal_log_likelihood(result, responses)

    model_loadings, model_correlation = get_rotated_loadings(result.model)
    loadings, correlation = get_rotated_loadings(result)
    apply_rotation_to_model(result, "oblimin")

    assert isinstance(result.model, TwoParameterLogistic)
    assert correlation is not None and model_correlation is not None
    assert_allclose(loadings, model_loadings)
    assert_allclose(result.model.discrimination, loadings, atol=1e-12)
    assert result.factor_correlation is not None
    assert_allclose(result.factor_correlation, correlation, atol=1e-12)
    assert correlation[0, 1] > 0.2
    assert _marginal_log_likelihood(result, responses) == pytest.approx(
        log_likelihood, abs=1e-3
    )
    # Rotated parameters have no standard errors, as in R mirt.
    assert np.all(np.isnan(result.standard_errors["discrimination"]))
    assert np.all(np.isnan(result.standard_errors["difficulty"]))
    assert result.vcov is not None and np.all(np.isnan(result.vcov))


def test_rotation_of_a_fit_with_a_latent_population_keeps_the_model() -> None:
    model = _exploratory_model("2PL")
    original = model.copy()
    covariance = np.array([[1.5, 0.3], [0.3, 0.8]])
    mean = np.array([0.2, -0.1])
    result = _fit_result(model, latent_covariance=covariance, latent_mean=mean)
    responses = original.simulate(np.random.default_rng(1).normal(size=(80, 2)), 4)
    marginal = _marginal_log_likelihood(
        _fit_result(original, latent_covariance=covariance, latent_mean=mean),
        responses,
    )

    apply_rotation_to_model(result, "geominQ")

    inverse = np.linalg.inv(model._rotation_matrix)
    assert result.latent_covariance is not None and result.latent_mean is not None
    assert_allclose(
        result.latent_covariance, inverse @ covariance @ inverse.T, atol=1e-12
    )
    assert_allclose(np.diag(result.latent_covariance), 1.0, atol=1e-12)
    assert_allclose(result.latent_mean, inverse @ mean, atol=1e-12)
    assert _marginal_log_likelihood(result, responses) == pytest.approx(
        marginal, abs=1e-3
    )


def test_oblique_rotation_of_a_bare_model_is_refused() -> None:
    for family in ("MIRT", "2PL"):
        model = _exploratory_model(family)
        before = model.parameters
        _, rotation, correlation = rotate_loadings(EXPLORATORY_SLOPES, method="oblimin")

        with pytest.raises(MirtValidationError, match="FitResult"):
            apply_rotation_to_model(model, rotation, correlation)
        for name, values in before.items():
            assert_allclose(model.parameters[name], values)

        apply_rotation_to_model(model, "varimax")
        assert model._factor_correlation is None


def test_rotation_keeps_slope_sum_intercepts_or_refuses() -> None:
    model = TwoParameterLogistic(3, n_factors=2)
    model.set_parameters(
        discrimination=np.array([[1.0, 0.0], [0.4, 0.9], [0.8, 0.5]]),
        difficulty=np.array([0.5, -0.3, 0.2]),
    )
    intercepts = model.discrimination.sum(axis=1) * model.difficulty
    angle = np.pi / 4
    # Rotates the first item's slopes to (cos 45, -sin 45), which sum to zero.
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    with pytest.raises(MirtValidationError, match="sum to zero"):
        apply_rotation_to_model(model, rotation)
    assert_allclose(model.discrimination.sum(axis=1) * model.difficulty, intercepts)

    apply_rotation_to_model(model, rotation.T)
    assert_allclose(model.discrimination.sum(axis=1) * model.difficulty, intercepts)


def test_restricted_or_fixed_slopes_are_not_rotated() -> None:
    model = _exploratory_model("2PL")
    mask = model.free_parameter_masks["discrimination"].copy()
    mask[0, 1] = False
    model.set_free_parameter_masks({"discrimination": mask})

    with pytest.raises(ValueError, match="fixed or restricted"):
        apply_rotation_to_model(model, "varimax")
    with pytest.raises(MirtValidationError, match="method name"):
        apply_rotation_to_model(_exploratory_model("2PL"), np.eye(2), tol=1e-8)


def test_get_rotated_loadings_defaults_to_oblimin() -> None:
    model = _exploratory_model("MIRT")

    default = get_rotated_loadings(model)
    oblimin_result = get_rotated_loadings(model, "oblimin")

    assert default[1] is not None
    assert_allclose(default[0], oblimin_result[0])
