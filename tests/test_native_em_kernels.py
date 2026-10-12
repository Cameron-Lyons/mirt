"""Native EM kernels against small NumPy references of the same algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from mirt._rust_backend import RUST_AVAILABLE
from mirt.estimation.quadrature import GaussHermiteQuadrature

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")

EPSILON = 1e-10


def _responses(n_persons: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_persons)
    discrimination = rng.uniform(0.7, 1.8, n_items)
    difficulty = np.linspace(-1.5, 1.5, n_items)
    probability = expit(discrimination * (theta[:, None] - difficulty))
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    responses[rng.random(responses.shape) < 0.08] = -1
    responses[0] = -1
    return responses


def test_em_iteration_3pl_frequencies_match_expanded_rows() -> None:
    from mirt.backends.rust.estimation import em_iteration_3pl

    responses = _responses(200, 5, seed=12)
    frequencies = np.random.default_rng(3).integers(1, 4, len(responses))
    expanded = np.repeat(responses, frequencies, axis=0)
    quadrature = GaussHermiteQuadrature(n_points=21)
    args = (
        quadrature.nodes.ravel(),
        quadrature.weights.ravel(),
        np.linspace(0.8, 1.6, 5),
        np.linspace(-1.0, 1.0, 5),
        np.full(5, 0.15),
    )

    weighted = em_iteration_3pl(responses, *args, frequencies=frequencies.astype(float))
    repeated = em_iteration_3pl(expanded, *args)
    assert weighted is not None and repeated is not None

    for weighted_values, repeated_values in zip(
        weighted[:3], repeated[:3], strict=True
    ):
        np.testing.assert_allclose(weighted_values, repeated_values, rtol=1e-10)
    np.testing.assert_allclose(
        np.repeat(weighted[3], frequencies, axis=0), repeated[3], rtol=1e-13
    )
    assert weighted[4] == pytest.approx(repeated[4], rel=1e-12)


def test_em_iteration_3pl_ignores_deprecated_damping() -> None:
    from mirt.backends.rust.estimation import em_iteration_3pl

    responses = _responses(150, 4, seed=8)
    quadrature = GaussHermiteQuadrature(n_points=15)
    args = (
        responses,
        quadrature.nodes.ravel(),
        quadrature.weights.ravel(),
        np.ones(4),
        np.zeros(4),
        np.full(4, 0.1),
    )

    expected = em_iteration_3pl(*args)
    with pytest.warns(DeprecationWarning, match="damping_ab and damping_c"):
        damped = em_iteration_3pl(*args, damping_ab=0.5, damping_c=0.3)

    assert expected is not None and damped is not None
    for value, reference in zip(damped, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


def _projected_3pl_scores(
    params: np.ndarray, r_k: np.ndarray, n_k: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Expected-count 3PL scores with the components that push out of the box zeroed."""
    a, b, c = params
    keep = n_k >= EPSILON
    r_k, n_k, points = r_k[keep], n_k[keep], points[keep]
    p_star = expit(a * (points - b))
    p = np.clip(c + (1 - c) * p_star, EPSILON, 1 - EPSILON)
    slope = (1 - c) * p_star * (1 - p_star)
    gradient = np.stack([slope * (points - b), -slope * a, 1 - p_star])
    score = gradient @ ((r_k - n_k * p) / np.maximum(p * (1 - p), EPSILON))
    lower, upper = np.array([0.1, -6.0, 0.0]), np.array([5.0, 6.0, 0.35])
    score[(params <= lower) & (score < 0)] = 0.0
    score[(params >= upper) & (score > 0)] = 0.0
    return score


def test_em_iteration_3pl_m_step_reaches_the_bounded_optimum() -> None:
    from mirt.backends.rust.estimation import em_iteration_3pl

    # Generating guessing values of 0.5 and zero put many optima on the guessing
    # bounds, and the first discrimination lies above its bound. The former
    # clipped joint Newton step stalled on such items, and projected Newton
    # without an active band jammed with guessing just inside 0.35.
    rng = np.random.default_rng(39)
    n_items = 8
    n_persons = int(rng.integers(100, 3000))
    discrimination = rng.uniform(0.3, 4.0, n_items)
    difficulty = rng.normal(0.0, 1.5, n_items)
    guessing = rng.choice([0.0, 0.05, 0.3, 0.5], n_items)
    discrimination[0] = 6.0
    theta = rng.standard_normal(n_persons)
    probability = guessing + (1 - guessing) * expit(
        discrimination * (theta[:, None] - difficulty)
    )
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    quadrature = GaussHermiteQuadrature(n_points=21)
    points = quadrature.nodes.ravel()

    result = em_iteration_3pl(
        responses,
        points,
        quadrature.weights.ravel(),
        np.ones(n_items),
        np.zeros(n_items),
        np.full(n_items, 0.2),
        max_m_iter=500,
        m_tol=1e-12,
    )

    assert result is not None
    a, b, c, posterior, _ = result
    r_k = (responses == 1).T.astype(np.float64) @ posterior
    n_k = (responses >= 0).T.astype(np.float64) @ posterior
    for item, params in enumerate(np.column_stack([a, b, c])):
        score = _projected_3pl_scores(params, r_k[item], n_k[item], points)
        np.testing.assert_allclose(score, 0.0, atol=1e-6, err_msg=f"item {item}")
    assert np.any(c == 0.0) and np.any(c == 0.35)


@pytest.mark.parametrize("grm", [True, False])
def test_polytomous_m_step_accepts_non_contiguous_posteriors(grm: bool) -> None:
    from mirt import mirt_rs

    rng = np.random.default_rng(4)
    categories = np.array([2, 4, 3, 5], dtype=np.int32)
    responses = np.column_stack([rng.integers(-1, k, 150) for k in categories]).astype(
        np.int32
    )
    points = np.linspace(-3.0, 3.0, 9)
    posterior = rng.random((150, 9))
    posterior /= posterior.sum(axis=1, keepdims=True)
    parameters = np.zeros((4, 5))
    parameters[:, 0] = 1.0
    for j, k in enumerate(categories):
        parameters[j, 1:k] = np.linspace(-1.0, 1.0, k - 1)
    free = np.zeros((4, 5), dtype=bool)
    for j, k in enumerate(categories):
        free[j, :k] = True

    def fit(values: np.ndarray) -> np.ndarray:
        return mirt_rs.m_step_polytomous(
            responses,
            values,
            points,
            parameters,
            free,
            categories,
            grm,
            50,
            1e-10,
            1e-10,
            1,
            (0.1, 5.0),
            (-6.0, 6.0),
            None,
        )

    contiguous = fit(np.ascontiguousarray(posterior))
    np.testing.assert_array_equal(fit(np.asfortranarray(posterior)), contiguous)
    np.testing.assert_array_equal(
        fit(np.ascontiguousarray(posterior[:, ::-1])[:, ::-1]), contiguous
    )
    assert not np.array_equal(contiguous, parameters)


def _bounded_reference_em(
    responses: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    disc_box: tuple[float, float],
    diff_box: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """2PL EM whose M-steps maximize each item's expected counts in the box."""
    from scipy.optimize import minimize
    from scipy.special import log_expit, logsumexp

    correct = (responses == 1).astype(np.float64)
    observed = (responses >= 0).astype(np.float64)
    n_items = responses.shape[1]
    a, b = np.ones(n_items), np.zeros(n_items)
    previous = -np.inf
    for _ in range(2000):
        logits = a * (points[:, None] - b)
        joint = (
            correct @ log_expit(logits).T
            + (observed - correct) @ log_expit(-logits).T
            + np.log(weights)
        )
        marginal = logsumexp(joint, axis=1, keepdims=True)
        if abs(marginal.sum() - previous) < 1e-10:
            break
        previous = marginal.sum()
        posterior = np.exp(joint - marginal)
        r_k, n_k = correct.T @ posterior, observed.T @ posterior
        for j in range(n_items):

            def negative(x: np.ndarray, j: int = j) -> float:
                z = x[0] * (points - x[1])
                return -float(
                    np.sum(r_k[j] * log_expit(z) + (n_k[j] - r_k[j]) * log_expit(-z))
                )

            a[j], b[j] = minimize(
                negative,
                [a[j], b[j]],
                method="L-BFGS-B",
                bounds=[disc_box, diff_box],
                options={"ftol": 1e-15, "gtol": 1e-12},
            ).x
    return a, b


def test_bootstrap_items_on_a_bound_reach_the_box_optimum() -> None:
    # Regression (EMCORE-4): clamping each coordinate of the joint Newton step
    # left items whose difficulty sits on its bound at non-stationary slopes;
    # on these data item 1 ended at a = 0.1, 1600 log-likelihood units below
    # the box optimum of the same EM.
    from mirt.backends.rust.estimation import bootstrap_fit_2pl

    rng = np.random.default_rng(5)
    discrimination = np.r_[0.5, 0.45, rng.uniform(0.8, 2.0, 8)]
    difficulty = np.r_[-7.5, 7.8, rng.uniform(-1.5, 1.5, 8)]
    theta = rng.standard_normal(5000)
    probability = expit(discrimination * (theta[:, None] - difficulty))
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    quadrature = GaussHermiteQuadrature(n_points=21)
    points, weights = quadrature.nodes.ravel(), quadrature.weights
    boxes = {"disc_bounds": (0.1, 5.0), "diff_bounds": (-6.0, 6.0)}

    expected_a, expected_b = _bounded_reference_em(
        responses, points, weights, *boxes.values()
    )
    a, b = bootstrap_fit_2pl(
        responses,
        n_bootstrap=16,
        max_iter=500,
        tol=1e-6,
        seed=3,
        quad_points=points,
        quad_weights=weights,
        **boxes,
    )

    assert expected_b[1] == 6.0 and np.all(b[:, 1] == 6.0)
    # The replicate slopes scatter around the box optimum (sd 0.016).
    assert abs(a[:, 1].mean() - expected_a[1]) < 0.01


def test_bootstrap_replicates_of_neighbouring_seeds_are_unrelated() -> None:
    from mirt.backends.rust.estimation import bootstrap_fit_2pl

    responses = _responses(300, 5, seed=9)
    options = {"n_bootstrap": 3, "max_iter": 50, "tol": 1e-4}
    first, _ = bootstrap_fit_2pl(responses, seed=10, **options)
    second, _ = bootstrap_fit_2pl(responses, seed=11, **options)
    repeated, _ = bootstrap_fit_2pl(responses, seed=10, **options)

    # Replicate b used seed + b, so seed 11 replayed seed 10's replicates.
    assert not np.any(np.all(first[1:] == second[:-1], axis=1))
    np.testing.assert_array_equal(first, repeated)


def test_bootstrap_uses_the_given_grid_and_accurate_large_grids() -> None:
    from mirt.backends.rust.estimation import bootstrap_fit_2pl

    responses = _responses(400, 5, seed=13)
    options = {"n_bootstrap": 2, "max_iter": 200, "tol": 1e-6, "seed": 4}
    quadrature = GaussHermiteQuadrature(n_points=31)
    default = bootstrap_fit_2pl(responses, n_quadpts=31, **options)
    explicit = bootstrap_fit_2pl(
        responses,
        quad_points=quadrature.nodes.ravel(),
        quad_weights=quadrature.weights,
        **options,
    )
    for left, right in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(left, right)

    # The native Gauss-Hermite rule collapsed to two distinct nodes for 199 or
    # more points; the grid now comes from the package's quadrature.
    large = bootstrap_fit_2pl(responses, n_quadpts=201, **options)
    for coarse, fine in zip(default, large, strict=True):
        np.testing.assert_allclose(fine, coarse, atol=0.02)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"disc_bounds": (5.0, 0.1)}, "disc_bounds must satisfy lower <= upper"),
        ({"diff_bounds": (np.nan, 6.0)}, "diff_bounds must satisfy lower <= upper"),
        ({"quad_weights": np.array([0.5, 0.5])}, "quad_weights must have shape"),
        ({"quad_points": np.array([0.0, np.inf, 1.0])}, "quad_points must be finite"),
        ({"quad_weights": np.array([0.0, 0.0, 0.0])}, "positive sum"),
        ({"tol": 0.0}, "tol must be finite and positive"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
    ],
)
def test_native_bootstrap_kernel_rejects_invalid_inputs(
    changes: dict[str, object], message: str
) -> None:
    from mirt import mirt_rs

    arguments: dict[str, object] = {
        "responses": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "n_bootstrap": 1,
        "quad_points": np.array([-1.0, 0.0, 1.0]),
        "quad_weights": np.array([0.25, 0.5, 0.25]),
        "max_iter": 5,
        "tol": 1e-4,
        "seed": 1,
        "disc_bounds": (0.1, 5.0),
        "diff_bounds": (-6.0, 6.0),
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        mirt_rs.bootstrap_fit_2pl(*arguments.values())
