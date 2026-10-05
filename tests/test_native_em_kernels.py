"""Native EM kernels against small NumPy references of the same algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logsumexp

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


def _newton_2pl(
    r_k: np.ndarray,
    n_k: np.ndarray,
    points: np.ndarray,
    a: float,
    b: float,
    *,
    max_iter: int = 10,
    tol: float = 1e-4,
    damping: float = 0.5,
    regularization: float = 0.01,
) -> tuple[float, float]:
    keep = n_k >= EPSILON
    r_k, n_k, points = r_k[keep], n_k[keep], points[keep]
    for _ in range(max_iter):
        p = np.clip(expit(a * (points - b)), EPSILON, 1 - EPSILON)
        residual = r_k - n_k * p
        info = n_k * p * (1 - p)
        grad_a = np.sum(residual * (points - b))
        grad_b = np.sum(-residual * a)
        hess_aa = np.sum(-info * (points - b) ** 2) - regularization
        hess_bb = np.sum(-info * a * a) - regularization
        hess_ab = np.sum(info * a * (points - b))
        det = hess_aa * hess_bb - hess_ab**2
        if abs(det) < EPSILON:
            break
        delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det
        delta_b = (-hess_ab * grad_a + hess_aa * grad_b) / det
        a = float(np.clip(a - damping * delta_a, 0.1, 5.0))
        b = float(np.clip(b - damping * delta_b, -6.0, 6.0))
        if abs(delta_a) < tol and abs(delta_b) < tol:
            break
    return a, b


def _reference_em_fit_2pl(
    responses: np.ndarray,
    frequencies: np.ndarray,
    n_quadpts: int,
    n_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    quadrature = GaussHermiteQuadrature(n_points=n_quadpts)
    points = quadrature.nodes.ravel()
    log_weights = np.log(quadrature.weights.ravel())
    correct = (responses == 1).astype(np.float64)
    observed = (responses >= 0).astype(np.float64)
    weights = frequencies[:, None]

    proportion = np.clip(
        (correct * weights).sum(axis=0) / (observed * weights).sum(axis=0), 0.01, 0.99
    )
    a = np.ones(responses.shape[1])
    b = -np.log(proportion) / np.maximum(np.abs(np.log(1 - proportion)), 0.01)

    def e_step(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = a * (points[:, None] - b)
        log_likelihood = (
            correct @ -np.logaddexp(0.0, -logits).T
            + (observed - correct) @ -np.logaddexp(0.0, logits).T
        )
        log_joint = log_likelihood + log_weights
        log_marginal = logsumexp(log_joint, axis=1)
        return np.exp(log_joint - log_marginal[:, None]), log_marginal

    for _ in range(n_iterations):
        posterior, _ = e_step(a, b)
        posterior *= weights
        r_k, n_k = correct.T @ posterior, observed.T @ posterior
        a, b = map(
            np.array,
            zip(
                *(
                    _newton_2pl(r_k[j], n_k[j], points, a[j], b[j])
                    for j in range(len(a))
                ),
                strict=True,
            ),
        )
    _, log_marginal = e_step(a, b)
    return a, b, float(frequencies @ log_marginal)


def test_em_fit_2pl_matches_numpy_reference_with_weights_and_missing_data() -> None:
    from mirt.backends.rust.estimation import em_fit_2pl

    responses = _responses(400, 7, seed=31)
    frequencies = np.random.default_rng(2).integers(1, 4, len(responses)).astype(float)

    a, b, log_likelihood, iterations, converged = em_fit_2pl(
        responses, n_quadpts=15, max_iter=6, tol=1e-300, frequencies=frequencies
    )
    expected_a, expected_b, expected_ll = _reference_em_fit_2pl(
        responses, frequencies, n_quadpts=15, n_iterations=6
    )

    assert (iterations, converged) == (6, False)
    np.testing.assert_allclose(a, expected_a, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(b, expected_b, rtol=1e-9, atol=1e-10)
    assert log_likelihood == pytest.approx(expected_ll, rel=1e-12)


def test_converged_em_fit_reports_likelihood_at_returned_parameters() -> None:
    from mirt.backends.rust.estep import e_step_complete
    from mirt.backends.rust.estimation import em_fit_2pl

    responses = _responses(300, 6, seed=5)
    frequencies = np.ones(len(responses))

    a, b, log_likelihood, iterations, converged = em_fit_2pl(
        responses, n_quadpts=15, max_iter=500, tol=1e-3, frequencies=frequencies
    )
    quadrature = GaussHermiteQuadrature(n_points=15)
    _, marginal = e_step_complete(
        responses, quadrature.nodes.ravel(), quadrature.weights.ravel(), a, b
    )

    assert converged and iterations < 500
    assert log_likelihood == pytest.approx(np.log(marginal).sum(), abs=1e-10)


def test_em_iteration_2pl_m_step_matches_numpy_counts_and_newton() -> None:
    from mirt.backends.rust.estep import e_step_complete
    from mirt.backends.rust.estimation import em_iteration_2pl

    responses = _responses(350, 6, seed=9)
    quadrature = GaussHermiteQuadrature(n_points=21)
    points, weights = quadrature.nodes.ravel(), quadrature.weights.ravel()
    a0 = np.linspace(0.8, 1.6, 6)
    b0 = np.linspace(-1.0, 1.0, 6)

    result = em_iteration_2pl(responses, points, weights, a0, b0, max_m_iter=7)
    assert result is not None
    a, b, posterior, log_likelihood = result
    expected_posterior, marginal = e_step_complete(responses, points, weights, a0, b0)

    np.testing.assert_allclose(posterior, expected_posterior, rtol=1e-12, atol=1e-14)
    assert log_likelihood == pytest.approx(np.log(marginal).sum(), rel=1e-12)
    correct = (responses == 1).astype(np.float64)
    observed = (responses >= 0).astype(np.float64)
    r_k, n_k = correct.T @ posterior, observed.T @ posterior
    for j in range(6):
        expected = _newton_2pl(r_k[j], n_k[j], points, a0[j], b0[j], max_iter=7)
        np.testing.assert_allclose((a[j], b[j]), expected, rtol=1e-10, atol=1e-12)


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
            None,
        )

    contiguous = fit(np.ascontiguousarray(posterior))
    np.testing.assert_array_equal(fit(np.asfortranarray(posterior)), contiguous)
    np.testing.assert_array_equal(
        fit(np.ascontiguousarray(posterior[:, ::-1])[:, ::-1]), contiguous
    )
    assert not np.array_equal(contiguous, parameters)
