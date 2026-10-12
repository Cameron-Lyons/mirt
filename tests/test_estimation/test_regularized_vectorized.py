"""Regression coverage for regularized multidimensional estimation."""

from __future__ import annotations

from math import comb, log
from types import SimpleNamespace

import numpy as np
import pytest

import mirt
import mirt.estimation.regularized as regularized_module
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.regularized import (
    PenaltySpec,
    RegularizedMIRTEstimator,
    ebic_penalty,
)
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.numeric import logsumexp


def _simple_structure_responses(
    n_persons: int = 600, per_factor: int = 6, seed: int = 3
) -> np.ndarray:
    """Two-factor simple structure with slopes in [1, 2]."""
    rng = np.random.default_rng(seed)
    n_items = 2 * per_factor
    slopes = np.zeros((n_items, 2))
    for factor in range(2):
        rows = slice(factor * per_factor, (factor + 1) * per_factor)
        slopes[rows, factor] = rng.uniform(1.0, 2.0, per_factor)
    intercepts = rng.uniform(-1.0, 1.0, n_items)
    theta = rng.standard_normal((n_persons, 2))
    probabilities = 1 / (1 + np.exp(-(theta @ slopes.T + intercepts)))
    return (rng.random((n_persons, n_items)) < probabilities).astype(np.int64)


@pytest.fixture(scope="module")
def simple_structure() -> np.ndarray:
    return _simple_structure_responses()


def _slow_e_step(
    estimator: RegularizedMIRTEstimator,
    responses: np.ndarray,
    loadings: np.ndarray,
    intercepts: np.ndarray,
    density: GaussianDensity,
) -> tuple[np.ndarray, np.ndarray]:
    quadrature = estimator._quadrature
    assert quadrature is not None
    quad_points = quadrature.nodes
    quad_weights = quadrature.weights
    log_likelihoods = np.zeros((responses.shape[0], len(quad_weights)))

    for q, theta_q in enumerate(quad_points):
        probabilities = regularized_module.sigmoid(theta_q @ loadings.T + intercepts)
        probabilities = np.clip(
            probabilities,
            regularized_module.PROB_EPSILON,
            1 - regularized_module.PROB_EPSILON,
        )
        for item_idx in range(responses.shape[1]):
            observed = responses[:, item_idx] >= 0
            item_responses = responses[observed, item_idx]
            log_likelihoods[observed, q] += item_responses * np.log(
                probabilities[item_idx]
            ) + (1 - item_responses) * np.log1p(-probabilities[item_idx])

    log_prior_mass = density.log_quadrature_mass(quad_points, quad_weights)
    log_joint = log_likelihoods + log_prior_mass[None, :]
    log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
    return np.exp(log_joint - log_marginal), log_marginal.ravel()


@pytest.mark.parametrize("blocked_preparation", [False, True])
def test_vectorized_e_step_matches_reference_with_forced_chunks(
    monkeypatch: pytest.MonkeyPatch,
    blocked_preparation: bool,
) -> None:
    rng = np.random.default_rng(314)
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(n_points=7, n_dimensions=2)
    density = GaussianDensity(
        mean=np.array([0.25, -0.4]),
        cov=np.array([[1.2, 0.25], [0.25, 0.8]]),
        n_dimensions=2,
    )
    responses = rng.integers(0, 2, size=(18, 9), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[0] = -1
    responses[:, 3] = -1
    loadings = rng.normal(0.0, 0.8, size=(9, 2))
    intercepts = rng.normal(0.0, 1.0, size=9)

    expected_posterior, expected_marginal = _slow_e_step(
        estimator,
        responses,
        loadings,
        intercepts,
        density,
    )
    monkeypatch.setattr(regularized_module, "_MAX_ESTEP_TEMP_ENTRIES", 20)
    if blocked_preparation:
        from mirt.estimation import _em_context

        monkeypatch.setattr(_em_context, "_MAX_COUNT_ENTRIES", 20)

    responses.flags.writeable = loadings.flags.writeable = (
        intercepts.flags.writeable
    ) = False

    actual_posterior, actual_marginal = estimator._e_step(
        responses,
        loadings,
        intercepts,
        density,
    )

    np.testing.assert_allclose(
        actual_posterior,
        expected_posterior,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        actual_marginal,
        expected_marginal,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(actual_posterior.sum(axis=1), 1.0, atol=1e-14)


def test_vectorized_expected_counts_match_observed_item_reference() -> None:
    rng = np.random.default_rng(2718)
    responses = rng.integers(0, 2, size=(23, 8), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.25] = -1
    responses[:, 5] = -1
    posterior = rng.random((23, 17))
    posterior /= posterior.sum(axis=1, keepdims=True)

    actual_correct, actual_observed = EMFitContext(responses).expected_counts(posterior)
    expected_correct = np.zeros_like(actual_correct)
    expected_observed = np.zeros_like(actual_observed)
    for item_idx in range(responses.shape[1]):
        observed = responses[:, item_idx] >= 0
        expected_correct[item_idx] = np.sum(
            responses[observed, item_idx, None] * posterior[observed],
            axis=0,
        )
        expected_observed[item_idx] = posterior[observed].sum(axis=0)

    np.testing.assert_allclose(actual_correct, expected_correct, atol=1e-14)
    np.testing.assert_allclose(actual_observed, expected_observed, atol=1e-14)


def test_vectorized_lambda_max_matches_coordinatewise_reference() -> None:
    rng = np.random.default_rng(1618)
    responses = rng.integers(0, 2, size=(31, 7), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.2] = -1
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=5)

    actual = estimator._compute_lambda_max(responses)

    quadrature = estimator._quadrature
    assert quadrature is not None
    density = GaussianDensity(
        mean=np.zeros(2),
        cov=np.eye(2),
        n_dimensions=2,
    )
    with estimator._prepared(responses):
        loadings, intercepts = estimator._reference_solution(responses)
        posterior, _ = estimator._e_step(responses, loadings, intercepts, density)
    max_gradient = 0.0
    for item_idx in range(responses.shape[1]):
        observed = responses[:, item_idx] >= 0
        proportion = responses[observed, item_idx].mean()
        expected_correct = np.sum(
            responses[observed, item_idx, None] * posterior[observed],
            axis=0,
        )
        expected_observed = posterior[observed].sum(axis=0)
        residual = expected_correct - proportion * expected_observed
        for factor_idx in range(estimator.n_factors):
            gradient = abs(np.sum(residual * quadrature.nodes[:, factor_idx]))
            max_gradient = max(max_gradient, gradient)

    assert actual == pytest.approx(max_gradient, rel=1e-12, abs=1e-14)


def _first_m_step(
    estimator: RegularizedMIRTEstimator, responses: np.ndarray
) -> np.ndarray:
    with estimator._prepared(responses):
        loadings, intercepts = estimator._reference_solution(responses)
        estimator._adaptive_weights = np.ones_like(loadings)
        posterior, _ = estimator._e_step(
            responses, loadings, intercepts, GaussianDensity(n_dimensions=2)
        )
        return estimator._m_step_penalized(responses, posterior, loadings, intercepts)[
            0
        ]


def test_lambda_max_is_the_penalty_that_zeroes_the_first_m_step(simple_structure):
    # Zero loadings are stationary for every penalty, so the old gradient at
    # zero loadings gave lambda_max ~ 1e-14 and an unpenalized default path.
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=9)
    lambda_max = estimator._compute_lambda_max(simple_structure)

    assert lambda_max > 50.0
    above = RegularizedMIRTEstimator(
        n_factors=2, n_quadpts=9, lambda_val=lambda_max * 1.001
    )
    below = RegularizedMIRTEstimator(
        n_factors=2, n_quadpts=9, lambda_val=lambda_max * 0.999
    )
    assert np.count_nonzero(_first_m_step(above, simple_structure)) == 0
    assert np.count_nonzero(_first_m_step(below, simple_structure)) >= 1
    assert above.fit(simple_structure).n_nonzero == 0


def test_default_path_runs_from_the_null_model_to_a_dense_model(simple_structure):
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=9)

    path = estimator.fit_path(simple_structure, n_lambda=6)

    lambdas = [result.lambda_val for result in path]
    assert lambdas == sorted(lambdas, reverse=True)
    assert lambdas[0] == pytest.approx(estimator._compute_lambda_max(simple_structure))
    assert lambdas[-1] == pytest.approx(lambdas[0] * 0.01)
    assert path[0].n_nonzero == 0
    assert path[-1].n_nonzero >= 12
    assert estimator.penalty.lambda_val == 0.1


def test_moderate_penalty_recovers_simple_structure(simple_structure):
    # Equal starting slopes on both factors trapped this fit in one factor
    # duplicated: items 0-5 all zero and items 6-11 equal on both factors.
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=9, lambda_val=10.0)

    result = estimator.fit(simple_structure)

    dominant = np.argmax(np.abs(result.loadings), axis=1)
    assert len(set(dominant[:6])) == 1
    assert len(set(dominant[6:])) == 1
    assert dominant[0] != dominant[6]
    assert np.all(np.abs(result.loadings[np.arange(12), dominant]) > 0.4)
    symmetric = estimator.fit(
        simple_structure,
        start={
            "loadings": np.full((12, 2), 0.8),
            "intercepts": np.zeros(12),
        },
    )
    assert result.penalized_ll > symmetric.penalized_ll + 50.0


def test_fit_path_warm_starts_from_the_next_smaller_penalty(
    simple_structure, monkeypatch
):
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=5, max_iter=5)
    original = estimator._fit_prepared
    calls = []

    def record(responses, lambda_val, start=None):
        calls.append((lambda_val, None if start is None else start[0].copy()))
        return original(responses, lambda_val, start)

    monkeypatch.setattr(estimator, "_fit_prepared", record)

    path = estimator.fit_path(simple_structure, lambda_values=[20.0, 0.0, 5.0])

    assert [result.lambda_val for result in path] == [20.0, 0.0, 5.0]
    assert [lam for lam, _ in calls] == [0.0, 5.0, 20.0]
    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[1][1], path[1].loadings)
    np.testing.assert_array_equal(calls[2][1], path[2].loadings)


@pytest.mark.parametrize(
    "start",
    [
        "model",
        {"loadings": np.zeros((12, 2))},
        {"loadings": np.zeros((3, 2)), "intercepts": np.zeros(12)},
    ],
)
def test_fit_rejects_invalid_start(simple_structure, start) -> None:
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=3, max_iter=1)

    with pytest.raises(MirtValidationError, match="start"):
        estimator.fit(simple_structure, start=start)


@pytest.mark.parametrize("penalty", ["lasso", "ridge", "elastic_net"])
def test_numpy_coordinate_descent_matches_native_kernel(
    simple_structure, penalty: str
) -> None:
    # The NumPy updates used the gradient at a zero loading, which diverged,
    # and the native kernel applied the LASSO to ridge penalties.
    if not mirt.is_rust_available():
        pytest.skip("native backend unavailable")
    previous = mirt.get_backend()
    try:
        fits = {}
        for backend in ("numpy", "auto"):
            mirt.set_backend(backend)
            fits[backend] = RegularizedMIRTEstimator(
                penalty=penalty,
                lambda_val=15.0,
                alpha=0.5,
                n_factors=2,
                n_quadpts=7,
                max_iter=3,
            ).fit(simple_structure)
    finally:
        mirt.set_backend(previous)

    np.testing.assert_allclose(fits["numpy"].loadings, fits["auto"].loadings, atol=1e-6)
    np.testing.assert_allclose(
        fits["numpy"].intercepts, fits["auto"].intercepts, atol=1e-6
    )
    if penalty == "ridge":
        assert fits["auto"].n_nonzero == fits["auto"].loadings.size


def test_unpenalized_numpy_m_step_maximizes_expected_log_likelihood(
    simple_structure,
) -> None:
    previous = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        estimator = RegularizedMIRTEstimator(
            n_factors=2, n_quadpts=7, lambda_val=0.0, cd_max_iter=500, cd_tol=1e-12
        )
        with estimator._prepared(simple_structure):
            loadings, intercepts = estimator._reference_solution(simple_structure)
            estimator._adaptive_weights = np.ones_like(loadings)
            posterior, _ = estimator._e_step(
                simple_structure, loadings, intercepts, GaussianDensity(n_dimensions=2)
            )
            loadings, intercepts = estimator._m_step_penalized(
                simple_structure, posterior, loadings, intercepts
            )
            correct, observed = estimator._response_context(
                simple_structure
            ).expected_counts(posterior)
    finally:
        mirt.set_backend(previous)

    nodes = estimator._quadrature.nodes
    probabilities = 1 / (1 + np.exp(-(nodes @ loadings.T + intercepts)))
    residual = correct - observed * probabilities.T
    np.testing.assert_allclose(residual @ nodes, 0.0, atol=1e-6)
    np.testing.assert_allclose(residual.sum(axis=1), 0.0, atol=1e-6)


@pytest.mark.parametrize("penalty_type", ["unknown", "", 3])
def test_penalty_spec_rejects_unknown_types(penalty_type: object) -> None:
    with pytest.raises(ValueError, match="penalty type"):
        PenaltySpec(penalty_type, 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("lambda_value", [-0.1, np.nan, np.inf, True, "0.1"])
def test_penalty_spec_rejects_invalid_strengths(lambda_value: object) -> None:
    with pytest.raises(ValueError, match="lambda_val"):
        PenaltySpec("lasso", lambda_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_penalty_spec_rejects_invalid_mixing(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PenaltySpec("elastic_net", 0.1, alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_factors": 1}, "n_factors"),
        ({"n_factors": 2.5}, "n_factors"),
        ({"n_quadpts": 0}, "n_quadpts"),
        ({"cd_max_iter": 0}, "cd_max_iter"),
        ({"cd_tol": 0.0}, "cd_tol"),
        ({"cd_tol": np.inf}, "cd_tol"),
    ],
)
def test_estimator_rejects_invalid_solver_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RegularizedMIRTEstimator(**kwargs)  # type: ignore[arg-type]


def test_estimator_requires_boolean_adaptive_flag() -> None:
    with pytest.raises(TypeError, match="adaptive"):
        RegularizedMIRTEstimator(adaptive=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.array([[0.5, 1.0], [0.0, 1.0]]), "integer response codes"),
        (np.array([[2, 1], [0, 1]]), "binary responses"),
        (np.array([[np.inf, 1.0], [0.0, 1.0]]), "finite response codes"),
    ],
)
def test_fit_rejects_invalid_binary_responses(
    responses: np.ndarray,
    message: str,
) -> None:
    estimator = RegularizedMIRTEstimator(
        n_factors=2,
        n_quadpts=3,
        max_iter=1,
        cd_max_iter=1,
    )

    with pytest.raises(MirtDataError, match=message):
        estimator.fit(responses)


@pytest.mark.parametrize("lambda_value", [-1.0, np.nan, np.inf])
def test_fit_rejects_invalid_lambda_override(lambda_value: float) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="lambda_val"):
        estimator.fit(responses, lambda_val=lambda_value)


def test_large_loading_space_has_finite_ebic() -> None:
    rng = np.random.default_rng(20260829)
    responses = rng.integers(0, 2, size=(12, 50), dtype=np.int32)
    estimator = RegularizedMIRTEstimator(
        n_factors=2,
        n_quadpts=2,
        max_iter=1,
        cd_max_iter=1,
    )

    result = estimator.fit(responses, lambda_val=0.1)

    assert np.isfinite(result.ebic)
    assert result.ebic == pytest.approx(
        result.bic + log(comb(result.loadings.size, result.n_nonzero))
    )


def test_ebic_model_space_term_depends_on_model_size(simple_structure) -> None:
    # EBIC added the constant n_items * n_factors * log(2), ranking penalties
    # exactly as BIC did.
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=7)

    path = estimator.fit_path(simple_structure, lambda_values=[0.0, 5.0, 20.0])

    differences = [result.ebic - result.bic for result in path]
    for result, difference in zip(path, differences, strict=True):
        assert difference == pytest.approx(log(comb(24, result.n_nonzero)))
    assert len({result.n_nonzero for result in path}) == 3
    assert max(differences) - min(differences) > 1.0


def test_ebic_penalty_follows_chen_and_chen() -> None:
    assert ebic_penalty(24, 0) == 0.0
    assert ebic_penalty(24, 24) == pytest.approx(0.0, abs=1e-12)
    assert ebic_penalty(24, 12) == pytest.approx(log(comb(24, 12)))
    assert ebic_penalty(24, 12, gamma=1.0) == pytest.approx(2 * log(comb(24, 12)))
    assert ebic_penalty(0, 0) == 0.0
    with pytest.raises(MirtValidationError, match="n_selected"):
        ebic_penalty(4, 5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_values": []}, "lambda_values"),
        ({"lambda_values": [-0.1]}, "lambda_values"),
        ({"lambda_values": [np.nan]}, "lambda_values"),
        ({"n_lambda": 0}, "n_lambda"),
        ({"lambda_min_ratio": 0.0}, "lambda_min_ratio"),
        ({"lambda_min_ratio": 1.1}, "lambda_min_ratio"),
    ],
)
def test_fit_path_rejects_invalid_grid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match=message):
        estimator.fit_path(responses, **kwargs)  # type: ignore[arg-type]


def test_fit_path_handles_zero_lambda_max_without_logarithmic_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.full((4, 3), -1, dtype=np.int32)
    fitted_lambdas: list[float] = []
    monkeypatch.setattr(estimator, "_compute_lambda_max", lambda _responses: 0.0)

    def record_fit(
        _responses: np.ndarray,
        lambda_val: float | None = None,
        start: object = None,
    ) -> object:
        assert lambda_val is not None
        fitted_lambdas.append(lambda_val)
        return SimpleNamespace(loadings=np.zeros((3, 2)), intercepts=np.zeros(3))

    monkeypatch.setattr(estimator, "_fit_prepared", record_fit)

    results = estimator.fit_path(responses, n_lambda=3)

    assert len(results) == 3
    assert fitted_lambdas == [0.0, 0.0, 0.0]


def test_long_fit_retains_log_likelihood_and_avoids_false_convergence(monkeypatch):
    responses = np.tile([0, 1], (4, 800))
    estimator = RegularizedMIRTEstimator(
        n_factors=2, n_quadpts=3, max_iter=2, tol=1e-12, cd_max_iter=1
    )
    monkeypatch.setattr(GaussianDensity, "update", lambda *_args: None)

    def move_intercepts(_responses, _posterior, loadings, intercepts):
        return loadings.copy(), intercepts + 0.4

    monkeypatch.setattr(estimator, "_m_step_penalized", move_intercepts)
    result = estimator.fit(responses)
    quadrature = estimator._quadrature
    likelihoods = np.column_stack(
        [
            result.model.log_likelihood(responses, point[None])
            for point in quadrature.nodes
        ]
    )
    expected = float(logsumexp(likelihoods + np.log(quadrature.weights), axis=1).sum())
    assert expected < -750 * len(responses)
    assert result.log_likelihood == pytest.approx(expected, abs=1e-9)
    assert result.penalized_ll == pytest.approx(
        expected - estimator._compute_penalty(result.loadings)
    )
    assert result.aic == pytest.approx(-2 * expected + 2 * result.n_parameters)
    assert result.bic == pytest.approx(
        -2 * expected + result.n_parameters * np.log(len(responses))
    )
    assert not result.converged
    assert result.n_iterations == 2
    assert len(estimator.convergence_history) == 3
    assert estimator.convergence_history[-1] == pytest.approx(expected)


def test_fit_reuses_response_preparation_and_releases_it_between_calls(monkeypatch):
    estimator = RegularizedMIRTEstimator(
        n_quadpts=3, max_iter=2, cd_max_iter=1, tol=1e-12
    )
    original = estimator._e_step
    contexts = []
    component_ids = []

    def record(*args):
        result = original(*args)
        context = estimator._fit_context
        contexts.append(context)
        component_ids.append(id(context._components))
        return result

    monkeypatch.setattr(estimator, "_e_step", record)
    rng = np.random.default_rng(937)
    first = rng.integers(-1, 2, (19, 5))
    second = rng.integers(-1, 2, (13, 4))
    estimator.fit(first)
    assert contexts and contexts[0] is not None
    assert all(context is contexts[0] for context in contexts)
    assert len(set(component_ids)) == 1
    assert estimator._fit_context is None
    previous_context = contexts[0]
    contexts.clear()
    component_ids.clear()
    result = estimator.fit(second)
    assert all(context is contexts[0] for context in contexts)
    assert contexts[0] is not previous_context
    assert estimator._fit_context is None
    expected = RegularizedMIRTEstimator(
        n_quadpts=3, max_iter=2, cd_max_iter=1, tol=1e-12
    ).fit(second)
    assert result.log_likelihood == pytest.approx(expected.log_likelihood, abs=1e-12)
    np.testing.assert_allclose(result.loadings, expected.loadings, atol=1e-12)


@pytest.mark.parametrize("stage", ["e_step", "m_step", "density"])
def test_failed_adaptive_warmstart_restores_penalty_weights_and_context(
    monkeypatch, stage
):
    estimator = RegularizedMIRTEstimator(
        lambda_val=0.37, adaptive=True, n_quadpts=3, max_iter=1, cd_max_iter=1
    )
    original_weights = np.full((3, 2), 2.0)
    estimator._adaptive_weights = original_weights

    def fail(*_args):
        assert estimator.penalty.lambda_val == 0.0
        np.testing.assert_array_equal(estimator._adaptive_weights, 1.0)
        raise RuntimeError("warmstart failed")

    if stage == "density":
        monkeypatch.setattr(GaussianDensity, "update", fail)
    else:
        monkeypatch.setattr(
            estimator, "_e_step" if stage == "e_step" else "_m_step_penalized", fail
        )
    with pytest.raises(RuntimeError, match="warmstart failed"):
        estimator.fit(np.array([[0, 1, 0], [1, 0, 1]]))
    assert estimator.penalty.lambda_val == 0.37
    assert estimator._adaptive_weights is original_weights
    assert estimator._fit_context is None


@pytest.mark.parametrize("penalty", ["lasso", "ridge", "elastic_net"])
def test_adaptive_fit_finishes_with_requested_regularization(penalty):
    estimator = RegularizedMIRTEstimator(
        penalty=penalty,
        lambda_val=0.23,
        adaptive=True,
        n_quadpts=3,
        max_iter=2,
        cd_max_iter=1,
    )
    responses = np.random.default_rng(171).integers(-1, 2, (21, 4))
    result = estimator.fit(responses)
    assert result.lambda_val == 0.23
    assert estimator.penalty.lambda_val == 0.23
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(estimator._adaptive_weights).all()
    assert result.penalized_ll == pytest.approx(
        result.log_likelihood - estimator._compute_penalty(result.loadings)
    )
    assert estimator._fit_context is None
