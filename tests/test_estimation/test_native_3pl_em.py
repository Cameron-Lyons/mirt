"""Coverage for public 3PL EM native dispatch and fallback behavior."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import mirt
import mirt.estimation.em as em_module
from mirt._backend_state import get_backend_preference, set_backend_preference
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.models.dichotomous import ThreeParameterLogistic


@pytest.fixture(autouse=True)
def _restore_backend() -> Iterator[None]:
    previous = get_backend_preference()
    set_backend_preference("auto")
    try:
        yield
    finally:
        set_backend_preference(previous)


def _responses() -> np.ndarray:
    return np.array(
        [
            [1, 0, 1],
            [0, 1, -1],
            [1, 1, 0],
            [-1, 0, 1],
        ],
        dtype=np.int32,
    )


def _enable_fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em_module, "RUST_AVAILABLE", True)
    monkeypatch.setattr(em_module, "should_use_rust", lambda use_rust: use_rust)
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)


def test_native_iteration_commits_only_after_convergence_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_iteration(
        responses: np.ndarray,
        quad_points: np.ndarray,
        quad_weights: np.ndarray,
        discrimination: np.ndarray,
        difficulty: np.ndarray,
        guessing: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        nonlocal calls
        calls += 1
        posterior = np.full(
            (responses.shape[0], len(quad_points)),
            1.0 / len(quad_points),
        )
        return (
            discrimination + 0.1,
            difficulty + 0.2,
            guessing + 0.01,
            posterior,
            -10.0,
        )

    _enable_fake_native(monkeypatch)
    monkeypatch.setattr(em_module, "em_iteration_3pl", fake_iteration)

    model = ThreeParameterLogistic(n_items=3)
    estimator = EMEstimator(n_quadpts=5, max_iter=3, tol=1e-8, use_gpu=False)
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})

    result = estimator.fit(model, _responses())

    assert calls == 2
    np.testing.assert_allclose(model.discrimination, 1.1)
    np.testing.assert_allclose(model.difficulty, 0.2)
    np.testing.assert_allclose(model.guessing, 0.21)
    assert estimator.convergence_history == [-10.0, -10.0]
    assert result.log_likelihood == -10.0
    assert result.n_iterations == 2
    assert result.converged is True


@pytest.mark.parametrize("native_result", [None, object()])
def test_unavailable_or_malformed_native_result_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    native_result: object,
) -> None:
    _enable_fake_native(monkeypatch)
    monkeypatch.setattr(
        em_module,
        "em_iteration_3pl",
        lambda *args, **kwargs: native_result,
    )

    estimator = EMEstimator(n_quadpts=5, max_iter=1, use_gpu=False)
    python_e_steps = 0
    original_e_step = estimator._e_step

    def counting_e_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal python_e_steps
        python_e_steps += 1
        return original_e_step(*args, **kwargs)

    monkeypatch.setattr(estimator, "_e_step", counting_e_step)
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})

    result = estimator.fit(ThreeParameterLogistic(n_items=3), _responses())

    assert python_e_steps == 2
    assert np.isfinite(result.log_likelihood)


@pytest.mark.parametrize(
    ("backend", "use_rust"),
    [("numpy", True), ("auto", False)],
)
def test_backend_controls_can_disable_native_iteration(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    use_rust: bool,
) -> None:
    set_backend_preference(backend)
    monkeypatch.setattr(em_module, "RUST_AVAILABLE", True)
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)
    monkeypatch.setattr(
        em_module,
        "em_iteration_3pl",
        lambda *args, **kwargs: pytest.fail("native iteration should be disabled"),
    )

    estimator = EMEstimator(
        n_quadpts=5,
        max_iter=1,
        use_gpu=False,
        use_rust=use_rust,
    )
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})

    result = estimator.fit(ThreeParameterLogistic(n_items=3), _responses())

    assert np.isfinite(result.log_likelihood)


def test_native_eligibility_is_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_native(monkeypatch)
    model = ThreeParameterLogistic(n_items=3)
    responses = _responses()

    standard = EMEstimator(use_gpu=False)
    standard._latent_density = GaussianDensity()
    assert standard._can_use_rust_3pl(model, responses) is True

    shifted = EMEstimator(use_gpu=False)
    shifted._latent_density = GaussianDensity(mean=np.array([0.25]))
    assert shifted._can_use_rust_3pl(model, responses) is False

    custom_epsilon = EMEstimator(prob_epsilon=1e-8, use_gpu=False)
    custom_epsilon._latent_density = GaussianDensity()
    assert custom_epsilon._can_use_rust_3pl(model, responses) is False

    # The native M-step runs on a pool of n_jobs workers.
    parallel = EMEstimator(n_jobs=2, use_gpu=False)
    parallel._latent_density = GaussianDensity()
    assert parallel._can_use_rust_3pl(model, responses) is True

    nonbinary = responses.copy()
    nonbinary[0, 0] = 2
    assert standard._can_use_rust_3pl(model, nonbinary) is False


def test_fit_mirt_propagates_native_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    init_kwargs: dict[str, Any] = {}

    class FakeEstimator:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def fit(self, model: object, responses: np.ndarray) -> object:
            return sentinel

    monkeypatch.setattr(em_module, "EMEstimator", FakeEstimator)

    result = mirt.fit_mirt(_responses(), model="3PL", use_rust=False)

    assert result is sentinel
    assert init_kwargs["use_rust"] is False


@pytest.mark.skipif(
    not em_module.RUST_AVAILABLE,
    reason="compiled backend is not available",
)
def test_real_native_fit_refreshes_final_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(8031)
    responses = rng.integers(0, 2, size=(80, 5), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.1] = -1
    native_iteration = em_module.em_iteration_3pl
    calls = 0

    def counting_iteration(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return native_iteration(*args, **kwargs)

    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)
    monkeypatch.setattr(em_module, "em_iteration_3pl", counting_iteration)

    model = ThreeParameterLogistic(n_items=responses.shape[1])
    estimator = EMEstimator(
        n_quadpts=7,
        max_iter=2,
        tol=1e-12,
        item_optim_maxiter=5,
        use_gpu=False,
    )
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})

    result = estimator.fit(model, responses)
    _, log_marginal = estimator._e_step(model, responses)
    expected_log_likelihood = np.sum(log_marginal)

    assert calls == 2
    assert result.log_likelihood == pytest.approx(expected_log_likelihood)
    assert np.all(np.isfinite(model.discrimination))
    assert np.all(np.isfinite(model.difficulty))
    assert np.all(np.isfinite(model.guessing))


@pytest.mark.skipif(
    not em_module.RUST_AVAILABLE,
    reason="compiled backend is not available",
)
def test_real_native_fit_tracks_numpy_with_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(90210)
    n_persons, n_items = 600, 8
    theta = rng.standard_normal(n_persons)
    discrimination = rng.uniform(0.8, 1.6, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    guessing = rng.uniform(0.08, 0.25, n_items)
    linear_predictor = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    probability = guessing[None, :] + (1.0 - guessing[None, :]) / (
        1.0 + np.exp(-linear_predictor)
    )
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    responses[rng.random(responses.shape) < 0.1] = -1
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)

    def fit(backend: str) -> tuple[Any, list[float]]:
        set_backend_preference(backend)
        estimator = EMEstimator(
            n_quadpts=15,
            max_iter=10,
            tol=1e-12,
            item_optim_maxiter=20,
            use_gpu=False,
        )
        monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})
        result = estimator.fit(ThreeParameterLogistic(n_items=n_items), responses)
        return result, estimator.convergence_history

    numpy_result, _ = fit("numpy")
    native_result, native_history = fit("rust")
    relative_likelihood_gap = abs(
        native_result.log_likelihood - numpy_result.log_likelihood
    ) / abs(numpy_result.log_likelihood)

    assert np.min(np.diff(native_history)) >= -1e-8
    assert relative_likelihood_gap < 0.003
    assert (
        np.corrcoef(
            native_result.model.discrimination,
            numpy_result.model.discrimination,
        )[0, 1]
        > 0.9
    )
    assert (
        np.corrcoef(native_result.model.difficulty, numpy_result.model.difficulty)[0, 1]
        > 0.9
    )
    assert (
        np.mean(np.abs(native_result.model.guessing - numpy_result.model.guessing))
        < 0.05
    )


def _bounded_guessing_responses() -> np.ndarray:
    """Responses whose 3PL optimum puts several guessing parameters at zero."""
    rng = np.random.default_rng(3)
    discrimination = np.array([1.6, 1.2, 1.8, 1.4, 1.0, 1.5])
    difficulty = np.array([-0.8, 0.2, 0.6, -0.2, 0.9, 1.3])
    guessing = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.2])
    theta = rng.standard_normal(1500)
    probability = guessing + (1.0 - guessing) / (
        1.0 + np.exp(-discrimination * (theta[:, None] - difficulty))
    )
    return (rng.random(probability.shape) < probability).astype(np.int32)


def _fit_3pl(
    model: ThreeParameterLogistic,
    responses: np.ndarray,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    max_iter: int,
) -> Any:
    set_backend_preference(backend)
    estimator = EMEstimator(n_quadpts=15, max_iter=max_iter, tol=1e-8, use_gpu=False)
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *args: {})
    return estimator.fit(model, responses)


@pytest.mark.skipif(
    not em_module.RUST_AVAILABLE,
    reason="compiled backend is not available",
)
def test_native_optimum_with_guessing_on_its_bound_is_a_numpy_fixed_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The native M-step used to solve the joint (a, b, c) Newton system and then
    # clip c at zero, so it stalled with large (a, b) scores on those items and
    # a log-likelihood about 2.9 below the bounded optimum.
    responses = _bounded_guessing_responses()
    native_iteration = em_module.em_iteration_3pl
    calls = 0

    def counting_iteration(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return native_iteration(*args, **kwargs)

    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)
    monkeypatch.setattr(em_module, "em_iteration_3pl", counting_iteration)
    model = ThreeParameterLogistic(n_items=responses.shape[1])
    native = _fit_3pl(model, responses, "rust", monkeypatch, max_iter=1000)
    assert native.converged
    assert calls > 1
    assert np.count_nonzero(model.guessing == 0.0) >= 3
    native_parameters = {
        name: values.copy() for name, values in model.parameters.items()
    }

    # Warm-started NumPy EM must have nothing left to improve.
    calls = 0
    refit = _fit_3pl(model, responses, "numpy", monkeypatch, max_iter=100)

    assert calls == 0
    assert refit.log_likelihood - native.log_likelihood < 1e-4
    for name, values in native_parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-3)


@pytest.mark.slow
@pytest.mark.skipif(
    not em_module.RUST_AVAILABLE,
    reason="compiled backend is not available",
)
def test_native_fit_reaches_numpy_optimum_with_guessing_on_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _bounded_guessing_responses()
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)
    fits = {
        backend: _fit_3pl(
            ThreeParameterLogistic(n_items=responses.shape[1]),
            responses,
            backend,
            monkeypatch,
            max_iter=3000,
        )
        for backend in ("numpy", "rust")
    }

    assert fits["numpy"].converged and fits["rust"].converged
    assert fits["rust"].log_likelihood == pytest.approx(
        fits["numpy"].log_likelihood, abs=1e-3
    )
    for name in ("discrimination", "difficulty", "guessing"):
        np.testing.assert_allclose(
            fits["rust"].model.parameters[name],
            fits["numpy"].model.parameters[name],
            atol=5e-3,
        )


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_parallel_fits_keep_the_native_iteration(
    monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    # Regression: n_jobs != 1 ran the per-item SciPy M-step, 6-9x slower.
    pools = []
    native = em_module.em_iteration_3pl

    def record(*args: Any, **kwargs: Any) -> Any:
        pools.append(kwargs["pool"])
        return native(*args, **kwargs)

    data = np.asarray(mirt.simdata(n_persons=400, n_items=6, model="3PL", seed=3))
    options = {"compute_standard_errors": False, "use_gpu": False}
    serial = EMEstimator(**options).fit(ThreeParameterLogistic(6), data)
    monkeypatch.setattr(em_module, "em_iteration_3pl", record)
    parallel = EMEstimator(n_jobs=n_jobs, **options).fit(
        ThreeParameterLogistic(6), data
    )

    assert pools and all(pool is pools[0] and pool is not None for pool in pools)
    assert parallel.n_iterations == serial.n_iterations
    for name, values in serial.model.parameters.items():
        np.testing.assert_array_equal(parallel.model.parameters[name], values)


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"quad_weights": np.array([1.0])}, "quad_weights must have shape"),
        ({"discrimination": np.array([1.0])}, "discrimination must have shape"),
        ({"prior_var": 0.0}, "prior_var finite and positive"),
        ({"prior_mean": np.nan}, "prior_mean must be finite"),
        ({"max_m_iter": 0}, "max_m_iter must be a positive integer"),
        ({"disc_bounds": (5.0, 0.1)}, "disc_bounds must satisfy lower <= upper"),
        ({"guess_bounds": (np.nan, 0.3)}, "guess_bounds must satisfy"),
        ({"responses": np.array([[0, 2]], dtype=np.int32)}, "only 0 or 1"),
        ({"n_jobs": 0}, "n_jobs must be positive"),
    ],
)
def test_native_kernel_raises_value_errors_instead_of_panicking(
    changes: dict[str, Any], message: str
) -> None:
    # Regression: these reached assert_eq!, an index panic or f64::clamp in
    # the kernel, which the wrapper's callers could not catch.
    from mirt import mirt_rs

    arguments: dict[str, Any] = {
        "responses": np.array([[0, 1]], dtype=np.int32),
        "quad_points": np.array([0.0, 1.0]),
        "quad_weights": np.array([0.5, 0.5]),
        "discrimination": np.array([1.0, 1.0]),
        "difficulty": np.array([0.0, 0.0]),
        "guessing": np.array([0.1, 0.1]),
        "prior_mean": 0.0,
        "prior_var": 1.0,
        "max_m_iter": 5,
        "m_tol": 1e-4,
        "disc_bounds": (0.1, 5.0),
        "diff_bounds": (-6.0, 6.0),
        "guess_bounds": (0.0, 0.35),
        "regularization": 0.01,
        "regularization_c": 0.1,
        "frequencies": None,
        "n_jobs": 1,
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        mirt_rs.em_iteration_3pl(*arguments.values())


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
def test_native_kernel_rejects_a_pool_of_another_size() -> None:
    from mirt import mirt_rs

    pool = mirt_rs.EMThreadPool(3)
    with pytest.raises(ValueError, match="pool size must match"):
        mirt_rs.em_iteration_3pl(
            np.array([[0, 1]], dtype=np.int32),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            np.ones(2),
            np.zeros(2),
            np.full(2, 0.1),
            0.0,
            1.0,
            5,
            1e-4,
            (0.1, 5.0),
            (-6.0, 6.0),
            (0.0, 0.35),
            0.01,
            0.1,
            None,
            2,
            pool,
        )
