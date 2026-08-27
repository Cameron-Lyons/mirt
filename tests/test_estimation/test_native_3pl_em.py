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

    parallel = EMEstimator(n_jobs=2, use_gpu=False)
    parallel._latent_density = GaussianDensity()
    assert parallel._can_use_rust_3pl(model, responses) is False

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
    _, marginal_likelihood = estimator._e_step(model, responses)
    expected_log_likelihood = np.sum(np.log(marginal_likelihood + 1e-300))

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
