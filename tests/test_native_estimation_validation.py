"""Validation contracts for accelerated estimation entry points."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mirt._rust_backend import RUST_AVAILABLE
from mirt.backends.rust.estimation import (
    bootstrap_fit_2pl,
    em_fit_2pl,
    gibbs_sample_2pl,
    mhrm_fit_2pl,
)

native = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")


def _binary_responses(n_persons: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_persons)
    difficulty = np.linspace(-1.5, 1.5, n_items)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty)))
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    responses[rng.random(responses.shape) < 0.05] = -1
    return responses


@pytest.mark.parametrize("native_fit", [em_fit_2pl, bootstrap_fit_2pl])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_quadpts": 0}, "n_quadpts must be positive"),
        ({"n_quadpts": True}, "n_quadpts must be positive"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"max_iter": 1.5}, "max_iter must be at least 1"),
        ({"tol": 0.0}, "tol must be positive"),
        ({"tol": np.nan}, "tol must be positive"),
    ],
)
def test_native_fit_rejects_invalid_em_controls(
    native_fit: Callable[..., Any],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    responses = np.array([[0], [1]], dtype=np.int32)

    with pytest.raises(ValueError, match=message):
        native_fit(responses, **kwargs)


@pytest.mark.parametrize("n_bootstrap", [0, False, 1.5])
def test_native_bootstrap_requires_positive_integer_replicates(
    n_bootstrap: Any,
) -> None:
    responses = np.array([[0], [1]], dtype=np.int32)

    with pytest.raises(ValueError, match="n_bootstrap must be at least 1"):
        bootstrap_fit_2pl(responses, n_bootstrap=n_bootstrap)


def test_native_bootstrap_validates_warm_start_parameters() -> None:
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="provided together"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.ones(2),
        )
    with pytest.raises(ValueError, match="one value per item"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.ones(1),
            initial_difficulty=np.zeros(1),
        )
    with pytest.raises(ValueError, match="finite"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.array([1.0, np.nan]),
            initial_difficulty=np.zeros(2),
        )


def test_native_bootstrap_accepts_warm_start_parameters() -> None:
    responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int32)
    kwargs = {
        "n_bootstrap": 2,
        "max_iter": 2,
        "seed": 42,
        "initial_discrimination": np.array([1.2, 0.8]),
        "initial_difficulty": np.array([-0.25, 0.25]),
    }

    if not RUST_AVAILABLE:
        with pytest.raises(
            RuntimeError, match="Rust backend required for bootstrap_fit_2pl"
        ):
            bootstrap_fit_2pl(responses, **kwargs)
        return

    discrimination, difficulty = bootstrap_fit_2pl(responses, **kwargs)
    repeated_discrimination, repeated_difficulty = bootstrap_fit_2pl(
        responses, **kwargs
    )

    assert discrimination.shape == (2, 2)
    assert difficulty.shape == (2, 2)
    assert np.isfinite(discrimination).all()
    assert np.isfinite(difficulty).all()
    np.testing.assert_array_equal(discrimination, repeated_discrimination)
    np.testing.assert_array_equal(difficulty, repeated_difficulty)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iter": 0}, "n_iter must be at least 1"),
        ({"n_iter": 5.0}, "n_iter must be at least 1"),
        ({"thin": 0}, "thin must be at least 1"),
        ({"thin": True}, "thin must be at least 1"),
        ({"burnin": -1}, "burnin must be a non-negative integer"),
        ({"n_iter": 5, "burnin": 10}, "burnin must be less than n_iter"),
        ({"n_iter": 5, "burnin": 5}, "burnin must be less than n_iter"),
    ],
)
def test_native_gibbs_rejects_invalid_schedules(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        gibbs_sample_2pl(np.array([[0], [1]], dtype=np.int32), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_cycles": 0}, "n_cycles must be at least 1"),
        ({"burnin": -1}, "burnin must be a non-negative integer"),
        ({"burnin": 2.5}, "burnin must be a non-negative integer"),
        ({"proposal_sd": 0.0}, "proposal_sd must be positive"),
        ({"proposal_sd": np.inf}, "proposal_sd must be positive"),
        ({"gain_sequence": "fast"}, "gain_sequence must be 'standard' or 'adaptive'"),
    ],
)
def test_native_mhrm_rejects_invalid_controls(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mhrm_fit_2pl(np.array([[0], [1]], dtype=np.int32), **kwargs)


@native
@pytest.mark.parametrize("args", [(10, 0, 0, 1), (5, 10, 1, 1), (5, 5, 1, 1)])
def test_native_gibbs_kernel_raises_value_errors_instead_of_panicking(
    args: tuple[int, int, int, int],
) -> None:
    from mirt import mirt_rs

    with pytest.raises(ValueError):
        mirt_rs.gibbs_sample_2pl(np.array([[0], [1]], dtype=np.int32), *args)


@native
@pytest.mark.parametrize(
    "args",
    [(0, 0, 0.5, 1, "standard"), (5, 0, 0.0, 1, "standard"), (5, 0, 0.5, 1, "fast")],
)
def test_native_mhrm_kernel_rejects_invalid_controls(
    args: tuple[int, int, float, int, str],
) -> None:
    from mirt import mirt_rs

    with pytest.raises(ValueError):
        mirt_rs.mhrm_fit_2pl(np.array([[0], [1]], dtype=np.int32), *args)


@native
@pytest.mark.parametrize(
    ("n_iter", "burnin", "thin"), [(60, 20, 3), (10, 0, 3), (10, 9, 4), (12, 2, 5)]
)
def test_native_gibbs_keeps_every_thinned_draw(
    n_iter: int, burnin: int, thin: int
) -> None:
    responses = _binary_responses(40, 4, seed=1)
    n_draws = -(-(n_iter - burnin) // thin)

    discrimination, difficulty, theta, log_likelihood = gibbs_sample_2pl(
        responses, n_iter=n_iter, burnin=burnin, thin=thin, seed=5
    )

    assert discrimination.shape == (n_draws, 4)
    assert difficulty.shape == (n_draws, 4)
    assert theta.shape == (n_draws, 40, 1)
    assert log_likelihood.shape == (n_draws,)
    assert np.all(np.isfinite(log_likelihood))


@native
def test_native_gibbs_thinning_matches_unthinned_chain() -> None:
    responses = _binary_responses(60, 5, seed=2)

    full = gibbs_sample_2pl(responses, n_iter=23, burnin=3, thin=1, seed=8)
    thinned = gibbs_sample_2pl(responses, n_iter=23, burnin=3, thin=3, seed=8)

    for complete, kept in zip(full, thinned, strict=True):
        np.testing.assert_array_equal(kept, complete[::3])


@native
def test_native_mcmc_kernels_are_reproducible_for_a_seed() -> None:
    # Large enough for the log-likelihood reduction to run on several threads.
    responses = _binary_responses(3000, 8, seed=3)

    first = gibbs_sample_2pl(responses, n_iter=6, burnin=1, thin=1, seed=13)
    second = gibbs_sample_2pl(responses, n_iter=6, burnin=1, thin=1, seed=13)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)

    first_mhrm = mhrm_fit_2pl(responses, n_cycles=6, burnin=2, seed=13)
    second_mhrm = mhrm_fit_2pl(responses, n_cycles=6, burnin=2, seed=13)
    for left, right in zip(first_mhrm, second_mhrm, strict=True):
        np.testing.assert_array_equal(left, right)


@native
@pytest.mark.parametrize("gain_sequence", ["standard", "adaptive"])
def test_native_mhrm_updates_parameters_during_burnin(gain_sequence: str) -> None:
    responses = _binary_responses(300, 6, seed=4)

    # burnin >= n_cycles keeps the final iterate, which is also the average of
    # the single post-burn-in iterate when burnin == n_cycles - 1.
    final = mhrm_fit_2pl(
        responses, n_cycles=40, burnin=500, seed=21, gain_sequence=gain_sequence
    )
    last = mhrm_fit_2pl(
        responses, n_cycles=40, burnin=39, seed=21, gain_sequence=gain_sequence
    )
    averaged = mhrm_fit_2pl(
        responses, n_cycles=40, burnin=10, seed=21, gain_sequence=gain_sequence
    )

    np.testing.assert_array_equal(final[0], last[0])
    np.testing.assert_array_equal(final[1], last[1])
    assert not np.allclose(final[0], 1.0)
    assert np.std(final[1]) > 0.3
    assert not np.array_equal(averaged[1], final[1])
    assert np.corrcoef(averaged[1], np.linspace(-1.5, 1.5, 6))[0, 1] > 0.95
