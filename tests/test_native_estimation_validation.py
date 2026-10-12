"""Validation contracts for accelerated estimation entry points."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mirt._rust_backend import RUST_AVAILABLE
from mirt.backends.rust.estimation import bootstrap_fit_2pl, mhrm_fit_2pl

native = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")


def _binary_responses(n_persons: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_persons)
    difficulty = np.linspace(-1.5, 1.5, n_items)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty)))
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    responses[rng.random(responses.shape) < 0.05] = -1
    return responses


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
def test_native_bootstrap_rejects_invalid_em_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    responses = np.array([[0], [1]], dtype=np.int32)

    with pytest.raises(ValueError, match=message):
        bootstrap_fit_2pl(responses, **kwargs)


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
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 0, 0.5, 1, (0.1, 5.0), (-6.0, 6.0), "standard"), "n_cycles"),
        ((5, 0, 0.0, 1, (0.1, 5.0), (-6.0, 6.0), "standard"), "proposal_sd"),
        ((5, 0, 0.5, 1, (0.1, 5.0), (-6.0, 6.0), "fast"), "gain_sequence"),
        ((5, 0, 0.5, 1, (5.0, 0.1), (-6.0, 6.0), "standard"), "disc_bounds"),
        ((5, 0, 0.5, 1, (0.1, 5.0), (np.nan, 6.0), "standard"), "diff_bounds"),
    ],
)
def test_native_mhrm_kernel_rejects_invalid_controls(
    args: tuple[object, ...], message: str
) -> None:
    from mirt import mirt_rs

    with pytest.raises(ValueError, match=message):
        mirt_rs.mhrm_fit_2pl(np.array([[0], [1]], dtype=np.int32), *args)


@native
def test_native_mhrm_kernel_keeps_its_items_in_the_given_box() -> None:
    responses = _binary_responses(400, 6, seed=6)
    discrimination, difficulty, _ = mhrm_fit_2pl(
        responses,
        n_cycles=60,
        burnin=60,
        seed=2,
        disc_bounds=(0.1, 0.8),
        diff_bounds=(-0.5, 0.5),
    )
    assert np.all((discrimination >= 0.1) & (discrimination <= 0.8))
    assert np.all((difficulty >= -0.5) & (difficulty <= 0.5))
    # The items have unit slopes and difficulties out to +-1.5.
    assert np.any(discrimination == 0.8) and np.any(np.abs(difficulty) == 0.5)


@native
def test_native_mhrm_kernel_is_reproducible_for_a_seed() -> None:
    # Large enough for the log-likelihood reduction to run on several threads.
    responses = _binary_responses(3000, 8, seed=3)

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


# A stand-in with the name and module of pyo3's PanicException, a
# BaseException that ``except Exception`` handlers do not catch.
_PanicException = type(
    "PanicException", (BaseException,), {"__module__": "pyo3_runtime"}
)


def _panicking_kernel(*args: Any, **kwargs: Any) -> Any:
    raise _PanicException("index out of bounds")


def test_call_native_reraises_rust_panics_as_runtime_errors() -> None:
    from mirt.backends.rust._helpers import _call_native

    with pytest.raises(RuntimeError, match="index out of bounds"):
        _call_native(_panicking_kernel, 1)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _call_native(interrupted)


@native
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda R: R.bkt_forward(
                np.array([1, 0, 1], dtype=np.int32),
                np.array([0, 3, 1], dtype=np.int32),
                *[np.full(2, 0.2)] * 5,
            ),
            r"skill indices must lie in \[0, 2\)",
        ),
        (
            lambda R: R.bkt_forward_backward_batch(
                np.array([[1, 0, 1]], dtype=np.int32),
                np.array([0, -1, 1], dtype=np.int32),
                *[np.full(2, 0.2)] * 5,
            ),
            "skill indices",
        ),
        (
            lambda R: R.bkt_viterbi(
                np.array([1, 0, 1], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                *[np.full(2, 0.2)] * 5,
            ),
            "skill_assignments must have shape",
        ),
        (
            lambda R: R.bkt_backward(
                np.array([1, 0], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.ones(3),
                *[np.full(2, 0.2)] * 4,
            ),
            "scaling must have shape",
        ),
        (
            lambda R: R.bkt_forward(
                np.array([1, 0], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.full(2, 0.2),
                np.full(3, 0.2),
                *[np.full(2, 0.2)] * 3,
            ),
            "skill parameters",
        ),
        (
            lambda R: R.rt_joint_log_likelihood(
                np.zeros((20, 5), dtype=np.int32),
                np.zeros((20, 5)),
                np.zeros(10),
                np.zeros(20),
                np.ones(5),
                np.zeros(5),
                np.ones(5),
                np.zeros(5),
            ),
            "theta must have shape",
        ),
        (
            lambda R: R.rt_joint_log_likelihood_3pl(
                np.zeros((20, 5), dtype=np.int32),
                np.zeros((20, 4)),
                np.zeros(20),
                np.zeros(20),
                np.ones(5),
                np.zeros(5),
                np.zeros(5),
                np.ones(5),
                np.zeros(5),
            ),
            "log_rt must have shape",
        ),
        (
            lambda R: R.rt_joint_log_likelihood_samples(
                np.zeros((20, 5), dtype=np.int32),
                np.zeros((20, 5)),
                np.zeros((3, 20)),
                np.zeros((3, 20)),
                np.ones(3),
                np.zeros(5),
                None,
                np.ones(5),
                np.zeros(5),
            ),
            "discrimination must have shape",
        ),
        (
            lambda R: R.rt_accept_person_proposals(
                np.zeros((20, 5), dtype=np.int32),
                np.zeros((20, 5)),
                *[np.zeros(20)] * 5,
                np.ones(5),
                np.zeros(5),
                np.ones(5),
                np.zeros(5),
                np.zeros(2),
                np.eye(3),
                0.0,
                None,
            ),
            "covariance_inverse must have shape",
        ),
        (
            lambda R: R.cat_simulate_batch(
                np.zeros(2),
                np.ones(4),
                np.zeros(3),
                np.zeros(3),
                np.ones(3) / 3,
                0.3,
                3,
                1,
                2,
                1,
            ),
            "same length",
        ),
        (
            lambda R: R.cat_conditional_mse(
                np.zeros(2),
                np.ones(3),
                np.zeros(3),
                np.zeros(3),
                np.ones(3) / 3,
                0.3,
                3,
                1,
                -1,
                1,
            ),
            "n_replications must be non-negative",
        ),
    ],
)
def test_native_kernels_raise_value_errors_instead_of_panicking(
    call: Callable, message: str
) -> None:
    # Regression: these indexed out of bounds and raised pyo3's
    # PanicException, which ``except Exception`` fallbacks do not catch.
    from mirt import mirt_rs

    with pytest.raises(ValueError, match=message):
        call(mirt_rs)


def test_em_falls_back_when_the_native_3pl_iteration_panics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the PanicException passed through ``except Exception`` in
    # EMEstimator instead of falling back to the NumPy iteration.
    import types

    import mirt.backends.rust.estimation as native_estimation
    import mirt.estimation.em as em_module
    from mirt import simdata
    from mirt.estimation.em import EMEstimator
    from mirt.models.dichotomous import ThreeParameterLogistic

    data = np.asarray(simdata(n_persons=200, n_items=5, model="3PL", seed=4))
    options = {"n_quadpts": 15, "max_iter": 5, "compute_standard_errors": False}
    expected = EMEstimator(use_rust=False, **options).fit(
        ThreeParameterLogistic(5), data
    )

    calls = []

    def panicking(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        _panicking_kernel()

    monkeypatch.setattr(
        native_estimation, "mirt_rs", types.SimpleNamespace(em_iteration_3pl=panicking)
    )
    monkeypatch.setattr(native_estimation, "rust_enabled", lambda: True)
    monkeypatch.setattr(em_module, "RUST_AVAILABLE", True)
    monkeypatch.setattr(em_module, "should_use_rust", lambda use_rust: use_rust)
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: False)
    actual = EMEstimator(**options).fit(ThreeParameterLogistic(5), data)

    assert calls
    for name, values in expected.model.parameters.items():
        np.testing.assert_allclose(actual.model.parameters[name], values)


def test_bkt_model_falls_back_when_a_native_kernel_panics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import mirt.backends.rust.dynamic as native_dynamic
    from mirt.models.dynamic import BKTModel

    model = BKTModel(n_skills=2)
    responses = np.array([1, 0, 1, 1])
    skills = np.array([0, 1, 0, 1])
    expected = model._forward_python(responses, skills)

    monkeypatch.setattr(
        native_dynamic, "mirt_rs", types.SimpleNamespace(bkt_forward=_panicking_kernel)
    )
    monkeypatch.setattr(native_dynamic, "rust_enabled", lambda: True)
    monkeypatch.setattr(model, "_can_use_native_inference", lambda: True)
    alpha, scaling = model.forward(responses, skills)

    np.testing.assert_allclose(alpha, expected[0])
    np.testing.assert_allclose(scaling, expected[1])


def _three_pl_arguments(**changes: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "responses": np.array([[0, 1]], dtype=np.int32),
        "quad_points": np.array([0.0, 1.0]),
        "quad_weights": np.array([0.5, 0.5]),
        "discrimination": np.array([1.0, 1.0]),
        "difficulty": np.array([0.0, 0.0]),
        "guessing": np.array([0.1, 0.1]),
    }
    arguments.update(changes)
    return arguments


@native
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"quad_weights": np.array([1.0])}, "quad_weights must have shape"),
        ({"quad_points": np.array([]), "quad_weights": np.array([])}, "non-empty"),
        ({"discrimination": np.array([1.0])}, "discrimination must have shape"),
        ({"prior_var": 0.0}, "prior_var finite and positive"),
        ({"prior_mean": np.nan}, "prior_mean must be finite"),
        ({"max_m_iter": 0}, "max_m_iter must be a positive integer"),
        ({"disc_bounds": (5.0, 0.1)}, "disc_bounds must satisfy lower <= upper"),
        ({"guess_bounds": (np.nan, 0.3)}, "guess_bounds must satisfy"),
        ({"responses": np.array([[0, 2]], dtype=np.int32)}, "only 0 or 1"),
    ],
)
def test_native_3pl_iteration_rejects_inconsistent_inputs(
    changes: dict[str, Any], message: str
) -> None:
    # Regression: these reached assert_eq!, an index panic or f64::clamp.
    from mirt.backends.rust.estimation import em_iteration_3pl

    with pytest.raises(ValueError, match=message):
        em_iteration_3pl(**_three_pl_arguments(**changes))


@native
@pytest.mark.parametrize(
    "native_fit",
    [
        lambda data: bootstrap_fit_2pl(data, n_bootstrap=1, n_quadpts=5, max_iter=2),
        lambda data: mhrm_fit_2pl(data, n_cycles=5, burnin=1, seed=1),
    ],
)
def test_native_2pl_kernels_reject_non_binary_codes(native_fit: Callable) -> None:
    data = _binary_responses(30, 3, seed=5)
    data[0, 0] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        native_fit(data)


@native
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda R: R.bkt_forward(
                np.array([1, 0, 1]), np.array([0, 3, 1]), *[np.full(2, 0.2)] * 5
            ),
            "skill indices",
        ),
        (
            lambda R: R.bkt_forward_backward_batch(
                np.array([[1, 0, 1]]), np.array([0, -1, 1]), *[np.full(2, 0.2)] * 5
            ),
            "skill indices",
        ),
        (
            lambda R: R.bkt_viterbi(
                np.array([1, 0, 1]), np.array([0, 1]), *[np.full(2, 0.2)] * 5
            ),
            "skill_assignments must have shape",
        ),
        (
            lambda R: R.bkt_backward(
                np.array([1, 0]),
                np.array([0, 1]),
                np.ones(3),
                *[np.full(2, 0.2)] * 4,
            ),
            "scaling must have shape",
        ),
        (
            lambda R: R.bkt_forward(
                np.array([1, 0]),
                np.array([0, 1]),
                np.full(2, 0.2),
                np.full(3, 0.2),
                *[np.full(2, 0.2)] * 3,
            ),
            "skill parameters",
        ),
    ],
)
def test_native_wrappers_reject_inconsistent_shapes(
    call: Callable, message: str
) -> None:
    # Regression: these indexed out of bounds and panicked in Rust.
    import mirt.backends.rust as R

    with pytest.raises(ValueError, match=message):
        call(R)


@pytest.mark.parametrize("use_rust", [False, pytest.param(True, marks=native)])
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"theta": np.zeros(10)}, "theta must have shape"),
        ({"discrimination": np.ones(3)}, "discrimination must have shape"),
        ({"log_rt": np.zeros((20, 4))}, "log_rt must have shape"),
        ({"guessing": np.zeros(4)}, "guessing must have shape"),
    ],
)
def test_response_time_wrappers_reject_inconsistent_shapes(
    use_rust: bool, changes: dict[str, Any], message: str
) -> None:
    from mirt.backends.rust.response_time import (
        rt_accept_person_proposals,
        rt_joint_log_likelihood,
    )

    rng = np.random.default_rng(0)
    arguments: dict[str, Any] = {
        "responses": rng.integers(0, 2, (20, 5)),
        "log_rt": rng.normal(size=(20, 5)),
        "theta": np.zeros(20),
        "tau": np.zeros(20),
        "discrimination": np.ones(5),
        "difficulty": np.zeros(5),
        "time_discrimination": np.ones(5),
        "time_intensity": np.zeros(5),
        "guessing": None,
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        rt_joint_log_likelihood(**arguments, use_rust=use_rust)
    with pytest.raises(ValueError, match=message):
        rt_accept_person_proposals(
            arguments["responses"],
            arguments["log_rt"],
            arguments["theta"],
            arguments["tau"],
            np.zeros(20),
            np.zeros(20),
            np.zeros(20),
            arguments["discrimination"],
            arguments["difficulty"],
            arguments["time_discrimination"],
            arguments["time_intensity"],
            np.zeros(2),
            np.eye(2),
            0.0,
            arguments["guessing"],
            use_rust=use_rust,
        )
