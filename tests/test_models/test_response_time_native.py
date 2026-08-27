"""Compiled response-time inference dispatch and parity tests."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
import mirt.backends.rust.response_time as response_time_backend
import mirt.estimation.rt_gibbs as rt_gibbs_module
from mirt._rust_backend import RUST_AVAILABLE
from mirt.estimation.rt_gibbs import ResponseTimeGibbsSampler, RTModelPriors
from mirt.exceptions import MirtValidationError
from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult


def _inputs(
    accuracy_model: str = "2PL",
) -> tuple[ResponseTimeModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters: dict[str, object] = {
        "n_items": 3,
        "accuracy_model": accuracy_model,
        "discrimination": np.array([1.2, 0.8, 1.5]),
        "difficulty": np.array([-0.4, 0.2, 0.9]),
        "time_discrimination": np.array([0.9, 1.3, 0.7]),
        "time_intensity": np.array([2.8, 3.2, 2.5]),
    }
    if accuracy_model == "3PL":
        parameters["guessing"] = np.array([0.15, 0.22, 0.18])
    model = ResponseTimeModel(**parameters)
    responses = np.array([[1.0, 0.0, np.nan], [-1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    log_rt = np.array([[2.6, np.nan, 2.4], [3.0, 3.5, np.nan], [2.9, 3.1, 2.2]])
    theta = np.array([-0.8, 0.1, 1.2])
    tau = np.array([0.2, -0.4, 0.6])
    return model, responses, log_rt, theta, tau


def test_response_time_api_is_available_lazily() -> None:
    assert mirt.ResponseTimeModel is ResponseTimeModel
    assert mirt.ResponseTimeResult is ResponseTimeResult
    assert mirt.ResponseTimeGibbsSampler is ResponseTimeGibbsSampler
    assert mirt.RTModelPriors is RTModelPriors
    assert mirt.models.ResponseTimeModel is ResponseTimeModel
    assert mirt.estimation.ResponseTimeGibbsSampler is ResponseTimeGibbsSampler


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ResponseTimeModel(1, use_rust="yes"),
        lambda: ResponseTimeGibbsSampler(use_rust="yes"),
    ],
)
def test_use_rust_must_be_boolean(factory) -> None:
    with pytest.raises(MirtValidationError, match="use_rust"):
        factory()


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_model_dispatches_validated_contiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
    accuracy_model: str,
) -> None:
    calls: list[str] = []

    class FakeExtension:
        def rt_joint_log_likelihood(self, *arguments):
            calls.append("2PL")
            self._validate(arguments)
            return np.array([10.0, 20.0, 30.0])

        def rt_joint_log_likelihood_3pl(self, *arguments):
            calls.append("3PL")
            self._validate((*arguments[:6], *arguments[7:]))
            assert arguments[6].flags.c_contiguous
            return np.array([10.0, 20.0, 30.0])

        @staticmethod
        def _validate(arguments) -> None:
            responses, log_rt, theta, tau, *_ = arguments
            assert responses.dtype == np.int32
            assert responses.flags.c_contiguous
            assert log_rt.flags.c_contiguous
            assert theta.flags.c_contiguous
            assert tau.flags.c_contiguous
            assert responses[0, 2] == -1

    monkeypatch.setattr(response_time_backend, "rust_enabled", lambda: True)
    monkeypatch.setattr(response_time_backend, "mirt_rs", FakeExtension())
    model, responses, log_rt, theta, tau = _inputs(accuracy_model)

    result = model.joint_log_likelihood(
        responses[:, ::-1][:, ::-1],
        log_rt[:, ::-1][:, ::-1],
        theta[::-1][::-1],
        tau[::-1][::-1],
    )

    np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])
    assert calls == [accuracy_model]


def test_per_model_numpy_control_skips_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingExtension:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected extension call: {name}")

    monkeypatch.setattr(response_time_backend, "rust_enabled", lambda: True)
    monkeypatch.setattr(response_time_backend, "mirt_rs", FailingExtension())
    model, responses, log_rt, theta, tau = _inputs("3PL")
    model.use_rust = False

    result = model.joint_log_likelihood(responses, log_rt, theta, tau)

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_global_numpy_control_skips_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingExtension:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected extension call: {name}")

    monkeypatch.setattr(response_time_backend, "mirt_rs", FailingExtension())
    model, responses, log_rt, theta, tau = _inputs()
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        result = model.joint_log_likelihood(responses, log_rt, theta, tau)
    finally:
        mirt.set_backend(previous)

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled extension unavailable")
@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_compiled_joint_likelihood_matches_numpy(accuracy_model: str) -> None:
    model, responses, log_rt, theta, tau = _inputs(accuracy_model)

    compiled = model.joint_log_likelihood(responses, log_rt, theta, tau)
    model.use_rust = False
    numpy_result = model.joint_log_likelihood(responses, log_rt, theta, tau)

    np.testing.assert_allclose(compiled, numpy_result, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled extension unavailable")
@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_compiled_person_proposals_match_numpy(accuracy_model: str) -> None:
    model, responses, log_rt, theta, tau = _inputs(accuracy_model)
    response_integers = np.where(
        np.isfinite(responses) & (responses >= 0), responses, -1
    )
    theta_proposed = theta + np.array([0.1, -0.2, 0.3])
    tau_proposed = tau + np.array([-0.15, 0.25, -0.05])
    log_uniform = np.log(np.array([0.2, 0.8, 0.4]))
    covariance_inverse = np.linalg.inv(model.ability_speed_cov)
    log_determinant = np.linalg.slogdet(model.ability_speed_cov)[1]
    arguments = (
        response_integers,
        log_rt,
        theta,
        tau,
        theta_proposed,
        tau_proposed,
        log_uniform,
        model.discrimination,
        model.difficulty,
        model.time_discrimination,
        model.time_intensity,
        model.ability_speed_mean,
        covariance_inverse,
        log_determinant,
        model.guessing,
    )

    compiled = response_time_backend.rt_accept_person_proposals(
        *arguments, use_rust=True
    )
    numpy_result = response_time_backend.rt_accept_person_proposals(
        *arguments, use_rust=False
    )

    np.testing.assert_allclose(compiled[0], numpy_result[0])
    np.testing.assert_allclose(compiled[1], numpy_result[1])
    np.testing.assert_array_equal(compiled[2], numpy_result[2])


def test_sampler_dispatches_with_shared_random_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_accept(*arguments, use_rust):
        captured["theta_proposed"] = arguments[4]
        captured["tau_proposed"] = arguments[5]
        captured["log_uniform"] = arguments[6]
        captured["use_rust"] = use_rust
        return arguments[4], arguments[5], np.ones(2, dtype=bool)

    monkeypatch.setattr(rt_gibbs_module, "rt_accept_person_proposals", fake_accept)
    sampler = ResponseTimeGibbsSampler(use_rust=False)
    rng = np.random.default_rng(17)
    theta = np.array([-0.2, 0.4])
    tau = np.array([0.1, -0.3])
    result = sampler._sample_person_params(
        np.array([[1, 0], [0, 1]], dtype=np.int32),
        np.zeros((2, 2)),
        theta,
        tau,
        np.ones(2),
        np.zeros(2),
        None,
        np.ones(2),
        np.zeros(2),
        np.zeros(2),
        np.eye(2),
        0.25,
        rng,
    )

    np.testing.assert_array_equal(result[0], captured["theta_proposed"])
    np.testing.assert_array_equal(result[1], captured["tau_proposed"])
    np.testing.assert_array_equal(result[2], np.ones(2, dtype=bool))
    assert np.asarray(captured["log_uniform"]).shape == (2,)
    assert captured["use_rust"] is False
