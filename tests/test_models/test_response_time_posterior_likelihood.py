"""Contracts for batched response-time posterior likelihood evaluation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import mirt.backends.rust.response_time as response_time_backend
from mirt._rust_backend import RUST_AVAILABLE
from mirt.estimation.rt_gibbs import ResponseTimeGibbsSampler
from mirt.exceptions import MirtValidationError
from mirt.models.response_time import ResponseTimeModel


def _case(
    accuracy_model: str = "2PL",
    *,
    use_rust: bool = False,
) -> tuple[ResponseTimeModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters: dict[str, object] = {
        "n_items": 4,
        "accuracy_model": accuracy_model,
        "discrimination": np.array([0.8, 1.1, 1.4, 0.9]),
        "difficulty": np.array([-0.7, -0.1, 0.6, 1.0]),
        "time_discrimination": np.array([0.9, 1.2, 0.8, 1.4]),
        "time_intensity": np.array([2.7, 3.1, 2.9, 3.4]),
        "use_rust": use_rust,
    }
    if accuracy_model == "3PL":
        parameters["guessing"] = np.array([0.12, 0.18, 0.21, 0.09])
    model = ResponseTimeModel(**parameters)
    responses = np.array(
        [
            [1.0, 0.0, np.nan, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, -1.0],
        ]
    )
    log_rt = np.array(
        [
            [2.5, np.nan, 3.0, 3.2],
            [2.8, 3.3, np.nan, 3.0],
            [np.nan, 3.1, 2.7, 3.5],
        ]
    )
    theta = np.array(
        [
            [-1.0, -0.2, 0.7],
            [-0.7, 0.1, 1.0],
            [-0.4, 0.4, 1.3],
            [-0.1, 0.7, 1.6],
            [0.2, 1.0, 1.9],
        ]
    )
    tau = np.array(
        [
            [0.5, 0.1, -0.3],
            [0.3, -0.1, -0.5],
            [0.1, -0.3, -0.7],
            [-0.1, -0.5, -0.9],
            [-0.3, -0.7, -1.1],
        ]
    )
    return model, responses, log_rt, theta, tau


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_sample_likelihoods_match_individual_evaluation(accuracy_model: str) -> None:
    model, responses, log_rt, theta, tau = _case(accuracy_model)
    expected = np.stack(
        [
            model.joint_log_likelihood(responses, log_rt, theta_row, tau_row)
            for theta_row, tau_row in zip(theta, tau, strict=True)
        ]
    )

    actual = model.joint_log_likelihood_samples(
        responses,
        log_rt,
        theta,
        tau,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_numpy_sample_chunk_size_does_not_change_results() -> None:
    model, responses, log_rt, theta, tau = _case("3PL")
    expected = model.joint_log_likelihood_samples(
        responses,
        log_rt,
        theta,
        tau,
        sample_chunk_size=1,
    )

    for chunk_size in (2, len(theta), len(theta) + 10):
        actual = model.joint_log_likelihood_samples(
            responses,
            log_rt,
            theta,
            tau,
            sample_chunk_size=chunk_size,
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_sample_likelihoods_dispatch_validated_native_inputs(
    monkeypatch: pytest.MonkeyPatch,
    accuracy_model: str,
) -> None:
    calls = 0

    class FakeExtension:
        def rt_joint_log_likelihood_samples(self, *arguments: object) -> np.ndarray:
            nonlocal calls
            calls += 1
            responses, log_rt, theta, tau, _, _, guessing, *_ = arguments
            assert isinstance(responses, np.ndarray)
            assert responses.dtype == np.int32
            assert responses.flags.c_contiguous
            assert isinstance(log_rt, np.ndarray)
            assert log_rt.flags.c_contiguous
            assert isinstance(theta, np.ndarray)
            assert theta.flags.c_contiguous
            assert isinstance(tau, np.ndarray)
            assert tau.flags.c_contiguous
            if accuracy_model == "2PL":
                assert guessing is None
            else:
                assert isinstance(guessing, np.ndarray)
                assert guessing.flags.c_contiguous
            return np.full(theta.shape, 7.0)

    monkeypatch.setattr(response_time_backend, "rust_enabled", lambda: True)
    monkeypatch.setattr(response_time_backend, "mirt_rs", FakeExtension())
    model, responses, log_rt, theta, tau = _case(accuracy_model, use_rust=True)

    actual = model.joint_log_likelihood_samples(
        responses[:, ::-1][:, ::-1],
        log_rt[:, ::-1][:, ::-1],
        theta[:, ::-1][:, ::-1],
        tau[:, ::-1][:, ::-1],
    )

    np.testing.assert_array_equal(actual, np.full(theta.shape, 7.0))
    assert calls == 1


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled extension unavailable")
@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_native_sample_likelihoods_match_numpy(accuracy_model: str) -> None:
    model, responses, log_rt, theta, tau = _case(accuracy_model, use_rust=True)
    native = model.joint_log_likelihood_samples(responses, log_rt, theta, tau)

    model.use_rust = False
    numpy_result = model.joint_log_likelihood_samples(responses, log_rt, theta, tau)

    np.testing.assert_allclose(native, numpy_result, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("theta_transform", "tau_transform", "message"),
    [
        (lambda values: values[0], lambda values: values, "theta must have shape"),
        (lambda values: values[:0], lambda values: values[:0], "theta must have shape"),
        (
            lambda values: values[:, :2],
            lambda values: values[:, :2],
            "theta must have shape",
        ),
        (lambda values: values, lambda values: values[:-1], "same shape"),
        (
            lambda values: np.where(
                np.arange(values.size).reshape(values.shape) == 0,
                np.nan,
                values,
            ),
            lambda values: values,
            "theta must contain only finite",
        ),
    ],
)
def test_sample_likelihoods_validate_latent_matrices(
    theta_transform,
    tau_transform,
    message: str,
) -> None:
    model, responses, log_rt, theta, tau = _case()

    with pytest.raises(MirtValidationError, match=message):
        model.joint_log_likelihood_samples(
            responses,
            log_rt,
            theta_transform(theta),
            tau_transform(tau),
        )


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_sample_likelihoods_validate_chunk_size(chunk_size: object) -> None:
    model, responses, log_rt, theta, tau = _case()

    with pytest.raises(MirtValidationError, match="sample_chunk_size"):
        model.joint_log_likelihood_samples(
            responses,
            log_rt,
            theta,
            tau,
            sample_chunk_size=chunk_size,  # type: ignore[arg-type]
        )


def test_sampler_uses_one_batched_posterior_evaluation() -> None:
    class BatchModel:
        calls = 0

        def joint_log_likelihood_samples(
            self,
            responses: np.ndarray,
            log_rt: np.ndarray,
            theta: np.ndarray,
            tau: np.ndarray,
        ) -> np.ndarray:
            del log_rt
            self.calls += 1
            return theta + tau + np.sum(responses, axis=1)[None, :]

        def joint_log_likelihood(self, *args: object) -> np.ndarray:
            raise AssertionError("individual likelihood path should not run")

    model = BatchModel()
    responses = np.array([[1, 0], [0, 1], [1, 1]])
    log_rt = np.ones_like(responses, dtype=np.float64)
    theta = np.arange(12, dtype=np.float64).reshape(4, 3)
    tau = theta / 10.0

    actual = ResponseTimeGibbsSampler._posterior_log_likelihoods(
        cast(ResponseTimeModel, model),
        responses,
        log_rt,
        theta,
        tau,
    )

    np.testing.assert_array_equal(
        actual,
        theta + tau + np.sum(responses, axis=1)[None, :],
    )
    assert model.calls == 1
