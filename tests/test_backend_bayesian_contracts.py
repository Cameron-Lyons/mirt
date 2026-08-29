from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.backends.rust import diagnostics as backend


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    responses = np.array(
        [
            [1, 0, -1, 1],
            [0, 1, 1, -8],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    discrimination = np.array(
        [
            [0.7, 1.0, 1.3, 1.6],
            [0.8, 1.1, 1.4, 1.7],
            [0.9, 1.2, 1.5, 1.8],
        ]
    )
    difficulty = np.array(
        [
            [-1.0, -0.4, 0.2, 0.8],
            [-0.8, -0.2, 0.4, 1.0],
            [-0.6, 0.0, 0.6, 1.2],
        ]
    )
    theta = np.array(
        [
            [-1.1, 0.0, 1.2],
            [-0.9, 0.2, 1.4],
            [-0.7, 0.4, 1.6],
        ]
    )
    return responses, discrimination, difficulty, theta


def _expected(aggregation: str) -> np.ndarray:
    responses, discrimination, difficulty, theta = _inputs()
    observed = responses >= 0
    logits = discrimination[:, None, :] * (theta[:, :, None] - difficulty[:, None, :])
    values = np.where(
        responses[None, :, :] == 1,
        -np.logaddexp(0.0, -logits),
        -np.logaddexp(0.0, logits),
    )
    values[:, ~observed] = 0.0
    if aggregation == "person":
        return values.sum(axis=2)
    if aggregation == "observation":
        return values.reshape(values.shape[0], -1)
    return values[:, observed]


@pytest.mark.parametrize("aggregation", ["person", "observation", "observed"])
def test_pointwise_numpy_fallback_matches_reference(aggregation: str) -> None:
    inputs = _inputs()
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        actual = backend.compute_pointwise_loglik_2pl(
            *inputs,
            aggregation=aggregation,  # type: ignore[arg-type]
        )
    finally:
        mirt.set_backend(previous)

    np.testing.assert_allclose(actual, _expected(aggregation), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("aggregation", ["person", "observation", "observed"])
def test_pointwise_native_matches_numpy(aggregation: str) -> None:
    inputs = _inputs()
    previous = mirt.get_backend()
    try:
        mirt.set_backend("auto")
        native = backend.compute_pointwise_loglik_2pl(
            *inputs,
            aggregation=aggregation,  # type: ignore[arg-type]
        )
        mirt.set_backend("numpy")
        numpy_result = backend.compute_pointwise_loglik_2pl(
            *inputs,
            aggregation=aggregation,  # type: ignore[arg-type]
        )
    finally:
        mirt.set_backend(previous)

    np.testing.assert_allclose(native, numpy_result, rtol=1e-14, atol=1e-14)


def test_pointwise_backends_preserve_clipped_extreme_likelihoods() -> None:
    responses = np.array([[1, 0], [0, 1]])
    discrimination = np.full((2, 2), 1_000.0)
    difficulty = np.zeros((2, 2))
    theta = np.array([[1.0, -1.0], [2.0, -2.0]])
    previous = mirt.get_backend()
    try:
        mirt.set_backend("auto")
        native = backend.compute_pointwise_loglik_2pl(
            responses,
            discrimination,
            difficulty,
            theta,
            aggregation="observation",
        )
        mirt.set_backend("numpy")
        numpy_result = backend.compute_pointwise_loglik_2pl(
            responses,
            discrimination,
            difficulty,
            theta,
            aggregation="observation",
        )
    finally:
        mirt.set_backend(previous)

    epsilon = 1e-10
    expected = np.array(
        [
            [
                np.log(1.0 - epsilon),
                np.log1p(-(1.0 - epsilon)),
                np.log1p(-epsilon),
                np.log(epsilon),
            ],
            [
                np.log(1.0 - epsilon),
                np.log1p(-(1.0 - epsilon)),
                np.log1p(-epsilon),
                np.log(epsilon),
            ],
        ]
    )
    np.testing.assert_allclose(native, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(numpy_result, expected, rtol=0.0, atol=1e-12)


def test_pointwise_numpy_fallback_bounds_temporary_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = backend.sigmoid
    input_shapes: list[tuple[int, ...]] = []

    def tracked(values: np.ndarray) -> np.ndarray:
        input_shapes.append(values.shape)
        return np.asarray(original(values))

    monkeypatch.setattr(backend, "sigmoid", tracked)
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        actual = backend.compute_pointwise_loglik_2pl(
            *_inputs(),
            aggregation="observation",
        )
    finally:
        mirt.set_backend(previous)

    np.testing.assert_allclose(actual, _expected("observation"))
    assert input_shapes == [(3, 4), (3, 4), (3, 4)]


@pytest.mark.parametrize(
    ("position", "replacement", "message"),
    [
        (0, np.ones(3), "two-dimensional"),
        (0, np.array([[0.5, 1.0, 0.0, 1.0]]), "integer-valued"),
        (0, np.array([[0, 1, 2, 0]]), "only 0 or 1"),
        (1, np.ones((0, 4)), "at least one sample"),
        (1, np.ones((3, 3)), "shape"),
        (2, np.ones((2, 4)), "shape"),
        (3, np.ones((3, 2)), "shape"),
        (3, np.full((3, 3), np.nan), "finite"),
    ],
)
def test_pointwise_backend_validates_inputs(
    position: int,
    replacement: np.ndarray,
    message: str,
) -> None:
    inputs = list(_inputs())
    inputs[position] = replacement

    with pytest.raises(ValueError, match=message):
        backend.compute_pointwise_loglik_2pl(*inputs)


def test_pointwise_backend_validates_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation must be"):
        backend.compute_pointwise_loglik_2pl(
            *_inputs(),
            aggregation="item",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("aggregation", ["person", "observation", "observed"])
def test_pointwise_backend_supports_empty_person_batches(aggregation: str) -> None:
    responses = np.empty((0, 4), dtype=np.int64)
    discrimination = np.ones((3, 4))
    difficulty = np.zeros((3, 4))
    theta = np.empty((3, 0))

    result = backend.compute_pointwise_loglik_2pl(
        responses,
        discrimination,
        difficulty,
        theta,
        aggregation=aggregation,  # type: ignore[arg-type]
    )

    assert result.shape == (3, 0)
