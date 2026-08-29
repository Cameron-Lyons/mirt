"""Bounded-memory contracts for random-testlet likelihoods."""

from collections.abc import Callable

import numpy as np
import pytest

import mirt.models.testlet as testlet_module
from mirt.exceptions import MirtValidationError
from mirt.models.testlet import RandomTestletEffectsModel


def _model() -> RandomTestletEffectsModel:
    model = RandomTestletEffectsModel(
        7,
        [2, 2, 2, 8, 8, -1, -1],
        n_quadpts=9,
    )
    model.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.4, 0.8, 1.3, 0.9, 1.2]),
        testlet_loadings=np.array([0.6, -0.3, 0.8, 0.4, 0.7, 0.0, 0.0]),
        difficulty=np.array([-0.8, 0.1, 0.6, -0.4, 0.9, 0.2, -0.2]),
        testlet_variances=np.array([0.7, 1.2]),
    )
    return model


def _responses() -> np.ndarray:
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, -1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, -1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1],
        ]
    )


def test_paired_likelihood_is_invariant_to_automatic_and_explicit_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    responses = _responses()
    theta = np.linspace(-1.5, 1.5, len(responses))
    expected = model.integrate_out_testlet_effects(responses, theta)

    monkeypatch.setattr(testlet_module, "_TESTLET_LIKELIHOOD_TARGET_ELEMENTS", 20)
    automatic = model.integrate_out_testlet_effects(responses, theta)
    one_at_a_time = model.log_likelihood(responses, theta, chunk_size=1)

    np.testing.assert_array_equal(automatic, expected)
    np.testing.assert_array_equal(one_at_a_time, expected)


def test_likelihood_grid_is_invariant_to_two_dimensional_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    responses = _responses()
    theta_grid = np.linspace(-2.0, 2.0, 13)
    expected = model.log_likelihood_batch(responses, theta_grid)

    monkeypatch.setattr(testlet_module, "_TESTLET_LIKELIHOOD_TARGET_ELEMENTS", 20)
    automatic = model.log_likelihood_batch(responses, theta_grid)
    one_pattern_at_a_time = model.log_likelihood_batch(
        responses,
        theta_grid,
        chunk_size=1,
    )

    np.testing.assert_allclose(automatic, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(
        one_pattern_at_a_time,
        expected,
        rtol=1e-15,
        atol=1e-15,
    )


def test_all_standalone_and_empty_inputs_support_chunks() -> None:
    model = RandomTestletEffectsModel(3, [-1, -1, -1])
    responses = np.array([[1, 0, -1], [0, 1, 1], [1, 1, 0]])
    theta = np.array([-0.5, 0.0, 0.8])

    expected = model.integrate_out_testlet_effects(responses, theta)
    chunked = model.integrate_out_testlet_effects(responses, theta, chunk_size=1)

    np.testing.assert_array_equal(chunked, expected)
    assert model.integrate_out_testlet_effects(
        np.empty((0, 3), dtype=int),
        np.empty(0),
    ).shape == (0,)
    assert model.log_likelihood_batch(responses, np.empty(0)).shape == (3, 0)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, np.bool_(False)])
@pytest.mark.parametrize(
    "evaluate",
    [
        lambda model, responses, theta, chunk_size: model.integrate_out_testlet_effects(
            responses,
            theta,
            chunk_size=chunk_size,
        ),
        lambda model, responses, theta, chunk_size: model.log_likelihood(
            responses,
            theta,
            chunk_size=chunk_size,
        ),
        lambda model, responses, theta, chunk_size: model.log_likelihood_batch(
            responses,
            theta,
            chunk_size=chunk_size,
        ),
    ],
)
def test_likelihood_methods_reject_invalid_chunk_sizes(
    evaluate: Callable[
        [RandomTestletEffectsModel, np.ndarray, np.ndarray, object], np.ndarray
    ],
    chunk_size: object,
) -> None:
    model = RandomTestletEffectsModel(2, [0, 0])
    responses = np.array([[1, 0], [0, 1]])
    theta = np.array([-0.5, 0.5])

    with pytest.raises(MirtValidationError, match="chunk_size"):
        evaluate(model, responses, theta, chunk_size)
