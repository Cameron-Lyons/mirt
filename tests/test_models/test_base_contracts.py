"""Shared model-state contract tests."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.models.base as base_module
from mirt.exceptions import MirtValidationError
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import (
    OneParameterLogistic,
    TwoParameterLogistic,
)


def test_parameter_updates_detach_caller_owned_arrays() -> None:
    model = TwoParameterLogistic(n_items=2)
    difficulty = np.array([-0.5, 0.5])

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 99.0

    assert_array_equal(model.parameters["difficulty"], [-0.5, 0.5])


def test_multi_parameter_update_is_atomic() -> None:
    model = TwoParameterLogistic(n_items=2)
    before = model.parameters

    with pytest.raises(MirtValidationError, match="Shape mismatch"):
        model.set_parameters(
            difficulty=np.array([-1.0, 1.0]),
            discrimination=np.ones(3),
        )

    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("item_idx", [True, np.bool_(False), 0.0])
def test_item_parameter_access_requires_integer_indices(item_idx: object) -> None:
    model = TwoParameterLogistic(n_items=2)

    with pytest.raises(IndexError, match="must be an integer"):
        model.get_item_parameters(item_idx)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="must be an integer"):
        model.set_item_parameter(  # type: ignore[arg-type]
            item_idx,
            "difficulty",
            1.0,
        )


def test_item_update_preserves_subclass_parameter_rules() -> None:
    model = OneParameterLogistic(n_items=2)
    before = model.parameters["discrimination"]

    model.set_item_parameter(0, "discrimination", 1.0)
    with pytest.raises(ValueError, match="Cannot set discrimination"):
        model.set_item_parameter(0, "discrimination", 2.0)

    assert_array_equal(model.parameters["discrimination"], before)


def test_simulation_matches_seeded_probability_draws_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TwoParameterLogistic(n_items=5).set_parameters(
        discrimination=np.array([0.7, 0.9, 1.1, 1.3, 1.5]),
        difficulty=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    )
    theta = np.linspace(-2.0, 2.0, 37)
    probabilities = model.probability(theta)
    expected = (
        np.random.default_rng(20260829).random(probabilities.shape) < probabilities
    ).astype(np.int32)
    monkeypatch.setattr(
        base_module,
        "_DICHOTOMOUS_MAX_PROBABILITY_VALUES",
        3 * model.n_items,
    )

    unchunked = model.simulate(theta, seed=20260829, chunk_size=len(theta))
    chunked = model.simulate(theta, seed=20260829, chunk_size=7)
    automatic = model.simulate(theta, seed=20260829)

    assert unchunked.shape == probabilities.shape
    assert unchunked.dtype == np.int32
    assert_array_equal(unchunked, expected)
    assert_array_equal(chunked, expected)
    assert_array_equal(automatic, expected)


def test_simulation_matches_empirical_response_probabilities() -> None:
    model = TwoParameterLogistic(n_items=4).set_parameters(
        discrimination=np.array([0.6, 0.9, 1.2, 1.5]),
        difficulty=np.array([-0.8, -0.2, 0.4, 1.0]),
    )
    theta = np.full(50_000, 0.25)

    responses = model.simulate(theta, seed=712, chunk_size=997)

    assert_allclose(
        responses.mean(axis=0),
        model.probability(theta[:1])[0],
        atol=0.008,
    )


def test_simulation_supports_multidimensional_models() -> None:
    model = BifactorModel(
        n_items=5,
        specific_factors=[10, 10, 42, 42, 99],
    ).set_parameters(
        general_loadings=np.array([0.5, 0.7, 0.4, 0.6, 0.9]),
        specific_loadings=np.array([0.3, 0.2, 0.8, 0.5, 0.4]),
        intercepts=np.array([-0.4, 0.2, 0.0, 0.5, -0.2]),
    )
    theta = np.random.default_rng(8).normal(size=(41, model.n_factors))
    probabilities = model.probability(theta)
    expected = (
        np.random.default_rng(91).random(probabilities.shape) < probabilities
    ).astype(np.int32)

    responses = model.simulate(theta, seed=91, chunk_size=6)

    assert_array_equal(responses, expected)


def test_simulation_accepts_single_person_unidimensional_theta() -> None:
    model = OneParameterLogistic(n_items=3)

    responses = model.simulate(np.array([0.3]), seed=17)

    assert responses.shape == (1, 3)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_simulation_rejects_invalid_chunk_sizes(chunk_size: object) -> None:
    model = OneParameterLogistic(n_items=2)

    with pytest.raises(ValueError, match="chunk_size"):
        model.simulate(
            np.array([0.0, 1.0]),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )
