"""Tests for excess-zero response simulation."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mirt import (
    HurdleIRT,
    ZeroInflated2PL,
    ZeroInflated3PL,
    generate_item_parameters,
    simdata,
)


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("ZI-2PL", ZeroInflated2PL),
        ("ZI-3PL", ZeroInflated3PL),
        ("HURDLE", HurdleIRT),
    ],
)
def test_excess_zero_simulation_matches_model_probabilities(
    model_name: str,
    model_class: type[ZeroInflated2PL | ZeroInflated3PL | HurdleIRT],
) -> None:
    n_persons = 200_000
    n_items = 2
    theta = np.zeros(n_persons)
    common = {
        "discrimination": np.array([1.3, 0.8]),
        "difficulty": np.array([-0.4, 0.7]),
    }
    if model_name == "ZI-2PL":
        extra = {"zero_inflation": np.array([0.3, 0.1])}
    elif model_name == "ZI-3PL":
        extra = {
            "guessing": np.array([0.2, 0.35]),
            "zero_inflation": np.array([0.3, 0.1]),
        }
    else:
        extra = {
            "engagement_intercept": np.array([-0.2, 1.1]),
            "engagement_slope": np.array([0.7, -0.3]),
        }

    responses, structural = simdata(
        model=model_name,
        theta=theta,
        n_items=n_items,
        return_structural_zeros=True,
        seed=2026,
        **common,
        **extra,
    )
    model = model_class(n_items=n_items)
    model.set_parameters(**common, **extra)

    expected_responses = model.probability(np.array([0.0]))[0]
    expected_structural = model.structural_zero_probability(np.array([0.0]))[0]
    expected_posterior = model.structural_zero_posterior(np.array([0.0]))[0]
    np.testing.assert_allclose(responses.mean(axis=0), expected_responses, atol=0.004)
    np.testing.assert_allclose(structural.mean(axis=0), expected_structural, atol=0.004)
    np.testing.assert_allclose(
        structural.sum(axis=0) / (responses == 0).sum(axis=0),
        expected_posterior,
        atol=0.004,
    )
    assert not np.any((responses == 1) & structural)


@pytest.mark.parametrize("model", ["ZI-2PL", "ZI-3PL", "HURDLE"])
def test_excess_zero_simulation_is_reproducible_and_labels_are_optional(
    model: str,
) -> None:
    parameters = generate_item_parameters(n_items=4, model=model, seed=123)
    kwargs = {
        "model": model,
        "n_persons": 80,
        "n_items": 4,
        "seed": 456,
        **parameters,
    }

    responses_only = simdata(**kwargs)
    responses, structural = simdata(**kwargs, return_structural_zeros=True)
    repeated_responses, repeated_structural = simdata(
        **kwargs, return_structural_zeros=True
    )

    np.testing.assert_array_equal(responses_only, responses)
    np.testing.assert_array_equal(repeated_responses, responses)
    np.testing.assert_array_equal(repeated_structural, structural)
    assert responses.shape == structural.shape == (80, 4)
    assert np.issubdtype(responses.dtype, np.integer)
    assert np.issubdtype(structural.dtype, np.bool_)


@pytest.mark.parametrize(
    ("model", "expected_keys"),
    [
        (
            "ZI-2PL",
            {"discrimination", "difficulty", "zero_inflation"},
        ),
        (
            "ZI-3PL",
            {"discrimination", "difficulty", "guessing", "zero_inflation"},
        ),
        (
            "HURDLE",
            {
                "discrimination",
                "difficulty",
                "engagement_intercept",
                "engagement_slope",
            },
        ),
    ],
)
def test_generated_excess_zero_parameters_round_trip(
    model: str,
    expected_keys: set[str],
) -> None:
    first = generate_item_parameters(n_items=5, model=model, seed=91)
    second = generate_item_parameters(n_items=5, model=model, seed=91)

    assert set(first) == expected_keys
    for name in expected_keys:
        assert first[name].shape == (5,)
        np.testing.assert_array_equal(first[name], second[name])

    responses = simdata(
        model=model,
        n_persons=30,
        n_items=5,
        seed=92,
        **first,
    )
    assert responses.shape == (30, 5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"model": "ZI-2PL", "zero_inflation": np.array([0.1])},
            "zero_inflation must have shape (2,)",
        ),
        (
            {"model": "ZI-3PL", "zero_inflation": np.array([0.1, 1.0])},
            "zero_inflation values must be in [0, 1)",
        ),
        (
            {"model": "ZI-3PL", "guessing": np.array([0.2, 1.0])},
            "guessing values must be in [0, 1)",
        ),
        (
            {"model": "HURDLE", "engagement_slope": np.array([0.2])},
            "engagement_slope must have shape (2,)",
        ),
        (
            {
                "model": "HURDLE",
                "engagement_intercept": np.array([0.2, np.inf]),
            },
            "engagement_intercept must contain finite values",
        ),
        (
            {"model": "ZI-2PL", "theta": np.zeros((3, 2))},
            "ZI-2PL simulation only supports unidimensional theta",
        ),
        (
            {"model": "HURDLE", "discrimination": np.array([1.0, 0.0])},
            "HURDLE discrimination must be strictly positive",
        ),
    ],
)
def test_excess_zero_simulation_validates_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        simdata(n_items=2, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "2PL", "zero_inflation": np.array([0.1, 0.2])},
        {"model": "ZI-2PL", "engagement_slope": np.array([0.1, 0.2])},
        {"model": "HURDLE", "guessing": np.array([0.1, 0.2])},
        {"model": "ZI-2PL", "upper": np.array([0.9, 0.9])},
        {"model": "HURDLE", "thresholds": np.zeros((2, 1))},
    ],
)
def test_excess_zero_parameters_are_family_specific(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        simdata(n_items=2, **kwargs)


def test_structural_zero_labels_require_an_excess_zero_model() -> None:
    with pytest.raises(ValueError, match="only valid for ZI-2PL, ZI-3PL, and HURDLE"):
        simdata(model="2PL", return_structural_zeros=True)

    with pytest.raises(ValueError, match="must be a boolean"):
        simdata(model="ZI-2PL", return_structural_zeros=1)


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        ("ZI-2PL", {"zero_inflation": np.array([0.2])}),
        (
            "ZI-3PL",
            {
                "zero_inflation": np.array([0.2]),
                "guessing": np.array([0.25]),
            },
        ),
        (
            "HURDLE",
            {
                "engagement_intercept": np.array([0.0]),
                "engagement_slope": np.array([np.finfo(np.float64).max]),
            },
        ),
    ],
)
def test_excess_zero_simulation_is_stable_at_float_limits(
    model: str,
    extra: dict[str, np.ndarray],
) -> None:
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        responses, structural = simdata(
            model=model,
            theta=np.array([-maximum, maximum]),
            n_items=1,
            discrimination=np.array([maximum]),
            difficulty=np.array([0.0]),
            return_structural_zeros=True,
            seed=7,
            **extra,
        )

    np.testing.assert_array_equal(responses.ravel(), np.array([0, 1]))
    assert not np.any((responses == 1) & structural)
