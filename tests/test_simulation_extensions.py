"""Tests for extended polytomous response simulation."""

import numpy as np
import pytest

import mirt._rust_backend as rust_backend
import mirt.utils.simulation as simulation
from mirt import generate_item_parameters, simdata


def test_pcm_generated_parameters_round_trip_reproducibly():
    first = generate_item_parameters(
        n_items=4,
        model="PCM",
        n_categories=5,
        seed=12,
    )
    second = generate_item_parameters(
        n_items=4,
        model="PCM",
        n_categories=5,
        seed=12,
    )

    assert set(first) == {"discrimination", "steps"}
    np.testing.assert_array_equal(first["discrimination"], np.ones(4))
    np.testing.assert_array_equal(first["steps"], second["steps"])

    responses = simdata(
        model="PCM",
        n_persons=100,
        n_items=4,
        n_categories=5,
        seed=99,
        **first,
    )
    assert responses.shape == (100, 4)
    assert np.all((responses >= 0) & (responses < 5))


def test_nrm_generated_parameters_support_multidimensional_simulation():
    parameters = generate_item_parameters(
        n_items=3,
        model="NRM",
        n_categories=4,
        n_factors=2,
        seed=21,
    )

    assert parameters["slopes"].shape == (3, 4, 2)
    assert parameters["intercepts"].shape == (3, 4)
    np.testing.assert_array_equal(parameters["slopes"][:, 0], 0.0)
    np.testing.assert_array_equal(parameters["intercepts"][:, 0], 0.0)

    responses = simdata(
        model="NRM",
        n_persons=80,
        n_items=3,
        n_categories=4,
        n_factors=2,
        seed=22,
        **parameters,
    )
    assert responses.shape == (80, 3)
    assert np.all((responses >= 0) & (responses < 4))


@pytest.mark.parametrize(("model", "n_factors"), [("PCM", 1), ("NRM", 2)])
def test_extended_models_are_seeded(model, n_factors):
    kwargs = {
        "model": model,
        "n_persons": 50,
        "n_items": 3,
        "n_categories": 4,
        "n_factors": n_factors,
        "seed": 90210,
    }

    first = simdata(**kwargs)
    second = simdata(**kwargs)

    np.testing.assert_array_equal(first, second)


def test_nrm_uses_supplied_category_parameters():
    theta = np.column_stack([np.full(20, -100.0), np.full(20, 100.0)])
    slopes = np.array(
        [
            [[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]],
            [[0.0, 0.0], [-1.0, 1.0], [1.0, -1.0]],
        ]
    )

    responses = simdata(
        model="NRM",
        theta=theta,
        n_items=2,
        n_categories=3,
        slopes=slopes,
        intercepts=np.zeros((2, 3)),
        seed=88,
    )

    np.testing.assert_array_equal(responses[:, 0], np.full(20, 2))
    np.testing.assert_array_equal(responses[:, 1], np.full(20, 1))


@pytest.mark.parametrize("model", ["GPCM", "PCM"])
def test_partial_credit_logits_are_stable_at_float_limits(model):
    maximum = np.finfo(np.float64).max
    kwargs = {"discrimination": np.ones(1)} if model == "GPCM" else {}

    with np.errstate(over="raise", invalid="raise"):
        responses = simdata(
            model=model,
            theta=np.array([-maximum, maximum]),
            n_items=1,
            n_categories=3,
            steps=np.zeros((1, 2)),
            seed=42,
            **kwargs,
        )

    np.testing.assert_array_equal(responses.ravel(), np.array([0, 2]))


def test_nrm_logits_are_stable_at_float_limits():
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        responses = simdata(
            model="NRM",
            theta=np.array([-maximum, maximum]),
            n_items=1,
            n_categories=3,
            slopes=np.array([[0.0, maximum, -maximum]]),
            intercepts=np.zeros((1, 3)),
            seed=42,
        )

    np.testing.assert_array_equal(responses.ravel(), np.array([2, 1]))


def test_pcm_dispatches_unit_discrimination_to_accelerated_wrapper(monkeypatch):
    captured = {}
    expected = np.full((3, 2), 2, dtype=np.int_)

    def fake_simulate_gpcm(theta, discrimination, steps, seed=None):
        captured["theta"] = theta
        captured["discrimination"] = discrimination
        captured["steps"] = steps
        return expected

    monkeypatch.setattr(simulation, "_should_use_rust", lambda: True)
    monkeypatch.setattr(rust_backend, "simulate_gpcm", fake_simulate_gpcm)

    actual = simdata(
        model="PCM",
        theta=np.array([-1.0, 0.0, 1.0]),
        n_items=2,
        n_categories=3,
        steps=np.array([[-1.0, 1.0], [-0.5, 0.5]]),
        seed=17,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(captured["theta"], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(captured["discrimination"], np.ones(2))


@pytest.mark.parametrize("model", ["GRM", "GPCM", "PCM", "NRM"])
def test_polytomous_models_require_two_categories(model):
    with pytest.raises(ValueError, match="n_categories must be at least 2"):
        simdata(model=model, n_categories=1)


def test_pcm_rejects_free_discrimination_and_multidimensional_theta():
    with pytest.raises(ValueError, match="PCM discrimination is fixed to 1.0"):
        simdata(
            model="PCM",
            n_items=2,
            n_categories=3,
            discrimination=np.array([1.0, 1.2]),
        )

    with pytest.raises(ValueError, match="only supports unidimensional"):
        simdata(
            model="PCM",
            theta=np.zeros((3, 2)),
            n_items=2,
            n_categories=3,
        )


def test_polytomous_parameter_shapes_are_validated():
    with pytest.raises(ValueError, match=r"steps must have shape \(2, 2\)"):
        simdata(
            model="PCM",
            n_items=2,
            n_categories=3,
            steps=np.zeros((2, 3)),
        )

    with pytest.raises(ValueError, match=r"intercepts must have shape \(2, 3\)"):
        simdata(
            model="NRM",
            theta=np.zeros((4, 2)),
            n_items=2,
            n_categories=3,
            slopes=np.zeros((2, 3, 2)),
            intercepts=np.zeros((2, 2)),
        )
