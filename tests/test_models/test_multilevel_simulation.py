"""Contracts for bounded multilevel response simulation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.models.multilevel as multilevel_module
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multilevel import (
    CrossedRandomEffectsModel,
    MultilevelIRTModel,
    ThreeLevelIRTModel,
)
from mirt.models.polytomous import GradedResponseModel


def _binary_model(n_items: int = 4, n_factors: int = 1) -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=n_items, n_factors=n_factors)
    if n_factors == 1:
        discrimination = np.linspace(0.7, 1.4, n_items)
    else:
        discrimination = np.column_stack(
            [np.linspace(0.7, 1.4, n_items), np.linspace(0.3, 0.9, n_items)]
        )
    return model.set_parameters(
        discrimination=discrimination,
        difficulty=np.linspace(-0.8, 0.9, n_items),
    )


def _categorical_draws(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=2)
    cumulative[:, :, -1] = 1.0
    uniforms = rng.random((*probabilities.shape[:2], 1))
    return np.sum(uniforms >= cumulative, axis=2).astype(np.int32)


def test_two_level_simulation_matches_seeded_reference_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    membership = np.array([7, 2, 7, 2, 7, 2, 2, 7, 2])
    model = MultilevelIRTModel(_binary_model(), membership)
    model.set_group_means(np.array([-0.55, 0.8]))
    model.set_variance_components(between=0.3, within=0.36)
    seed = 20260829
    rng = np.random.default_rng(seed)
    expected_theta = rng.normal(
        model.person_prior_mean(),
        np.sqrt(model.within_variance),
    ).reshape(-1, 1)
    probabilities = model.base_model.probability(expected_theta)
    expected_responses = (rng.random(probabilities.shape) < probabilities).astype(
        np.int32
    )
    monkeypatch.setattr(multilevel_module, "_MULTILEVEL_MAX_WORKING_VALUES", 16)

    unchunked, unchunked_theta = model.simulate(seed, chunk_size=len(membership))
    chunked, chunked_theta = model.simulate(seed, chunk_size=2)
    automatic, automatic_theta = model.simulate(seed)

    assert unchunked.dtype == np.int32
    assert_array_equal(unchunked_theta, expected_theta)
    assert_array_equal(chunked_theta, expected_theta)
    assert_array_equal(automatic_theta, expected_theta)
    assert_array_equal(unchunked, expected_responses)
    assert_array_equal(chunked, expected_responses)
    assert_array_equal(automatic, expected_responses)
    assert_array_equal(model.sample_abilities(seed), expected_theta)


def test_two_level_ability_draws_recover_conditional_moments() -> None:
    n_per_group = 30_000
    model = MultilevelIRTModel(
        _binary_model(1),
        np.repeat([2, 7], n_per_group),
    ).set_group_means(np.array([-0.7, 0.9]))
    model.set_variance_components(between=0.5, within=0.49)

    theta = model.sample_abilities(seed=731)[:, 0]

    assert_allclose(
        [theta[:n_per_group].mean(), theta[n_per_group:].mean()],
        [-0.7, 0.9],
        atol=0.012,
    )
    assert_allclose(
        [theta[:n_per_group].var(), theta[n_per_group:].var()],
        [0.49, 0.49],
        atol=0.012,
    )


def test_two_level_simulation_supports_polytomous_base_models() -> None:
    base = GradedResponseModel(n_items=3, n_categories=[2, 3, 4])
    base.set_parameters(
        discrimination=np.array([0.8, 1.1, 1.4]),
        thresholds=np.array(
            [
                [0.2, 0.0, 0.0],
                [-0.8, 0.7, 0.0],
                [-1.2, -0.1, 1.0],
            ]
        ),
    )
    theta = np.linspace(-2.0, 2.0, 17)
    model = MultilevelIRTModel(base, np.repeat([2, 7], [8, 9]))
    probabilities = model.base_model.probability(theta[:, None])
    expected = _categorical_draws(probabilities, np.random.default_rng(91))

    unchunked, returned_theta = model.simulate(
        91,
        theta=theta,
        chunk_size=len(theta),
    )
    chunked, _ = model.simulate(91, theta=theta, chunk_size=3)

    assert_array_equal(returned_theta, theta[:, None])
    assert_array_equal(unchunked, expected)
    assert_array_equal(chunked, expected)
    assert np.all(unchunked[:, 0] < 2)
    assert np.all(unchunked[:, 1] < 3)
    assert np.all(unchunked[:, 2] < 4)


def test_multidimensional_two_level_simulation_accepts_explicit_traits() -> None:
    model = MultilevelIRTModel(
        _binary_model(n_items=3, n_factors=2),
        np.array([0, 0, 1, 1]),
    )
    theta = np.array([[-1.0, 0.2], [-0.2, 0.8], [0.5, -0.6], [1.2, 0.1]])
    probabilities = model.base_model.probability(theta)
    expected = (
        np.random.default_rng(17).random(probabilities.shape) < probabilities
    ).astype(np.int32)

    responses, returned_theta = model.simulate(17, theta=theta, chunk_size=1)

    assert_array_equal(responses, expected)
    assert_array_equal(returned_theta, theta)
    with pytest.raises(ValueError, match="unidimensional"):
        model.sample_abilities(seed=17)


def test_three_level_simulation_uses_combined_level_effects() -> None:
    model = ThreeLevelIRTModel(
        _binary_model(3),
        level2_membership=np.array([7, 2, 7, 11, 2]),
        level3_membership=np.array([9, 4, 9]),
    )
    model.set_level_effects(
        level2=np.array([0.1, 0.2, 0.3]),
        level3=np.array([-1.0, 1.0]),
    )
    model.set_variance_components(level2=0.2, level3=0.3, within=0.25)
    rng = np.random.default_rng(118)
    expected_theta = rng.normal(model.person_prior_mean(), 0.5).reshape(-1, 1)
    probabilities = model.base_model.probability(expected_theta)
    expected_responses = (rng.random(probabilities.shape) < probabilities).astype(
        np.int32
    )

    responses, theta = model.simulate(118, chunk_size=2)

    assert_array_equal(theta, expected_theta)
    assert_array_equal(responses, expected_responses)
    assert_array_equal(model.sample_abilities(118), expected_theta)


def test_crossed_simulation_matches_assigned_rater_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = np.array(
        [
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [1, 2, 0, 1],
            [0, 0, 2, 1],
            [2, 1, 1, 0],
            [1, 0, 0, 2],
        ]
    )
    model = CrossedRandomEffectsModel(_binary_model(), 3, assignments)
    model.set_rater_effects(np.array([-0.6, 0.1, 0.75]))
    theta = np.linspace(-1.5, 1.5, len(assignments))
    probabilities = model.assigned_probability(theta[:, None])
    expected = (
        np.random.default_rng(909).random(probabilities.shape) < probabilities
    ).astype(np.int32)
    monkeypatch.setattr(multilevel_module, "_MULTILEVEL_MAX_WORKING_VALUES", 8)

    unchunked = model.simulate(theta, seed=909, chunk_size=len(theta))
    chunked = model.simulate(theta, seed=909, chunk_size=2)
    automatic = model.simulate(theta, seed=909)

    assert_array_equal(unchunked, expected)
    assert_array_equal(chunked, expected)
    assert_array_equal(automatic, expected)


def test_crossed_simulation_without_assignments_uses_base_model() -> None:
    model = CrossedRandomEffectsModel(_binary_model(2), n_raters=3)
    theta = np.linspace(-1.0, 1.0, 7)
    probabilities = model.base_model.probability(theta[:, None])
    expected = (
        np.random.default_rng(55).random(probabilities.shape) < probabilities
    ).astype(np.int32)

    responses = model.simulate(theta, seed=55, chunk_size=1)

    assert_array_equal(responses, expected)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_multilevel_simulation_rejects_invalid_chunk_sizes(chunk_size: object) -> None:
    model = MultilevelIRTModel(_binary_model(2), np.array([0, 1]))

    with pytest.raises(ValueError, match="chunk_size"):
        model.simulate(
            9,
            theta=np.array([-0.5, 0.5]),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "theta",
    [
        np.array([0.0]),
        np.array([0.0, np.nan]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
    ],
)
def test_multilevel_simulation_validates_explicit_traits(theta: np.ndarray) -> None:
    model = MultilevelIRTModel(_binary_model(2), np.array([0, 1]))

    with pytest.raises(ValueError, match="theta"):
        model.simulate(3, theta=theta)


def test_crossed_simulation_requires_one_trait_row_per_assignment() -> None:
    model = CrossedRandomEffectsModel(
        _binary_model(2),
        2,
        np.array([[0, 1], [1, 0]]),
    )

    with pytest.raises(ValueError, match="2 persons"):
        model.simulate(np.array([0.0]), seed=3)
