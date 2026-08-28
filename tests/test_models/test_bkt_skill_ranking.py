"""Contracts for adaptive Bayesian knowledge-tracing skill ranking."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from mirt.models.dynamic import (
    BKTBatchSkillRankingResult,
    BKTModel,
    BKTSkillCriterion,
    BKTSkillRankingResult,
)


@pytest.fixture
def model() -> BKTModel:
    return BKTModel(
        n_skills=4,
        allow_forgetting=True,
        p_init=np.array([0.2, 0.55, 0.8, 0.35]),
        p_learn=np.array([0.25, 0.12, 0.05, 0.3]),
        p_forget=np.array([0.02, 0.08, 0.15, 0.04]),
        p_slip=np.array([0.08, 0.15, 0.22, 0.05]),
        p_guess=np.array([0.12, 0.25, 0.35, 0.4]),
        use_rust=False,
    )


def _binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def test_models_module_exports_skill_ranking_results() -> None:
    from mirt.models import (
        BKTBatchSkillRankingResult as PublicBatchSkillRankingResult,
    )
    from mirt.models import BKTSkillRankingResult as PublicSkillRankingResult

    assert PublicSkillRankingResult is BKTSkillRankingResult
    assert PublicBatchSkillRankingResult is BKTBatchSkillRankingResult


def test_information_gain_matches_binary_mutual_information(
    model: BKTModel,
) -> None:
    priors = np.array([0.18, 0.42, 0.73, 0.91])
    original = priors.copy()
    response_probabilities = (
        priors * (1.0 - model.p_slip) + (1.0 - priors) * model.p_guess
    )
    expected_scores = np.array(
        [
            _binary_entropy(float(response_probability))
            - prior * _binary_entropy(float(1.0 - slip))
            - (1.0 - prior) * _binary_entropy(float(guess))
            for prior, slip, guess, response_probability in zip(
                priors,
                model.p_slip,
                model.p_guess,
                response_probabilities,
                strict=True,
            )
        ]
    )
    expected_order = np.argsort(-expected_scores, kind="stable")

    result = model.rank_skills(priors)

    assert isinstance(result, BKTSkillRankingResult)
    assert result.criterion == "information_gain"
    assert result.n_recommendations == model.n_skills
    assert result.best_skill_index == expected_order[0]
    assert result.best_score == pytest.approx(expected_scores[expected_order[0]])
    assert_array_equal(result.skill_indices, expected_order)
    assert_allclose(result.scores, expected_scores[expected_order])
    assert_allclose(result.mastery_probabilities, priors[expected_order])
    assert_allclose(
        result.response_probabilities,
        response_probabilities[expected_order],
    )
    assert_array_equal(priors, original)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "criterion", "lowest_mastery")


def test_information_gain_is_zero_without_mastery_uncertainty_or_separation() -> None:
    model = BKTModel(
        n_skills=3,
        p_slip=np.array([0.2, 0.1, 0.3]),
        p_guess=np.array([0.8, 0.2, 0.6]),
        use_rust=False,
    )

    result = model.rank_skills(np.array([0.4, 0.0, 1.0]))

    assert_array_equal(result.skill_indices, np.array([0, 1, 2]))
    assert_allclose(result.scores, 0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("criterion", "expected_scores"),
    [
        (
            "mastery_gain",
            lambda model, priors: (
                (1.0 - priors) * model.p_learn - priors * model.p_forget
            ),
        ),
        ("lowest_mastery", lambda _model, priors: 1.0 - priors),
        (
            "success_probability",
            lambda model, priors: (
                priors * (1.0 - model.p_slip) + (1.0 - priors) * model.p_guess
            ),
        ),
    ],
)
def test_alternative_criteria_match_their_definitions(
    model: BKTModel,
    criterion: BKTSkillCriterion,
    expected_scores: Callable[
        [BKTModel, NDArray[np.float64]],
        NDArray[np.float64],
    ],
) -> None:
    priors = np.array([0.18, 0.42, 0.73, 0.91])
    scores = expected_scores(model, priors)
    expected_order = np.argsort(-scores, kind="stable")

    result = model.rank_skills(priors, criterion=criterion)

    assert_array_equal(result.skill_indices, expected_order)
    assert_allclose(result.scores, scores[expected_order])


def test_candidate_subset_is_sorted_for_stable_ties() -> None:
    model = BKTModel(n_skills=4, use_rust=False)
    available_skills = np.array([3, 1, 2])
    original = available_skills.copy()

    result = model.rank_skills(
        np.full(4, 0.5),
        criterion="lowest_mastery",
        available_skills=available_skills,
        top_k=2,
    )

    assert_array_equal(result.skill_indices, np.array([1, 2]))
    assert_allclose(result.scores, 0.5)
    assert_array_equal(available_skills, original)


@pytest.mark.parametrize(
    "criterion",
    [
        "information_gain",
        "mastery_gain",
        "lowest_mastery",
        "success_probability",
    ],
)
def test_batch_ranking_matches_scalar_rows(
    model: BKTModel,
    criterion: BKTSkillCriterion,
) -> None:
    priors = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.45, 0.55, 0.65, 0.75],
            [0.9, 0.8, 0.7, 0.6],
            [0.0, 0.5, 1.0, 0.25],
        ]
    )
    original = priors.copy()
    available = np.array([3, 1, 2])

    result = model.rank_skills_batch(
        priors,
        criterion=criterion,
        available_skills=available,
        top_k=2,
    )

    assert isinstance(result, BKTBatchSkillRankingResult)
    assert result.n_persons == len(priors)
    assert result.n_recommendations == 2
    assert_array_equal(result.best_skill_indices, result.skill_indices[:, 0])
    for person_idx, person_priors in enumerate(priors):
        expected = model.rank_skills(
            person_priors,
            criterion=criterion,
            available_skills=available,
            top_k=2,
        )
        for field_name in (
            "skill_indices",
            "scores",
            "mastery_probabilities",
            "response_probabilities",
        ):
            assert_allclose(
                getattr(result, field_name)[person_idx],
                getattr(expected, field_name),
            )
    assert_array_equal(priors, original)


def test_ranking_continues_from_retained_history_priors(model: BKTModel) -> None:
    responses = np.array(
        [
            [1, 0, 1, -1, 1, 0, 1, 1],
            [0, 1, -1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, -1, 0],
        ],
        dtype=np.int32,
    )
    skills = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    diagnostics = model.predictive_diagnostics_batch(responses, skills)

    result = model.rank_skills_batch(
        diagnostics.next_mastery_priors,
        top_k=3,
    )

    for person_idx in range(len(responses)):
        expected = model.rank_skills(
            model.next_mastery_priors(responses[person_idx], skills),
            top_k=3,
        )
        assert_array_equal(result.skill_indices[person_idx], expected.skill_indices)
        assert_allclose(result.scores[person_idx], expected.scores)


@pytest.mark.parametrize(
    "available_skills",
    [
        np.array([], dtype=int),
        np.array([[0, 1]]),
        np.array([0.0, 1.0]),
        np.array([True, False]),
        np.array([-1, 1]),
        np.array([0, 4]),
        np.array([0, 0]),
    ],
)
def test_ranking_validates_candidate_skills(
    model: BKTModel,
    available_skills: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="available_skills"):
        model.rank_skills(
            np.full(4, 0.5),
            available_skills=available_skills,
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, 4])
def test_ranking_validates_top_k(model: BKTModel, top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        model.rank_skills(
            np.full(4, 0.5),
            available_skills=np.array([0, 1, 2]),
            top_k=cast(int, top_k),
        )


def test_ranking_validates_criterion(model: BKTModel) -> None:
    with pytest.raises(ValueError, match="criterion"):
        model.rank_skills(
            np.full(4, 0.5),
            criterion=cast(BKTSkillCriterion, "uncertainty"),
        )


@pytest.mark.parametrize(
    "prior_mastery",
    [
        np.array([0.2, 0.4, 0.6]),
        np.array([[0.2, 0.4, 0.6, 0.8]]),
        np.array([True, False, True, False]),
        np.array([0.2, np.nan, 0.6, 0.8]),
        np.array([0.2, -0.1, 0.6, 0.8]),
    ],
)
def test_scalar_ranking_validates_priors(
    model: BKTModel,
    prior_mastery: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="prior_mastery"):
        model.rank_skills(prior_mastery)


@pytest.mark.parametrize(
    "prior_mastery",
    [
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.empty((0, 4)),
        np.ones((2, 3)),
        np.ones((2, 4), dtype=bool),
        np.array([[0.2, 0.4, 0.6, 0.8], [0.3, 0.2, 1.1, 0.7]]),
    ],
)
def test_batch_ranking_validates_priors(
    model: BKTModel,
    prior_mastery: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="prior_mastery"):
        model.rank_skills_batch(prior_mastery)
