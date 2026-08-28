"""Contracts for online Bayesian knowledge tracing updates."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.dynamic import (
    BKTBatchStepResult,
    BKTModel,
    BKTStepResult,
)


@pytest.fixture
def model() -> BKTModel:
    return BKTModel(
        n_skills=3,
        allow_forgetting=True,
        p_init=np.array([0.2, 0.55, 0.8]),
        p_learn=np.array([0.25, 0.12, 0.05]),
        p_forget=np.array([0.02, 0.08, 0.15]),
        p_slip=np.array([0.08, 0.15, 0.22]),
        p_guess=np.array([0.12, 0.25, 0.35]),
        use_rust=False,
    )


def test_models_module_exports_online_results() -> None:
    from mirt.models import BKTBatchStepResult as PublicBatchStepResult
    from mirt.models import BKTStepResult as PublicStepResult

    assert PublicStepResult is BKTStepResult
    assert PublicBatchStepResult is BKTBatchStepResult


def test_online_sequence_matches_forward_recursion(model: BKTModel) -> None:
    responses = np.array([1, 0, 1, -1, 0, 1, 1, 0, 1], dtype=np.int32)
    skills = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    alpha, scaling = model.forward(responses, skills)
    _, expected_log_likelihood = model.forward_backward(responses, skills)
    priors = model.p_init.copy()
    latest_posteriors = model.p_init.copy()
    log_likelihood = 0.0

    for trial, (response, skill_idx) in enumerate(zip(responses, skills, strict=True)):
        prior = priors[skill_idx]
        result = model.online_step(
            int(response),
            int(skill_idx),
            prior_mastery=prior,
        )

        assert isinstance(result, BKTStepResult)
        assert result.updated_mastery == pytest.approx(alpha[trial, 1])
        assert np.exp(result.response_log_likelihood) == pytest.approx(scaling[trial])
        expected_probability = (
            prior * (1.0 - model.p_slip[skill_idx])
            + (1.0 - prior) * model.p_guess[skill_idx]
        )
        assert result.response_probability == pytest.approx(expected_probability)
        if response < 0:
            assert np.isnan(result.residual)
            assert np.isnan(result.standardized_residual)
        else:
            assert result.residual == pytest.approx(response - expected_probability)
            assert result.standardized_residual == pytest.approx(
                result.residual
                / np.sqrt(expected_probability * (1.0 - expected_probability))
            )
        latest_posteriors[skill_idx] = result.updated_mastery
        priors[skill_idx] = result.next_mastery
        log_likelihood += result.response_log_likelihood

    assert_allclose(
        latest_posteriors,
        model.predict_mastery_by_skill(responses, skills),
    )
    assert log_likelihood == pytest.approx(expected_log_likelihood)
    with pytest.raises(FrozenInstanceError):
        result.updated_mastery = 0.0


def test_online_batch_matches_person_specific_forward_recursions(
    model: BKTModel,
) -> None:
    responses = np.array(
        [
            [1, 0, 1, -1, 1, 0, 1, 1],
            [0, 1, -1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, -1, 0],
            [0, -1, 1, 1, 0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    skills = np.array(
        [
            [0, 1, 2, 0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1, 2, 0],
            [0, 2, 1, 0, 2, 1, 0, 2],
        ],
        dtype=np.int32,
    )
    expected_alpha = np.stack(
        [
            model.forward(person_responses, person_skills)[0]
            for person_responses, person_skills in zip(
                responses,
                skills,
                strict=True,
            )
        ]
    )
    _, expected_log_likelihoods = model.forward_backward_batch(responses, skills)
    priors = np.broadcast_to(model.p_init, (len(responses), model.n_skills)).copy()
    latest_posteriors = priors.copy()
    log_likelihoods = np.zeros(len(responses))
    rows = np.arange(len(responses))

    for trial in range(responses.shape[1]):
        trial_skills = skills[:, trial]
        result = model.online_step_batch(
            responses[:, trial],
            trial_skills,
            prior_mastery=priors[rows, trial_skills],
        )

        assert isinstance(result, BKTBatchStepResult)
        assert result.n_persons == len(responses)
        assert_allclose(result.updated_mastery, expected_alpha[:, trial, 1])
        for person_idx in range(len(responses)):
            scalar = model.online_step(
                int(responses[person_idx, trial]),
                int(trial_skills[person_idx]),
                prior_mastery=priors[person_idx, trial_skills[person_idx]],
            )
            assert scalar.updated_mastery == pytest.approx(
                result.updated_mastery[person_idx]
            )
            assert scalar.next_mastery == pytest.approx(result.next_mastery[person_idx])
        latest_posteriors[rows, trial_skills] = result.updated_mastery
        priors[rows, trial_skills] = result.next_mastery
        log_likelihoods += result.response_log_likelihoods

    assert_allclose(
        latest_posteriors,
        model.predict_mastery_batch(responses, skills),
    )
    assert_allclose(log_likelihoods, expected_log_likelihoods)


def test_online_batch_broadcasts_skill_and_prior(model: BKTModel) -> None:
    responses = np.array([1, 0, -1, 1], dtype=np.int32)

    result = model.online_step_batch(
        responses,
        1,
        prior_mastery=0.4,
    )

    for person_idx, response in enumerate(responses):
        scalar = model.online_step(int(response), 1, prior_mastery=0.4)
        assert result.response_probabilities[person_idx] == pytest.approx(
            scalar.response_probability
        )
        assert result.updated_mastery[person_idx] == pytest.approx(
            scalar.updated_mastery
        )
        assert result.next_mastery[person_idx] == pytest.approx(scalar.next_mastery)


def test_online_defaults_to_assigned_skill_initial_mastery(model: BKTModel) -> None:
    responses = np.array([1, 0, 1], dtype=np.int32)
    skills = np.array([0, 1, 2], dtype=np.int32)

    result = model.online_step_batch(responses, skills)

    for person_idx, skill_idx in enumerate(skills):
        scalar = model.online_step(int(responses[person_idx]), int(skill_idx))
        expected = model.online_step(
            int(responses[person_idx]),
            int(skill_idx),
            prior_mastery=float(model.p_init[skill_idx]),
        )
        assert scalar == expected
        assert result.updated_mastery[person_idx] == pytest.approx(
            expected.updated_mastery
        )


def test_missing_online_response_preserves_posterior_and_advances_transition(
    model: BKTModel,
) -> None:
    prior = 0.63
    skill_idx = 1

    result = model.online_step(-1, skill_idx, prior_mastery=prior)

    expected_next = (
        prior * (1.0 - model.p_forget[skill_idx])
        + (1.0 - prior) * model.p_learn[skill_idx]
    )
    assert result.response_log_likelihood == 0.0
    assert result.updated_mastery == prior
    assert result.next_mastery == pytest.approx(expected_next)
    assert np.isnan(result.residual)
    assert np.isnan(result.standardized_residual)


def test_online_diagnostics_remain_finite_for_boundary_parameters() -> None:
    model = BKTModel(
        n_skills=1,
        p_init=np.array([0.0]),
        p_learn=np.array([0.2]),
        p_slip=np.array([0.0]),
        p_guess=np.array([0.0]),
        use_rust=False,
    )

    result = model.online_step(1, 0)

    assert np.isfinite(result.response_log_likelihood)
    assert np.isfinite(result.residual)
    assert np.isfinite(result.standardized_residual)
    assert result.updated_mastery == 0.0
    assert result.next_mastery == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("responses", "skills", "prior_mastery", "match"),
    [
        (np.empty(0, dtype=int), 0, None, "responses"),
        (np.zeros((2, 1), dtype=int), 0, None, "responses"),
        (np.array([1.0, 0.0]), 0, None, "integer"),
        (np.array([1, 2]), 0, None, "only -1, 0, or 1"),
        (np.array([1, 0]), 1.0, None, "integer"),
        (np.array([1, 0]), np.array([0]), None, "match responses"),
        (np.array([1, 0]), np.array([0, 3]), None, "must be in"),
        (np.array([1, 0]), 0, True, "prior_mastery"),
        (np.array([1, 0]), 0, np.array([0.5]), "match responses"),
        (np.array([1, 0]), 0, np.array([0.5, 1.1]), "in \\[0, 1\\]"),
    ],
)
def test_online_batch_validates_inputs(
    model: BKTModel,
    responses: np.ndarray,
    skills: object,
    prior_mastery: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model.online_step_batch(
            responses,
            skills,  # type: ignore[arg-type]
            prior_mastery=prior_mastery,  # type: ignore[arg-type]
        )
