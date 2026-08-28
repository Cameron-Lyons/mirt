"""Contracts for causal Bayesian knowledge-tracing diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.models.dynamic import (
    BKTBatchPredictiveResult,
    BKTModel,
    BKTPredictiveResult,
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


def test_models_module_exports_predictive_results() -> None:
    from mirt.models import BKTBatchPredictiveResult as PublicBatchPredictiveResult
    from mirt.models import BKTPredictiveResult as PublicPredictiveResult

    assert PublicPredictiveResult is BKTPredictiveResult
    assert PublicBatchPredictiveResult is BKTBatchPredictiveResult


def test_predictive_diagnostics_match_forward_and_online_recursions(
    model: BKTModel,
) -> None:
    responses = np.array([1, 0, 1, -1, 0, 1, 1, 0, 1], dtype=np.int32)
    skills = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    alpha, scaling = model.forward(responses, skills)
    _, expected_log_likelihood = model.forward_backward(responses, skills)
    retained_priors = model.p_init.copy()

    result = model.predictive_diagnostics(responses, skills)

    assert isinstance(result, BKTPredictiveResult)
    assert result.n_trials == len(responses)
    assert_allclose(result.updated_mastery, alpha[:, 1])
    assert_allclose(np.exp(result.response_log_likelihoods), scaling)
    assert result.total_log_likelihood == pytest.approx(expected_log_likelihood)

    for trial, (response, skill_idx) in enumerate(zip(responses, skills, strict=True)):
        expected = model.online_step(
            int(response),
            int(skill_idx),
            prior_mastery=float(retained_priors[skill_idx]),
        )
        assert result.predicted_mastery[trial] == pytest.approx(
            retained_priors[skill_idx]
        )
        assert result.response_probabilities[trial] == pytest.approx(
            expected.response_probability
        )
        assert result.response_log_likelihoods[trial] == pytest.approx(
            expected.response_log_likelihood
        )
        if response < 0:
            assert np.isnan(result.residuals[trial])
            assert np.isnan(result.standardized_residuals[trial])
        else:
            assert result.residuals[trial] == pytest.approx(expected.residual)
            assert result.standardized_residuals[trial] == pytest.approx(
                expected.standardized_residual
            )
        assert result.updated_mastery[trial] == pytest.approx(expected.updated_mastery)
        assert result.next_mastery[trial] == pytest.approx(expected.next_mastery)
        retained_priors[skill_idx] = expected.next_mastery

    with pytest.raises(FrozenInstanceError):
        result.predicted_mastery = np.empty(0)


@pytest.mark.parametrize("person_specific", [False, True])
def test_batch_diagnostics_match_individual_histories(
    model: BKTModel,
    person_specific: bool,
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
    shared_skills = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
    skill_assignments = (
        np.vstack(
            [
                shared_skills,
                np.roll(shared_skills, 1),
                np.roll(shared_skills, 2),
                np.roll(shared_skills, 3),
            ]
        )
        if person_specific
        else shared_skills
    )

    result = model.predictive_diagnostics_batch(responses, skill_assignments)
    _, expected_log_likelihoods = model.forward_backward_batch(
        responses,
        skill_assignments,
    )

    assert isinstance(result, BKTBatchPredictiveResult)
    assert result.n_persons == len(responses)
    assert result.n_trials == responses.shape[1]
    assert_allclose(result.total_log_likelihoods, expected_log_likelihoods)
    for person_idx, person_responses in enumerate(responses):
        person_skills = (
            skill_assignments[person_idx]
            if skill_assignments.ndim == 2
            else skill_assignments
        )
        expected = model.predictive_diagnostics(person_responses, person_skills)
        for field_name in (
            "predicted_mastery",
            "response_probabilities",
            "response_log_likelihoods",
            "residuals",
            "standardized_residuals",
            "updated_mastery",
            "next_mastery",
        ):
            assert_allclose(
                getattr(result, field_name)[person_idx],
                getattr(expected, field_name),
                equal_nan=True,
            )


def test_predictive_diagnostics_treat_missing_trials_consistently(
    model: BKTModel,
) -> None:
    responses = np.full((2, 7), -1, dtype=np.int32)
    skills = np.array([0, 1, 0, 2, 1, 0, 2], dtype=np.int32)

    result = model.predictive_diagnostics_batch(responses, skills)

    assert_allclose(result.response_log_likelihoods, 0.0)
    assert_allclose(result.total_log_likelihoods, 0.0)
    assert np.all(np.isnan(result.residuals))
    assert np.all(np.isnan(result.standardized_residuals))
    assert_allclose(result.updated_mastery, result.predicted_mastery)
    for trial, skill_idx in enumerate(skills):
        expected_next = (
            result.updated_mastery[:, trial] * (1.0 - model.p_forget[skill_idx])
            + (1.0 - result.updated_mastery[:, trial]) * model.p_learn[skill_idx]
        )
        assert_allclose(result.next_mastery[:, trial], expected_next)


def test_predictive_outputs_are_causal(model: BKTModel) -> None:
    baseline = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    changed = baseline.copy()
    changed[4:] = 1 - changed[4:]
    skills = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

    baseline_result = model.predictive_diagnostics(baseline, skills)
    changed_result = model.predictive_diagnostics(changed, skills)

    assert_allclose(
        baseline_result.predicted_mastery[:5],
        changed_result.predicted_mastery[:5],
    )
    assert_allclose(
        baseline_result.response_probabilities[:5],
        changed_result.response_probabilities[:5],
    )
    for field_name in (
        "response_log_likelihoods",
        "residuals",
        "standardized_residuals",
        "updated_mastery",
        "next_mastery",
    ):
        assert_allclose(
            getattr(baseline_result, field_name)[:4],
            getattr(changed_result, field_name)[:4],
        )
    assert baseline_result.updated_mastery[4] != pytest.approx(
        changed_result.updated_mastery[4]
    )


def test_boundary_diagnostics_remain_finite() -> None:
    model = BKTModel(
        n_skills=1,
        p_init=np.array([0.0]),
        p_learn=np.array([0.2]),
        p_slip=np.array([0.0]),
        p_guess=np.array([0.0]),
        use_rust=False,
    )
    responses = np.array([1, 1, 0, -1], dtype=np.int32)
    skills = np.zeros(len(responses), dtype=np.int32)

    result = model.predictive_diagnostics(responses, skills)

    assert np.all(np.isfinite(result.response_probabilities))
    assert np.all(np.isfinite(result.response_log_likelihoods))
    assert np.all(np.isfinite(result.updated_mastery))
    assert np.all(np.isfinite(result.next_mastery))
    assert np.all(np.isfinite(result.residuals[responses >= 0]))
    assert np.all(np.isfinite(result.standardized_residuals[responses >= 0]))


def test_diagnostics_do_not_mutate_inputs(model: BKTModel) -> None:
    responses = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.int32)
    skills = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    original_responses = responses.copy()
    original_skills = skills.copy()

    model.predictive_diagnostics_batch(responses, skills)

    assert_array_equal(responses, original_responses)
    assert_array_equal(skills, original_skills)


@pytest.mark.parametrize(
    ("responses", "skills", "match"),
    [
        (np.array([], dtype=int), np.array([], dtype=int), "at least one"),
        (np.array([1.0, 0.0]), np.array([0, 0]), "integer"),
        (np.array([1, 2]), np.array([0, 0]), "only -1, 0, or 1"),
        (np.array([1, 0]), np.array([0]), "equal length"),
        (np.array([1, 0]), np.array([0, 3]), "must be in"),
    ],
)
def test_predictive_diagnostics_validate_inputs(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model.predictive_diagnostics(responses, skills)
