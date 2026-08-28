"""Contracts for Bayesian knowledge-tracing mastery forecasts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.models.dynamic import (
    BKTBatchForecastResult,
    BKTForecastResult,
    BKTModel,
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


def test_models_module_exports_forecast_results() -> None:
    from mirt.models import BKTBatchForecastResult as PublicBatchForecastResult
    from mirt.models import BKTForecastResult as PublicForecastResult

    assert PublicForecastResult is BKTForecastResult
    assert PublicBatchForecastResult is BKTBatchForecastResult


def test_forecast_from_priors_matches_iterative_transition(model: BKTModel) -> None:
    priors = np.array([0.31, 0.64, 0.47])
    original = priors.copy()
    n_steps = 9
    expected_mastery = np.empty((n_steps, model.n_skills))
    current = priors.copy()

    for step in range(n_steps):
        expected_mastery[step] = current
        current = current * (1.0 - model.p_forget) + (1.0 - current) * model.p_learn

    result = model.forecast_from_priors(priors, n_steps)
    expected_response = (
        expected_mastery * (1.0 - model.p_slip)
        + (1.0 - expected_mastery) * model.p_guess
    )

    assert isinstance(result, BKTForecastResult)
    assert result.n_steps == n_steps
    assert result.n_skills == model.n_skills
    assert result.mastery_probabilities.shape == (n_steps, model.n_skills)
    assert result.response_probabilities.shape == (n_steps, model.n_skills)
    assert_allclose(result.mastery_probabilities, expected_mastery)
    assert_allclose(result.response_probabilities, expected_response)
    assert_array_equal(priors, original)
    with pytest.raises(FrozenInstanceError):
        result.mastery_probabilities = np.empty((0, 0))


def test_forecast_handles_constant_and_alternating_transitions() -> None:
    model = BKTModel(
        n_skills=2,
        allow_forgetting=True,
        p_init=np.array([0.3, 0.2]),
        p_learn=np.array([0.0, 1.0]),
        p_forget=np.array([0.0, 1.0]),
        p_slip=np.array([0.1, 0.2]),
        p_guess=np.array([0.2, 0.3]),
        use_rust=False,
    )

    result = model.forecast_from_priors(np.array([0.3, 0.2]), 6)

    assert_allclose(result.mastery_probabilities[:, 0], 0.3)
    assert_allclose(
        result.mastery_probabilities[:, 1],
        np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
    )
    assert np.all(np.isfinite(result.response_probabilities))


def test_batch_forecast_matches_scalar_forecasts(model: BKTModel) -> None:
    priors = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.45, 0.55, 0.65],
            [0.9, 0.8, 0.7],
            [0.0, 0.5, 1.0],
        ]
    )

    result = model.forecast_from_priors_batch(priors, 7)

    assert isinstance(result, BKTBatchForecastResult)
    assert result.n_persons == len(priors)
    assert result.n_steps == 7
    assert result.n_skills == model.n_skills
    for person_idx, person_priors in enumerate(priors):
        expected = model.forecast_from_priors(person_priors, 7)
        assert_allclose(
            result.mastery_probabilities[person_idx],
            expected.mastery_probabilities,
        )
        assert_allclose(
            result.response_probabilities[person_idx],
            expected.response_probabilities,
        )


def test_history_forecast_matches_retained_online_priors(model: BKTModel) -> None:
    responses = np.array([1, 0, -1, 1, 1, 0, 1, -1], dtype=np.int32)
    skills = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    retained_priors = model.p_init.copy()

    for response, skill_idx in zip(responses, skills, strict=True):
        update = model.online_step(
            int(response),
            int(skill_idx),
            prior_mastery=float(retained_priors[skill_idx]),
        )
        retained_priors[skill_idx] = update.next_mastery

    history_priors = model.next_mastery_priors(responses, skills)
    history_forecast = model.forecast(responses, skills, 8)
    retained_forecast = model.forecast_from_priors(retained_priors, 8)

    assert_allclose(history_priors, retained_priors)
    assert_allclose(
        history_forecast.mastery_probabilities,
        retained_forecast.mastery_probabilities,
    )
    assert_allclose(
        history_forecast.response_probabilities,
        retained_forecast.response_probabilities,
    )
    assert history_priors[2] == model.p_init[2]
    assert history_forecast.mastery_probabilities[0, 2] == model.p_init[2]


@pytest.mark.parametrize("person_specific", [False, True])
def test_batch_history_forecast_matches_individual_histories(
    model: BKTModel,
    person_specific: bool,
) -> None:
    responses = np.array(
        [
            [1, 0, 1, -1, 0, 1],
            [0, 1, -1, 1, 1, 0],
            [1, 1, 0, 0, -1, 1],
            [0, -1, 1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    shared_skills = np.array([0, 1, 0, 2, 1, 0], dtype=np.int32)
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

    priors = model.next_mastery_priors_batch(responses, skill_assignments)
    result = model.forecast_batch(responses, skill_assignments, 5)

    for person_idx, person_responses in enumerate(responses):
        person_skills = (
            skill_assignments[person_idx]
            if skill_assignments.ndim == 2
            else skill_assignments
        )
        expected_priors = model.next_mastery_priors(
            person_responses,
            person_skills,
        )
        expected = model.forecast(person_responses, person_skills, 5)
        assert_allclose(priors[person_idx], expected_priors)
        assert_allclose(
            result.mastery_probabilities[person_idx],
            expected.mastery_probabilities,
        )
        assert_allclose(
            result.response_probabilities[person_idx],
            expected.response_probabilities,
        )


def test_long_forecast_remains_bounded_and_converges(model: BKTModel) -> None:
    result = model.forecast_from_priors(np.array([0.0, 0.5, 1.0]), 10_000)
    equilibrium = model.p_learn / (model.p_learn + model.p_forget)

    assert np.all(np.isfinite(result.mastery_probabilities))
    assert np.all(result.mastery_probabilities >= 0.0)
    assert np.all(result.mastery_probabilities <= 1.0)
    assert np.all(result.response_probabilities >= 0.0)
    assert np.all(result.response_probabilities <= 1.0)
    assert_allclose(result.mastery_probabilities[-1], equilibrium)


@pytest.mark.parametrize(
    ("prior_mastery", "n_steps", "match"),
    [
        (np.array([0.2, 0.4, 0.6]), 0, "positive integer"),
        (np.array([0.2, 0.4, 0.6]), -1, "positive integer"),
        (np.array([0.2, 0.4, 0.6]), 1.5, "positive integer"),
        (np.array([0.2, 0.4, 0.6]), True, "positive integer"),
        (np.array([[0.2, 0.4, 0.6]]), 2, "shape"),
        (np.array([0.2, 0.4]), 2, "shape"),
        (np.array([True, False, True]), 2, "values"),
        (np.array([0.2 + 0.1j, 0.4, 0.6]), 2, "values"),
        (np.array([0.2, np.nan, 0.6]), 2, "values"),
        (np.array([0.2, 1.1, 0.6]), 2, "values"),
    ],
)
def test_scalar_forecast_validates_inputs(
    model: BKTModel,
    prior_mastery: np.ndarray,
    n_steps: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model.forecast_from_priors(
            prior_mastery,
            n_steps,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "prior_mastery",
    [
        np.array([0.2, 0.4, 0.6]),
        np.empty((0, 3)),
        np.empty((2, 0)),
        np.ones((2, 2)),
        np.ones((2, 3), dtype=bool),
        np.array([[0.2, 0.4, 0.6], [0.3, -0.1, 0.7]]),
    ],
)
def test_batch_forecast_validates_priors(
    model: BKTModel,
    prior_mastery: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="prior_mastery"):
        model.forecast_from_priors_batch(prior_mastery, 3)
