"""Contracts for efficient terminal knowledge-tracing states."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.dynamic import BKTModel


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


def _histories() -> tuple[np.ndarray, np.ndarray]:
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
    person_skills = np.vstack(
        [
            shared_skills,
            np.roll(shared_skills, 1),
            np.roll(shared_skills, 2),
            np.roll(shared_skills, 3),
        ]
    )
    return responses, person_skills


def test_person_specific_terminal_states_match_individual_forward_recursions(
    model: BKTModel,
) -> None:
    responses, skills = _histories()

    latest = model.predict_mastery_batch(responses, skills)
    next_priors = model.next_mastery_priors_batch(responses, skills)

    for person_idx, person_responses in enumerate(responses):
        alpha, _ = model.forward(person_responses, skills[person_idx])
        expected_latest = model.p_init.copy()
        expected_next = model.p_init.copy()
        for skill_idx in range(model.n_skills):
            trial_indices = np.flatnonzero(skills[person_idx] == skill_idx)
            if len(trial_indices) == 0:
                continue
            posterior = alpha[trial_indices[-1], 1]
            expected_latest[skill_idx] = posterior
            expected_next[skill_idx] = (
                posterior * (1.0 - model.p_forget[skill_idx])
                + (1.0 - posterior) * model.p_learn[skill_idx]
            )
        assert_allclose(latest[person_idx], expected_latest)
        assert_allclose(next_priors[person_idx], expected_next)


def test_person_specific_terminal_states_skip_smoothing(
    model: BKTModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _histories()
    monkeypatch.setattr(
        model,
        "_forward_backward_batch_validated",
        lambda *args, **kwargs: pytest.fail("terminal states must be forward-only"),
    )
    monkeypatch.setattr(
        model,
        "_native_forward_backward_batch",
        lambda *args, **kwargs: pytest.fail("person-specific layouts are not native"),
    )

    latest = model.predict_mastery_batch(responses, skills)
    next_priors = model.next_mastery_priors_batch(responses, skills)

    assert latest.shape == next_priors.shape == (len(responses), model.n_skills)
    assert np.all(np.isfinite(latest))
    assert np.all(np.isfinite(next_priors))


def test_shared_terminal_states_preserve_native_dispatch(
    model: BKTModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, person_skills = _histories()
    shared_skills = person_skills[0]
    calls: list[tuple[int, int]] = []
    learned = np.linspace(0.1, 0.9, responses.size).reshape(responses.shape)

    def fake_native(
        response_values: np.ndarray,
        skill_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(response_values.shape)
        gamma = np.empty((*response_values.shape, 2))
        gamma[..., 1] = learned
        gamma[..., 0] = 1.0 - learned
        return gamma, np.zeros(len(response_values))

    monkeypatch.setattr(model, "_native_forward_backward_batch", fake_native)

    latest = model.predict_mastery_batch(responses, shared_skills)
    next_priors = model.next_mastery_priors_batch(responses, shared_skills)

    assert calls == [responses.shape, responses.shape]
    for skill_idx in range(model.n_skills):
        last_trial = np.flatnonzero(shared_skills == skill_idx)[-1]
        assert_allclose(latest[:, skill_idx], learned[:, last_trial])
        expected_next = (
            latest[:, skill_idx] * (1.0 - model.p_forget[skill_idx])
            + (1.0 - latest[:, skill_idx]) * model.p_learn[skill_idx]
        )
        assert_allclose(next_priors[:, skill_idx], expected_next)


def test_diagnostics_retain_states_for_forecasting(model: BKTModel) -> None:
    responses, skills = _histories()

    diagnostics = model.predictive_diagnostics_batch(responses, skills)
    retained_forecast = model.forecast_from_priors_batch(
        diagnostics.next_mastery_priors,
        7,
    )
    history_forecast = model.forecast_batch(responses, skills, 7)

    assert_allclose(
        diagnostics.latest_mastery_by_skill,
        model.predict_mastery_batch(responses, skills),
    )
    assert_allclose(
        diagnostics.next_mastery_priors,
        model.next_mastery_priors_batch(responses, skills),
    )
    assert_allclose(
        retained_forecast.mastery_probabilities,
        history_forecast.mastery_probabilities,
    )
    assert_allclose(
        retained_forecast.response_probabilities,
        history_forecast.response_probabilities,
    )


def test_terminal_states_recover_after_an_impossible_response() -> None:
    model = BKTModel(
        n_skills=2,
        p_init=np.array([0.0, 0.7]),
        p_learn=np.array([0.2, 0.1]),
        p_slip=np.array([0.0, 0.1]),
        p_guess=np.array([0.0, 0.2]),
        use_rust=False,
    )
    responses = np.array([1, 1], dtype=np.int32)
    skills = np.array([0, 0], dtype=np.int32)
    retained_prior = model.p_init[0]
    expected_latest = model.p_init.copy()

    for response in responses:
        step = model.online_step(
            int(response),
            0,
            prior_mastery=float(retained_prior),
        )
        expected_latest[0] = step.updated_mastery
        retained_prior = step.next_mastery

    latest = model.predict_mastery_by_skill(responses, skills)
    next_priors = model.next_mastery_priors(responses, skills)

    assert_allclose(latest, expected_latest)
    assert next_priors[0] == pytest.approx(retained_prior)
    assert latest[0] == 1.0
    assert latest[1] == model.p_init[1]
    assert next_priors[1] == model.p_init[1]


def test_terminal_state_methods_reuse_validated_inputs(
    model: BKTModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _histories()
    calls: list[str] = []

    def wrapped_update(*args: Any, **kwargs: Any) -> Any:
        calls.append("update")
        return original_update(*args, **kwargs)

    original_update = model._mastery_update_batch
    monkeypatch.setattr(model, "_mastery_update_batch", wrapped_update)

    model.predict_mastery_batch(responses, skills)

    assert len(calls) == responses.shape[1]
