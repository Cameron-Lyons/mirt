"""Regression coverage for vectorized BKT Gibbs updates."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.estimation.dynamic_gibbs import BKTGibbsSampler, BKTPriors
from mirt.models.dynamic import BKTModel


class _RecordingRNG:
    def __init__(self) -> None:
        self.beta_shapes: list[tuple[float, float]] = []

    def beta(self, alpha: float, beta: float) -> float:
        self.beta_shapes.append((float(alpha), float(beta)))
        return float(alpha / (alpha + beta))


def _reference_beta_shapes(
    priors: BKTPriors,
    responses: np.ndarray,
    learning_states: np.ndarray,
    skill_assignments: np.ndarray,
    n_skills: int,
) -> list[tuple[float, float]]:
    """Return Beta shapes using the original scalar counting rules."""
    shapes: list[tuple[float, float]] = []
    n_persons = learning_states.shape[0]

    for skill_idx in range(n_skills):
        first_states: list[int] = []
        for person_idx in range(n_persons):
            skill_trials = np.flatnonzero(skill_assignments == skill_idx)
            if len(skill_trials) > 0:
                first_states.append(int(learning_states[person_idx, skill_trials[0]]))
        if first_states:
            n_learned = sum(first_states)
            shapes.append(
                (
                    priors.p_init[0] + n_learned,
                    priors.p_init[1] + len(first_states) - n_learned,
                )
            )

    for skill_idx in range(n_skills):
        n_transitions = 0
        n_learned = 0
        skill_trials = np.flatnonzero(skill_assignments == skill_idx)
        for person_idx in range(n_persons):
            for previous_trial, current_trial in zip(
                skill_trials[:-1], skill_trials[1:]
            ):
                if learning_states[person_idx, previous_trial] == 0:
                    n_transitions += 1
                    if learning_states[person_idx, current_trial] == 1:
                        n_learned += 1
        shapes.append(
            (
                priors.p_learn[0] + n_learned,
                priors.p_learn[1] + n_transitions - n_learned,
            )
        )

    for skill_idx in range(n_skills):
        n_transitions = 0
        n_forgot = 0
        skill_trials = np.flatnonzero(skill_assignments == skill_idx)
        for person_idx in range(n_persons):
            for previous_trial, current_trial in zip(
                skill_trials[:-1], skill_trials[1:]
            ):
                if learning_states[person_idx, previous_trial] == 1:
                    n_transitions += 1
                    if learning_states[person_idx, current_trial] == 0:
                        n_forgot += 1
        shapes.append(
            (
                priors.p_forget[0] + n_forgot,
                priors.p_forget[1] + n_transitions - n_forgot,
            )
        )

    for skill_idx in range(n_skills):
        n_learned_trials = 0
        n_slips = 0
        skill_trials = np.flatnonzero(skill_assignments == skill_idx)
        for person_idx in range(n_persons):
            for trial_idx in skill_trials:
                response = responses[person_idx, trial_idx]
                if learning_states[person_idx, trial_idx] == 1 and response >= 0:
                    n_learned_trials += 1
                    if response == 0:
                        n_slips += 1
        shapes.append(
            (
                priors.p_slip[0] + n_slips,
                priors.p_slip[1] + n_learned_trials - n_slips,
            )
        )

    for skill_idx in range(n_skills):
        n_unlearned_trials = 0
        n_guessed = 0
        skill_trials = np.flatnonzero(skill_assignments == skill_idx)
        for person_idx in range(n_persons):
            for trial_idx in skill_trials:
                response = responses[person_idx, trial_idx]
                if learning_states[person_idx, trial_idx] == 0 and response >= 0:
                    n_unlearned_trials += 1
                    if response == 1:
                        n_guessed += 1
        shapes.append(
            (
                priors.p_guess[0] + n_guessed,
                priors.p_guess[1] + n_unlearned_trials - n_guessed,
            )
        )

    return shapes


def test_vectorized_updates_match_scalar_sufficient_statistics() -> None:
    responses = np.array(
        [
            [1, 0, -1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, -1, 0, 1, 1],
            [1, -1, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, -1, 1, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    learning_states = np.array(
        [
            [0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    skill_assignments = np.array([0, 1, 0, 2, 1, 0, 2, 1], dtype=np.int32)
    priors = BKTPriors(
        p_init=(1.5, 2.5),
        p_learn=(2.0, 3.0),
        p_forget=(3.0, 4.0),
        p_slip=(4.0, 5.0),
        p_guess=(5.0, 6.0),
    )
    sampler = BKTGibbsSampler(n_iter=2, burnin=1, priors=priors, use_rust=False)
    model = BKTModel(n_skills=4, allow_forgetting=True, use_rust=False)
    recorder = _RecordingRNG()
    rng = cast(np.random.Generator, recorder)
    skill_trials = model._skill_trials(skill_assignments)

    sampler._sample_p_init(model, learning_states, skill_assignments, rng, skill_trials)
    sampler._sample_p_learn(
        model, learning_states, skill_assignments, rng, skill_trials
    )
    sampler._sample_p_forget(
        model, learning_states, skill_assignments, rng, skill_trials
    )
    sampler._sample_p_slip(
        model,
        responses,
        learning_states,
        skill_assignments,
        rng,
        skill_trials,
    )
    sampler._sample_p_guess(
        model,
        responses,
        learning_states,
        skill_assignments,
        rng,
        skill_trials,
    )

    expected = _reference_beta_shapes(
        priors,
        responses,
        learning_states,
        skill_assignments,
        model.n_skills,
    )
    assert_allclose(recorder.beta_shapes, expected)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_iter": 0}, ValueError, "positive integer"),
        ({"n_iter": True}, ValueError, "positive integer"),
        ({"burnin": -1}, ValueError, "non-negative integer"),
        ({"n_iter": 2, "burnin": 2}, ValueError, "less than n_iter"),
        ({"thin": 0}, ValueError, "positive integer"),
        ({"priors": object()}, TypeError, "BKTPriors"),
        ({"verbose": 1}, TypeError, "boolean"),
        ({"seed": -1}, ValueError, "non-negative integer"),
        ({"seed": True}, ValueError, "non-negative integer"),
    ],
)
def test_sampler_configuration_is_validated(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        BKTGibbsSampler(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_init": (0.0, 1.0)},
        {"p_learn": (np.inf, 1.0)},
        {"p_forget": (1.0,)},
        {"p_slip": "invalid"},
        {"p_guess": (1.0, np.nan)},
    ],
)
def test_beta_prior_shapes_are_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="two finite positive values"):
        BKTPriors(**kwargs)


@pytest.mark.parametrize(
    ("responses", "skills", "fit_kwargs", "match"),
    [
        (np.array([1, 0]), np.array([0, 0]), {}, "shape"),
        (np.empty((0, 2), dtype=int), np.array([0, 0]), {}, "one person"),
        (np.empty((1, 0), dtype=int), np.array([], dtype=int), {}, "one trial"),
        (np.array([[1.0, 0.0]]), np.array([0, 0]), {}, "integer values"),
        (np.array([[1, 2]]), np.array([0, 0]), {}, "only -1, 0, or 1"),
        (np.array([[1, 0]]), np.array([[0, 0]]), {}, "one-dimensional"),
        (np.array([[1, 0]]), np.array([0]), {}, "number of trials"),
        (np.array([[1, 0]]), np.array([0.0, 0.0]), {}, "integer values"),
        (np.array([[1, 0]]), np.array([0, -1]), {}, "non-negative"),
        (np.array([[-1, -1]]), np.array([0, 0]), {}, "observed value"),
        (np.array([[1, 0]]), np.array([0, 1]), {"n_skills": 1}, "must be in"),
    ],
)
def test_fit_inputs_are_validated_before_sampling(
    responses: np.ndarray,
    skills: np.ndarray,
    fit_kwargs: dict[str, Any],
    match: str,
) -> None:
    sampler = BKTGibbsSampler(n_iter=2, burnin=1, use_rust=False)

    with pytest.raises(ValueError, match=match):
        sampler.fit(responses, skills, **fit_kwargs)


def test_allow_forgetting_must_be_boolean() -> None:
    sampler = BKTGibbsSampler(n_iter=2, burnin=1, use_rust=False)

    with pytest.raises(TypeError, match="allow_forgetting"):
        sampler.fit(
            np.array([[1, 0]], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            allow_forgetting=1,
        )
