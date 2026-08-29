"""Regression contracts for independent-skill knowledge tracing."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.backends.rust.dynamic import (
    bkt_backward,
    bkt_ffbs,
    bkt_ffbs_batch,
    bkt_forward,
    bkt_forward_backward_batch,
    bkt_viterbi,
)
from mirt.estimation.dynamic_gibbs import BKTGibbsSampler
from mirt.models.dynamic import BKTModel


@pytest.fixture
def model() -> BKTModel:
    return BKTModel(
        n_skills=3,
        allow_forgetting=True,
        p_init=np.array([0.2, 0.8, 0.45]),
        p_learn=np.array([0.3, 0.05, 0.15]),
        p_forget=np.array([0.02, 0.2, 0.08]),
        p_slip=np.array([0.1, 0.25, 0.15]),
        p_guess=np.array([0.15, 0.4, 0.2]),
    )


@pytest.fixture
def interleaved_sequence() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([1, 0, 1, -1, 0, 1, 1, 0, 1], dtype=np.int_),
        np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int_),
    )


def test_interleaved_inference_matches_isolated_skill_chains(
    model: BKTModel,
    interleaved_sequence: tuple[np.ndarray, np.ndarray],
) -> None:
    responses, skills = interleaved_sequence
    alpha, scaling = model.forward(responses, skills)
    gamma, log_likelihood = model.forward_backward(responses, skills)
    path = model.viterbi(responses, skills)

    isolated_log_likelihood = 0.0
    for skill_idx in range(model.n_skills):
        mask = skills == skill_idx
        isolated_alpha, isolated_scaling = model.forward(responses[mask], skills[mask])
        isolated_gamma, isolated_ll = model.forward_backward(
            responses[mask], skills[mask]
        )
        isolated_path = model.viterbi(responses[mask], skills[mask])

        assert_allclose(alpha[mask], isolated_alpha)
        assert_allclose(scaling[mask], isolated_scaling)
        assert_allclose(gamma[mask], isolated_gamma)
        assert_array_equal(path[mask], isolated_path)
        isolated_log_likelihood += isolated_ll

    assert_allclose(log_likelihood, isolated_log_likelihood)


def test_missing_response_has_neutral_emission(model: BKTModel) -> None:
    assert model.emission_probability(-1, learned=0, skill_idx=0) == 1.0
    assert model.emission_probability(-1, learned=1, skill_idx=0) == 1.0

    alpha, scaling = model.forward(np.array([-1]), np.array([2]))
    assert_allclose(alpha[0], [1.0 - model.p_init[2], model.p_init[2]])
    assert scaling[0] == pytest.approx(1.0)


def test_per_skill_mastery_retains_unobserved_initial_probability(
    model: BKTModel,
) -> None:
    responses = np.array([1, 0, 1, 1])
    skills = np.array([0, 1, 0, 1])

    mastery = model.predict_mastery_by_skill(responses, skills)

    assert mastery.shape == (3,)
    assert mastery[2] == model.p_init[2]
    for skill_idx in (0, 1):
        mask = skills == skill_idx
        isolated = model.predict_mastery(responses[mask], skills[mask])
        assert mastery[skill_idx] == pytest.approx(isolated)


def test_batch_mastery_supports_shared_and_person_specific_layouts(
    model: BKTModel,
) -> None:
    responses = np.array([[1, 0, 1, 1], [0, 1, -1, 0]])
    shared_skills = np.array([0, 1, 0, 1])
    person_skills = np.array([[0, 1, 0, 1], [2, 0, 2, 0]])

    shared = model.predict_mastery_batch(responses, shared_skills)
    specific = model.predict_mastery_batch(responses, person_skills)

    assert shared.shape == specific.shape == (2, 3)
    assert_allclose(
        shared[0], model.predict_mastery_by_skill(responses[0], shared_skills)
    )
    assert_allclose(
        specific[1], model.predict_mastery_by_skill(responses[1], person_skills[1])
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_skills": 0}, "positive integer"),
        ({"n_skills": 2, "skill_names": ["only-one"]}, "length"),
        ({"n_skills": 2, "p_init": np.array([0.2])}, "shape"),
        ({"n_skills": 1, "p_guess": np.array([1.2])}, "in \\[0, 1\\]"),
        ({"n_skills": 1, "p_forget": np.array([0.1])}, "allow_forgetting"),
    ],
)
def test_model_parameters_are_validated(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BKTModel(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "responses,skills,match",
    [
        (np.array([], dtype=int), np.array([], dtype=int), "at least one"),
        (np.array([1, 0]), np.array([0]), "equal length"),
        (np.array([1, 2]), np.array([0, 0]), "only -1, 0, or 1"),
        (np.array([1, 0]), np.array([0, 3]), "must be in"),
    ],
)
def test_trial_sequences_are_validated(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model.forward(responses, skills)


def test_native_kernels_match_python_for_interleaved_skills(
    model: BKTModel,
    interleaved_sequence: tuple[np.ndarray, np.ndarray],
) -> None:
    responses, skills = interleaved_sequence
    python_alpha, python_scaling = model.forward(responses, skills)
    python_gamma, python_ll = model.forward_backward(responses, skills)
    python_path = model.viterbi(responses, skills)

    native_forward = bkt_forward(
        responses,
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )
    if native_forward is None:
        pytest.skip("native backend unavailable")
    native_alpha, native_scaling = native_forward
    native_beta = bkt_backward(
        responses,
        skills,
        native_scaling,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )
    native_path = bkt_viterbi(
        responses,
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )

    assert native_beta is not None
    assert native_path is not None
    native_gamma = native_alpha * native_beta
    native_gamma /= native_gamma.sum(axis=1, keepdims=True)
    assert_allclose(native_alpha, python_alpha)
    assert_allclose(native_scaling, python_scaling)
    assert_allclose(native_gamma, python_gamma)
    assert_array_equal(native_path, python_path)

    native_batch = bkt_forward_backward_batch(
        np.vstack([responses, responses[::-1]]),
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )
    assert native_batch is not None
    learned, log_likelihoods = native_batch
    assert_allclose(learned[0], python_gamma[:, 1])
    assert log_likelihoods[0] == pytest.approx(python_ll)

    sampled = bkt_ffbs(
        responses,
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
        seed=41,
    )
    sampled_batch = bkt_ffbs_batch(
        responses[None, :],
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
        seed=41,
    )
    assert sampled is not None
    assert sampled_batch is not None
    assert_array_equal(sampled, sampled_batch[0])


def test_vectorized_simulation_is_reproducible(model: BKTModel) -> None:
    first = model.simulate(100, 8, seed=42)
    second = model.simulate(100, 8, seed=42)

    for first_array, second_array in zip(first, second, strict=True):
        assert_array_equal(first_array, second_array)
    responses, skills, states = first
    assert responses.shape == states.shape == (100, 24)
    assert skills.shape == (24,)
    assert set(np.unique(responses)).issubset({0, 1})


def test_gibbs_sampler_handles_interleaved_skills_and_missing_data(
    model: BKTModel,
) -> None:
    blocked_responses, blocked_skills, _ = model.simulate(12, 4, seed=7)
    order = np.arange(blocked_skills.size).reshape(model.n_skills, -1).T.ravel()
    responses = blocked_responses[:, order]
    skills = blocked_skills[order]
    responses[0, 2] = -1

    result = BKTGibbsSampler(n_iter=20, burnin=10, thin=2, seed=11).fit(
        responses,
        skills,
        n_skills=model.n_skills,
        allow_forgetting=True,
    )

    assert result.skill_mastery.shape == (12, model.n_skills)
    assert result.learning_curves.shape == (12, model.n_skills)
    assert np.isfinite(result.log_likelihood)
    assert result.n_observations == responses.size - 1
