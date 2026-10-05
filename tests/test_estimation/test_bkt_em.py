"""Tests for maximum-likelihood BKT estimation by Baum-Welch EM."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.estimation import bkt_em
from mirt.estimation.bkt_em import BKTEMResult, fit_bkt_em
from mirt.estimation.dynamic_gibbs import BKTGibbsSampler
from mirt.exceptions import MirtValidationError
from mirt.models.dynamic import BKTModel, BKTResult

_PARAMETERS = ("p_init", "p_learn", "p_forget", "p_slip", "p_guess")


def _truth(*, forgetting: bool) -> dict[str, np.ndarray]:
    truth = {
        "p_init": np.array([0.2, 0.35, 0.5]),
        "p_learn": np.array([0.1, 0.2, 0.3]),
        "p_slip": np.array([0.05, 0.1, 0.15]),
        "p_guess": np.array([0.15, 0.2, 0.3]),
    }
    if forgetting:
        truth["p_forget"] = np.array([0.02, 0.05, 0.08])
    return truth


def _small_model(*, forgetting: bool, use_rust: bool = False) -> BKTModel:
    return BKTModel(
        n_skills=3,
        allow_forgetting=forgetting,
        p_init=np.array([0.2, 0.55, 0.8]),
        p_learn=np.array([0.25, 0.12, 0.05]),
        p_forget=np.array([0.02, 0.08, 0.15]) if forgetting else None,
        p_slip=np.array([0.08, 0.15, 0.22]),
        p_guess=np.array([0.12, 0.25, 0.35]),
        use_rust=use_rust,
    )


def _interleave(
    responses: np.ndarray, skills: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Give every learner a random trial order that keeps each skill's order."""
    rng = np.random.default_rng(seed)
    blocked_order = np.argsort(skills, kind="stable")
    layouts = np.empty(responses.shape, dtype=np.int32)
    interleaved = np.empty_like(responses)
    for person in range(responses.shape[0]):
        layouts[person] = rng.permutation(skills)
        order = np.argsort(layouts[person], kind="stable")
        interleaved[person, order] = responses[person, blocked_order]
    return interleaved, layouts


def _reference_update(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
    slip_bounds: tuple[float, float],
    guess_bounds: tuple[float, float],
) -> tuple[float, dict[str, np.ndarray]]:
    """One textbook Baum-Welch step from per-learner scalar recursions."""
    n_skills = model.n_skills
    sums = {
        name: np.zeros(n_skills)
        for name in (
            "init",
            "chains",
            "learned",
            "unlearned_before",
            "forgot",
            "learned_before",
            "slips",
            "learned_observed",
            "guesses",
            "unlearned_observed",
        )
    }
    log_likelihood = 0.0
    layouts = np.broadcast_to(skills, responses.shape)
    for person_responses, layout in zip(responses, layouts, strict=True):
        alpha, scaling = model._forward_python(person_responses, layout)
        beta = model._backward_python(person_responses, layout, scaling)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        log_likelihood += float(np.sum(np.log(scaling)))
        for skill in range(n_skills):
            trials = np.flatnonzero(layout == skill)
            if trials.size == 0:
                continue
            sums["init"][skill] += gamma[trials[0], 1]
            sums["chains"][skill] += 1.0
            transition = model.transition_matrix(skill)
            for previous, trial in itertools.pairwise(trials):
                emission = model._emission_pair(int(person_responses[trial]), skill)
                xi = (
                    alpha[previous][:, None]
                    * transition
                    * (emission * beta[trial])[None, :]
                    / scaling[trial]
                )
                sums["learned"][skill] += xi[0, 1]
                sums["unlearned_before"][skill] += gamma[previous, 0]
                sums["forgot"][skill] += xi[1, 0]
                sums["learned_before"][skill] += gamma[previous, 1]
            for trial in trials:
                response = person_responses[trial]
                if response < 0:
                    continue
                sums["slips"][skill] += gamma[trial, 1] * (response == 0)
                sums["learned_observed"][skill] += gamma[trial, 1]
                sums["guesses"][skill] += gamma[trial, 0] * (response == 1)
                sums["unlearned_observed"][skill] += gamma[trial, 0]

    def ratio(numerator: str, denominator: str, current: np.ndarray) -> np.ndarray:
        total = sums[denominator]
        return np.where(
            total > 0, sums[numerator] / np.where(total > 0, total, 1), current
        )

    update = {
        "p_init": ratio("init", "chains", model.p_init),
        "p_learn": ratio("learned", "unlearned_before", model.p_learn),
        "p_slip": np.clip(
            ratio("slips", "learned_observed", model.p_slip), *slip_bounds
        ),
        "p_guess": np.clip(
            ratio("guesses", "unlearned_observed", model.p_guess), *guess_bounds
        ),
    }
    if model.allow_forgetting:
        update["p_forget"] = ratio("forgot", "learned_before", model.p_forget)
    return log_likelihood, update


def _em_update(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
    slip_bounds: tuple[float, float] = (1e-4, 0.5),
    guess_bounds: tuple[float, float] = (1e-4, 0.5),
) -> tuple[float, dict[str, np.ndarray]]:
    layout = bkt_em._ChainLayout.from_skills(skills, model.n_skills)
    counts = bkt_em._ResponseCounts.from_responses(responses, layout)
    return bkt_em._em_update(
        model, responses, layout, counts, slip_bounds, guess_bounds
    )


def _assert_fits_match(first: BKTResult, second: BKTResult) -> None:
    for name in _PARAMETERS:
        assert_allclose(
            getattr(first.model, name),
            getattr(second.model, name),
            rtol=0,
            atol=1e-10,
            err_msg=name,
        )
    assert first.log_likelihood == pytest.approx(second.log_likelihood, rel=1e-12)


@pytest.mark.parametrize("person_specific", [False, True])
def test_transition_posteriors_match_path_enumeration(person_specific: bool) -> None:
    model = _small_model(forgetting=True)
    responses = np.array(
        [[1, 0, -1, 1, 0, 1, 1, 0], [0, 1, 1, -1, 1, 0, 0, 1]], dtype=np.int32
    )
    skills = np.array([0, 1, 0, 2, 1, 0, 2, 0], dtype=np.int32)
    if person_specific:
        skills = np.array([skills, [2, 2, 0, 1, 0, 2, 1, 1]], dtype=np.int32)
    transitions = np.full((2, responses.shape[1], responses.shape[0]), np.nan)

    gamma, log_likelihoods = model._forward_backward_layouts(
        responses, skills, transition_out=transitions
    )
    plain_gamma, plain_log_likelihoods = model._forward_backward_layouts(
        responses, skills
    )

    assert_array_equal(gamma, plain_gamma)
    assert_array_equal(log_likelihoods, plain_log_likelihoods)
    layouts = np.broadcast_to(skills, responses.shape)
    for person, layout in enumerate(layouts):
        for skill in range(model.n_skills):
            trials = np.flatnonzero(layout == skill)
            transition = model.transition_matrix(skill)
            joint = np.zeros((trials.size, 2, 2))
            for path in itertools.product((0, 1), repeat=trials.size):
                weight = model.p_init[skill] if path[0] else 1 - model.p_init[skill]
                for step, (trial, state) in enumerate(zip(trials, path, strict=True)):
                    response = int(responses[person, trial])
                    weight *= model._emission_pair(response, skill)[state]
                    if step:
                        weight *= transition[path[step - 1], state]
                for step in range(1, trials.size):
                    joint[step, path[step - 1], path[step]] += weight
            for step in range(1, trials.size):
                pair = joint[step] / joint[step].sum()
                learned_rate = pair[0, 1] / pair[0].sum()
                forgotten_rate = pair[1, 0] / pair[1].sum()
                assert transitions[0, trials[step], person] == pytest.approx(
                    learned_rate, rel=1e-12
                )
                assert transitions[1, trials[step], person] == pytest.approx(
                    forgotten_rate, rel=1e-12
                )


@pytest.mark.parametrize("forgetting", [False, True])
@pytest.mark.parametrize("person_specific", [False, True])
def test_em_update_matches_textbook_baum_welch(
    forgetting: bool, person_specific: bool
) -> None:
    model = _small_model(forgetting=forgetting)
    responses, skills, _ = model.simulate(40, 6, seed=5)
    responses[::4, ::5] = -1
    if person_specific:
        responses, skills = _interleave(responses, skills, seed=8)
    bounds = (1e-4, 0.13)

    log_likelihood, update = _em_update(model, responses, skills, bounds, bounds)
    expected_log_likelihood, expected = _reference_update(
        model, responses, skills, bounds, bounds
    )

    assert log_likelihood == pytest.approx(expected_log_likelihood, rel=1e-12)
    assert set(update) == set(expected)
    for name, values in expected.items():
        assert_allclose(update[name], values, rtol=1e-10, atol=1e-13, err_msg=name)


@pytest.mark.parametrize("forgetting", [False, True])
def test_em_solution_is_a_stationary_point_of_the_log_likelihood(
    forgetting: bool,
) -> None:
    generator = BKTModel(
        n_skills=2,
        allow_forgetting=forgetting,
        p_init=np.array([0.3, 0.5]),
        p_learn=np.array([0.2, 0.15]),
        p_forget=np.array([0.05, 0.1]) if forgetting else None,
        p_slip=np.array([0.1, 0.15]),
        p_guess=np.array([0.2, 0.25]),
    )
    responses, skills, _ = generator.simulate(150, 6, seed=11)
    responses[::7, ::3] = -1
    responses, layouts = _interleave(responses, skills, seed=1)

    result = fit_bkt_em(
        responses, layouts, allow_forgetting=forgetting, tol=1e-10, max_iter=5000
    )

    # Independently of the Baum-Welch algebra, an interior maximum of the
    # observed-data log-likelihood has a zero gradient in every parameter.
    names = _PARAMETERS if forgetting else tuple(_PARAMETERS[i] for i in (0, 1, 3, 4))
    start = BKTModel(n_skills=2, allow_forgetting=forgetting, use_rust=False)
    fitted = result.model
    assert result.converged
    step = 1e-6
    for model, gradient_bound in ((fitted, 1e-2), (start, None)):
        gradients = []
        for name in names:
            values = getattr(model, name)
            upper = 0.499 if name in ("p_slip", "p_guess") else 0.999
            assert np.all((values > 1e-3) & (values < upper)), name
            for skill in range(2):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = values.copy()
                    moved[skill] += sign * step
                    setattr(model, name, moved)
                    shifted.append(
                        model._forward_backward_layouts(responses, layouts)[1].sum()
                    )
                setattr(model, name, values)
                gradients.append((shifted[0] - shifted[1]) / (2.0 * step))
        if gradient_bound is None:
            assert np.max(np.abs(gradients)) > 10.0
        else:
            assert np.max(np.abs(gradients)) < gradient_bound


def test_forgetting_e_step_matches_telescoped_learning_without_forgetting() -> None:
    reference = _small_model(forgetting=False)
    with_forgetting = BKTModel(
        n_skills=3,
        allow_forgetting=True,
        p_init=reference.p_init,
        p_learn=reference.p_learn,
        p_forget=np.zeros(3),
        p_slip=reference.p_slip,
        p_guess=reference.p_guess,
        use_rust=False,
    )
    responses, skills, _ = reference.simulate(30, 5, seed=2)
    responses, layouts = _interleave(responses, skills, seed=3)

    log_likelihood, update = _em_update(reference, responses, layouts)
    forgetting_log_likelihood, forgetting_update = _em_update(
        with_forgetting, responses, layouts
    )

    assert log_likelihood == pytest.approx(forgetting_log_likelihood, rel=1e-12)
    assert_allclose(forgetting_update["p_forget"], 0.0, atol=1e-15)
    for name, values in update.items():
        assert_allclose(forgetting_update[name], values, rtol=1e-10, err_msg=name)


@pytest.mark.parametrize("forgetting", [False, True])
def test_person_specific_interleavings_match_shared_layout(forgetting: bool) -> None:
    generator = BKTModel(
        n_skills=3, allow_forgetting=forgetting, **_truth(forgetting=forgetting)
    )
    responses, skills, _ = generator.simulate(300, 8, seed=4)
    responses[::6, ::7] = -1
    interleaved, layouts = _interleave(responses, skills, seed=12)

    shared = fit_bkt_em(responses, skills, allow_forgetting=forgetting)
    person = fit_bkt_em(interleaved, layouts, allow_forgetting=forgetting)

    # Each learner's per-skill chains are unchanged, so EM follows the same
    # path whichever kernel smooths each layout.
    assert person.n_iterations == shared.n_iterations
    assert_allclose(
        person.log_likelihood_history, shared.log_likelihood_history, rtol=1e-12
    )
    _assert_fits_match(shared, person)
    assert shared.converged and person.converged
    assert_allclose(person.learning_curves, shared.learning_curves, atol=1e-10)
    assert_allclose(person.skill_mastery, shared.skill_mastery, atol=1e-10)


def test_identical_person_rows_use_the_shared_layout() -> None:
    generator = BKTModel(n_skills=3, **_truth(forgetting=False))
    responses, skills, _ = generator.simulate(200, 6, seed=6)

    shared = fit_bkt_em(responses, skills)
    repeated = fit_bkt_em(responses, np.tile(skills, (responses.shape[0], 1)))

    _assert_fits_match(shared, repeated)
    assert_array_equal(repeated.log_likelihood_history, shared.log_likelihood_history)


@pytest.mark.parametrize("forgetting", [False, True])
def test_compiled_and_numpy_e_steps_agree(forgetting: bool) -> None:
    generator = BKTModel(
        n_skills=3, allow_forgetting=forgetting, **_truth(forgetting=forgetting)
    )
    responses, skills, _ = generator.simulate(150, 6, seed=9)

    compiled = fit_bkt_em(responses, skills, allow_forgetting=forgetting)
    numpy = fit_bkt_em(responses, skills, allow_forgetting=forgetting, use_rust=False)

    assert numpy.n_iterations == compiled.n_iterations
    _assert_fits_match(compiled, numpy)
    assert compiled.model.use_rust is True
    assert numpy.model.use_rust is False


def test_recovers_parameters_without_forgetting() -> None:
    truth = _truth(forgetting=False)
    generator = BKTModel(n_skills=3, **truth)
    responses, skills, _ = generator.simulate(2000, 10, seed=2)

    result = fit_bkt_em(responses, skills)

    assert isinstance(result, BKTEMResult)
    assert result.converged
    assert result.n_iterations < 500
    for name, values in truth.items():
        assert_allclose(getattr(result.model, name), values, atol=0.03, err_msg=name)
    assert_array_equal(result.model.p_forget, 0.0)


def test_recovers_parameters_with_forgetting() -> None:
    truth = _truth(forgetting=True)
    truth = {name: values[:2] for name, values in truth.items()}
    generator = BKTModel(n_skills=2, allow_forgetting=True, **truth)
    responses, skills, _ = generator.simulate(2000, 20, seed=6)

    result = fit_bkt_em(responses, skills, allow_forgetting=True)

    assert result.converged
    assert result.n_parameters == 10
    for name, values in truth.items():
        assert_allclose(getattr(result.model, name), values, atol=0.03, err_msg=name)


@pytest.mark.parametrize("forgetting", [False, True])
def test_log_likelihood_never_decreases(forgetting: bool) -> None:
    generator = BKTModel(
        n_skills=3, allow_forgetting=forgetting, **_truth(forgetting=forgetting)
    )
    responses, skills, _ = generator.simulate(300, 8, seed=1)
    responses[::5, ::3] = -1
    interleaved, layouts = _interleave(responses, skills, seed=2)

    result = fit_bkt_em(interleaved, layouts, allow_forgetting=forgetting)

    history = result.log_likelihood_history
    assert history.size == result.n_iterations + 1
    assert np.all(np.diff(history) >= -1e-9 * np.abs(history[1:]))
    assert history[-1] > history[0]
    assert result.log_likelihood == pytest.approx(history[-1], rel=1e-10)


def test_iteration_limit_reports_nonconvergence() -> None:
    generator = BKTModel(n_skills=3, **_truth(forgetting=False))
    responses, skills, _ = generator.simulate(300, 8, seed=1)

    result = fit_bkt_em(responses, skills, max_iter=3)

    assert not result.converged
    assert result.n_iterations == 3
    assert result.log_likelihood_history.size == 4
    assert result.log_likelihood == pytest.approx(
        result.log_likelihood_history[-1], rel=1e-10
    )


def test_agrees_with_gibbs_posterior_means() -> None:
    truth = {
        "p_init": np.array([0.25, 0.4]),
        "p_learn": np.array([0.15, 0.25]),
        "p_slip": np.array([0.08, 0.12]),
        "p_guess": np.array([0.2, 0.25]),
    }
    generator = BKTModel(n_skills=2, **truth)
    responses, skills, _ = generator.simulate(600, 8, seed=3)

    maximum_likelihood = fit_bkt_em(responses, skills)
    posterior = BKTGibbsSampler(n_iter=600, burnin=150, seed=3).fit(responses, skills)

    # With flat priors and 600 learners the posterior means sit within
    # Monte Carlo error (well under 0.005 here) of the maximum.
    for name in truth:
        assert_allclose(
            getattr(maximum_likelihood.model, name),
            getattr(posterior.model, name),
            atol=0.01,
            err_msg=name,
        )
    assert maximum_likelihood.log_likelihood >= posterior.log_likelihood
    assert maximum_likelihood.n_parameters == posterior.n_parameters


def test_bounds_clip_and_fix_slip_and_guess() -> None:
    generator = BKTModel(n_skills=3, **_truth(forgetting=False))
    responses, skills, _ = generator.simulate(500, 8, seed=7)

    result = fit_bkt_em(
        responses,
        skills,
        slip_bounds=(1e-4, 0.03),
        guess_bounds=(0.2, 0.2),
    )

    assert_allclose(result.model.p_slip, 0.03)
    assert_allclose(result.model.p_guess, 0.2)
    assert result.converged
    assert np.all(np.diff(result.log_likelihood_history) > -1e-9)
    # The fixed guess probabilities are not estimated; the clipped slips are.
    assert result.n_parameters == 9
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood + 18.0)
    assert result.bic == pytest.approx(
        -2.0 * result.log_likelihood + np.log(result.n_observations) * 9
    )


def test_fixed_slip_and_guess_with_forgetting_leave_three_parameters() -> None:
    generator = BKTModel(n_skills=2, allow_forgetting=True)
    responses, skills, _ = generator.simulate(100, 6, seed=3)

    result = fit_bkt_em(
        responses,
        skills,
        allow_forgetting=True,
        slip_bounds=(0.1, 0.1),
        guess_bounds=(0.25, 0.25),
    )

    assert result.n_parameters == 6
    assert_allclose(result.model.p_slip, 0.1)
    assert_allclose(result.model.p_guess, 0.25)


def test_bounds_protect_against_label_swapped_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = {
        "p_init": np.array([0.2, 0.35]),
        "p_learn": np.array([0.1, 0.2]),
        "p_slip": np.array([0.05, 0.1]),
        "p_guess": np.array([0.15, 0.2]),
    }
    generator = BKTModel(n_skills=2, **truth)
    responses, skills, _ = generator.simulate(2000, 10, seed=2)
    default_start = bkt_em._start_model

    def swapped_first_start(template: BKTModel, start: int, *args: Any) -> BKTModel:
        if start:
            return default_start(template, start, *args)
        return BKTModel(
            n_skills=2,
            p_init=np.array([0.8, 0.65]),
            p_learn=np.array([0.1, 0.1]),
            p_slip=np.array([0.85, 0.8]),
            p_guess=np.array([0.95, 0.9]),
        )

    monkeypatch.setattr(bkt_em, "_start_model", swapped_first_start)
    relaxed = (1e-4, 0.99)

    protected = fit_bkt_em(responses, skills)
    swapped = fit_bkt_em(responses, skills, slip_bounds=relaxed, guess_bounds=relaxed)
    restarted = fit_bkt_em(
        responses,
        skills,
        slip_bounds=relaxed,
        guess_bounds=relaxed,
        n_starts=3,
        seed=0,
    )

    for name, values in truth.items():
        assert_allclose(getattr(protected.model, name), values, atol=0.03)
    assert np.all(swapped.model.p_guess > 0.5)
    assert swapped.log_likelihood < protected.log_likelihood - 100.0
    assert restarted.start_log_likelihoods.shape == (3,)
    assert restarted.start_log_likelihoods[0] == pytest.approx(swapped.log_likelihood)
    assert restarted.log_likelihood == pytest.approx(protected.log_likelihood)
    assert restarted.log_likelihood == pytest.approx(
        restarted.start_log_likelihoods.max(), rel=1e-10
    )


def test_random_starts_are_seeded_and_respect_bounds() -> None:
    generator = BKTModel(n_skills=3, **_truth(forgetting=False))
    responses, skills, _ = generator.simulate(300, 6, seed=3)
    template = BKTModel(n_skills=3, allow_forgetting=True)
    rng = np.random.default_rng(0)

    starts = [
        bkt_em._start_model(template, start, rng, (0.01, 0.3), (0.05, 0.6))
        for start in range(20)
    ]
    first = fit_bkt_em(responses, skills, n_starts=4, seed=11)
    second = fit_bkt_em(responses, skills, n_starts=4, seed=11)

    for model in starts:
        assert np.all((model.p_slip >= 0.01) & (model.p_slip <= 0.3))
        assert np.all((model.p_guess >= 0.05) & (model.p_guess <= 0.4))
        assert np.all(model.p_forget > 0.0)
    assert_allclose(starts[0].p_init, 0.3)
    assert_array_equal(first.start_log_likelihoods, second.start_log_likelihoods)
    _assert_fits_match(first, second)


def test_missing_responses_and_unpracticed_skills() -> None:
    generator = BKTModel(n_skills=2, p_learn=np.array([0.2, 0.3]))
    responses, skills, _ = generator.simulate(400, 6, seed=5)
    responses[::3, 1::4] = -1

    result = fit_bkt_em(responses, skills, n_skills=3, skill_names=["a", "b", "c"])

    assert result.model.skill_names == ["a", "b", "c"]
    assert result.n_observations == int(np.count_nonzero(responses >= 0))
    assert result.n_parameters == 12
    # The unpracticed skill receives no expected data and keeps its start.
    assert result.model.p_init[2] == pytest.approx(0.3)
    assert result.model.p_learn[2] == pytest.approx(0.1)
    assert_array_equal(result.learning_curves[:, 2], 0.0)
    assert_array_equal(result.skill_mastery[:, 2], 0.0)
    assert result.converged


def test_result_fields_match_per_skill_reference() -> None:
    model = _small_model(forgetting=True)
    responses, skills, _ = model.simulate(25, 4, seed=1)
    responses[::4, ::3] = -1

    fields = bkt_em._bkt_result_fields(model, responses, skills)

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)
    for skill in range(model.n_skills):
        learned = gamma[:, skills == skill, 1]
        assert_allclose(fields["skill_mastery"][:, skill], learned[:, -1])
        assert_allclose(fields["learning_curves"][:, skill], learned.mean(axis=1))
    log_likelihood = log_likelihoods.sum()
    n_observations = np.count_nonzero(responses >= 0)
    assert fields["log_likelihood"] == pytest.approx(log_likelihood)
    assert fields["n_parameters"] == 15
    assert fields["n_observations"] == n_observations
    assert fields["aic"] == pytest.approx(-2 * log_likelihood + 30)
    assert fields["bic"] == pytest.approx(
        -2 * log_likelihood + np.log(n_observations) * 15
    )

    interleaved, layouts = _interleave(responses, skills, seed=4)
    person_fields = bkt_em._bkt_result_fields(model, interleaved, layouts)
    assert_allclose(person_fields["skill_mastery"], fields["skill_mastery"])
    assert_allclose(person_fields["learning_curves"], fields["learning_curves"])


def test_result_summary_reports_em_diagnostics() -> None:
    generator = BKTModel(n_skills=2)
    responses, skills, _ = generator.simulate(100, 5, seed=0)

    result = fit_bkt_em(responses, skills, n_starts=2, seed=1)
    summary = result.summary()

    assert isinstance(result, BKTResult)
    assert f"Iterations:         {result.n_iterations}" in summary
    assert "Starts:             2" in summary
    assert summary.index("Converged:") < summary.index("Iterations:")


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"max_iter": 0}, MirtValidationError, "max_iter"),
        ({"max_iter": True}, MirtValidationError, "max_iter"),
        ({"n_starts": 0}, MirtValidationError, "n_starts"),
        ({"tol": 0.0}, MirtValidationError, "tol"),
        ({"tol": np.nan}, MirtValidationError, "tol"),
        ({"tol": True}, MirtValidationError, "tol"),
        ({"seed": -1}, MirtValidationError, "seed"),
        ({"slip_bounds": (0.0, 0.5)}, MirtValidationError, "slip_bounds"),
        ({"slip_bounds": (0.3, 0.2)}, MirtValidationError, "slip_bounds"),
        ({"guess_bounds": (0.1, 1.0)}, MirtValidationError, "guess_bounds"),
        ({"guess_bounds": "low"}, MirtValidationError, "guess_bounds"),
        ({"guess_bounds": (0.1, 0.2, 0.3)}, MirtValidationError, "guess_bounds"),
        ({"use_rust": 1}, TypeError, "use_rust"),
        ({"allow_forgetting": 1}, TypeError, "allow_forgetting"),
        ({"n_skills": 1}, ValueError, "must be in"),
    ],
)
def test_options_are_validated(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    responses = np.array([[1, 0, 1], [0, 1, 1]])
    skills = np.array([0, 1, 0])

    with pytest.raises(error, match=match):
        fit_bkt_em(responses, skills, **kwargs)


@pytest.mark.parametrize(
    ("responses", "skills", "match"),
    [
        (np.array([1, 0]), np.array([0, 0]), "shape"),
        (np.empty((0, 2), dtype=int), np.array([0, 0]), "one person"),
        (np.array([[1, 0]]), np.array([0]), "number of trials"),
        (np.array([[1, 0]]), np.array([[0, 0], [0, 0]]), "match responses"),
        (np.array([[1, 0]]), np.empty((1, 0), dtype=int), "match responses"),
        (np.array([[1, 0]]), np.zeros((1, 2, 1), dtype=int), "match responses"),
        (np.array([[1, 0]]), np.array([[0, -1]]), "non-negative"),
        (np.array([[-1, -1]]), np.array([0, 0]), "observed value"),
        (np.array([[1, 2]]), np.array([0, 0]), "only -1, 0, or 1"),
    ],
)
def test_data_is_validated(
    responses: np.ndarray, skills: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_bkt_em(responses, skills)


def test_public_exports_resolve_to_the_same_objects() -> None:
    import mirt
    import mirt.estimation as estimation
    from mirt.estimation.dynamic_gibbs import BKTPriors

    assert mirt.fit_bkt_em is estimation.fit_bkt_em is fit_bkt_em
    assert mirt.BKTEMResult is estimation.BKTEMResult is BKTEMResult
    assert mirt.BKTGibbsSampler is estimation.BKTGibbsSampler is BKTGibbsSampler
    assert mirt.BKTPriors is estimation.BKTPriors is BKTPriors
