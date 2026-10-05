"""Regression coverage for accelerated BKT inference."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.backends.rust.dynamic as native_dynamic_module
import mirt.models.dynamic as dynamic_module
from mirt._backend_state import get_backend_preference, set_backend_preference
from mirt._rust_backend import RUST_AVAILABLE
from mirt.estimation.dynamic_gibbs import BKTGibbsSampler
from mirt.models.dynamic import BKTModel


@pytest.fixture(autouse=True)
def _restore_backend() -> Iterator[None]:
    previous = get_backend_preference()
    set_backend_preference("auto")
    try:
        yield
    finally:
        set_backend_preference(previous)


def _model(*, use_rust: bool = True) -> BKTModel:
    return BKTModel(
        n_skills=3,
        allow_forgetting=True,
        p_init=np.array([0.2, 0.55, 0.8]),
        p_learn=np.array([0.25, 0.12, 0.05]),
        p_forget=np.array([0.02, 0.08, 0.15]),
        p_slip=np.array([0.08, 0.15, 0.22]),
        p_guess=np.array([0.12, 0.25, 0.35]),
        use_rust=use_rust,
    )


def _batch() -> tuple[np.ndarray, np.ndarray]:
    responses = np.array(
        [
            [1, 0, 1, -1, 1, 0, 1, 1, 0],
            [0, 1, -1, 1, 0, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, -1, 0, 1],
        ],
        dtype=np.int32,
    )
    skills = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return responses, skills


def test_batch_api_matches_individual_python_inference() -> None:
    responses, skills = _batch()
    model = _model(use_rust=False)

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    assert gamma.shape == (*responses.shape, 2)
    assert log_likelihoods.shape == (responses.shape[0],)
    assert_allclose(gamma.sum(axis=2), 1.0)
    for person_idx in range(responses.shape[0]):
        expected_gamma, expected_ll = model.forward_backward(
            responses[person_idx], skills
        )
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_ll)


def _scalar_forward_backward(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Independent per-learner, per-skill recursion used as an oracle.

    A trial with zero scaling keeps its zero filter unnormalized, so the rest
    of that skill's chain stays at zero and contributes ``log(1e-300)``.
    """

    def emission(trial: int, skill: int) -> np.ndarray:
        if responses[trial] < 0:
            return np.ones(2)
        guess, slip = model.p_guess[skill], model.p_slip[skill]
        if responses[trial] == 1:
            return np.array([guess, 1.0 - slip])
        return np.array([1.0 - guess, slip])

    n_trials = len(responses)
    alpha = np.zeros((n_trials, 2))
    beta = np.ones((n_trials, 2))
    scaling = np.zeros(n_trials)
    for skill in range(model.n_skills):
        trials = np.flatnonzero(skills == skill)
        learn, forget = model.p_learn[skill], model.p_forget[skill]
        transition = np.array([[1.0 - learn, learn], [forget, 1.0 - forget]])
        prior = np.array([1.0 - model.p_init[skill], model.p_init[skill]])
        for trial in trials:
            alpha[trial] = prior * emission(trial, skill)
            scaling[trial] = alpha[trial].sum()
            if scaling[trial] > 0.0:
                alpha[trial] /= scaling[trial]
            prior = alpha[trial] @ transition
        for later, earlier in zip(trials[:0:-1], trials[-2::-1], strict=True):
            beta[earlier] = transition @ (emission(later, skill) * beta[later])
            if scaling[later] > 0.0:
                beta[earlier] /= scaling[later]

    gamma = alpha * beta
    total = gamma.sum(axis=1, keepdims=True)
    gamma = np.divide(gamma, total, out=np.zeros_like(gamma), where=total > 0.0)
    return gamma, float(np.sum(np.log(scaling + 1e-300)))


def _assert_matches_scalar_oracle(
    model: BKTModel,
    responses: np.ndarray,
    skills: np.ndarray,
    gamma: np.ndarray,
    log_likelihoods: np.ndarray,
) -> None:
    layouts = np.broadcast_to(skills, responses.shape)
    for person_idx, person_responses in enumerate(responses):
        expected_gamma, expected_ll = _scalar_forward_backward(
            model, person_responses, layouts[person_idx]
        )
        assert_allclose(gamma[person_idx], expected_gamma, rtol=1e-12, atol=1e-12)
        assert_allclose(log_likelihoods[person_idx], expected_ll, rtol=1e-12)


def test_shared_numpy_filter_and_layout_smoother_match_individual_fallbacks() -> None:
    responses, skills = _batch()
    model = _model(use_rust=False)

    alpha, scaling = model._forward_batch_shared_python(
        responses,
        skills,
        model._skill_trials(skills),
    )
    gamma, log_likelihoods = model._forward_backward_layouts(responses, skills)

    for person_idx, person_responses in enumerate(responses):
        expected_alpha, expected_scaling = model._forward_python(
            person_responses,
            skills,
        )
        expected_gamma, expected_log_likelihood = model._forward_backward_python(
            person_responses,
            skills,
        )
        assert_allclose(alpha[person_idx], expected_alpha)
        assert_allclose(scaling[person_idx], expected_scaling)
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_log_likelihood)
    _assert_matches_scalar_oracle(model, responses, skills, gamma, log_likelihoods)


def test_shared_numpy_fallback_avoids_per_person_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _batch()
    model = _model(use_rust=False)
    monkeypatch.setattr(
        model,
        "_forward_backward_python",
        lambda *args, **kwargs: pytest.fail(
            "shared fallback must not loop over learners"
        ),
    )

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    assert gamma.shape == (*responses.shape, 2)
    assert log_likelihoods.shape == (len(responses),)


def test_shared_numpy_fallback_preserves_zero_scaling_rows() -> None:
    model = BKTModel(
        n_skills=1,
        p_init=np.array([0.0]),
        p_learn=np.array([0.0]),
        p_slip=np.array([0.0]),
        p_guess=np.array([0.0]),
        use_rust=False,
    )
    responses = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [-1, 0, 1],
        ],
        dtype=np.int32,
    )
    skills = np.zeros(responses.shape[1], dtype=np.int32)

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    for person_idx, person_responses in enumerate(responses):
        expected_gamma, expected_log_likelihood = model._forward_backward_python(
            person_responses,
            skills,
        )
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_log_likelihood)


def test_batch_api_supports_person_specific_skill_layouts() -> None:
    responses, shared_skills = _batch()
    skill_assignments = np.vstack(
        [
            shared_skills,
            np.roll(shared_skills, 1),
            np.roll(shared_skills, 2),
        ]
    )
    model = _model()

    gamma, log_likelihoods = model.forward_backward_batch(responses, skill_assignments)
    mastery = model.predict_mastery_batch(responses, skill_assignments)

    for person_idx in range(responses.shape[0]):
        expected_gamma, expected_ll = model.forward_backward(
            responses[person_idx], skill_assignments[person_idx]
        )
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_ll)
        assert_allclose(
            mastery[person_idx],
            model.predict_mastery_by_skill(
                responses[person_idx], skill_assignments[person_idx]
            ),
        )


@pytest.mark.parametrize("native_available", [True, False])
def test_person_specific_layouts_are_smoothed_together(
    monkeypatch: pytest.MonkeyPatch,
    native_available: bool,
) -> None:
    responses, shared_skills = _batch()
    responses = np.vstack((responses, responses, responses[:1]))
    rolled_skills = np.roll(shared_skills, 1)
    skill_assignments = np.vstack(
        (
            shared_skills,
            rolled_skills,
            shared_skills,
            rolled_skills,
            shared_skills,
            np.roll(shared_skills, 2),
            shared_skills,
        )
    )
    model = _model(use_rust=native_available)
    monkeypatch.setattr(model, "_can_use_native_inference", lambda: native_available)
    monkeypatch.setattr(
        model,
        "_native_forward_backward_batch",
        lambda *args, **kwargs: pytest.fail("small layout groups must stay vectorized"),
    )
    monkeypatch.setattr(
        model,
        "_forward_backward_python",
        lambda *args, **kwargs: pytest.fail("layouts must not loop over learners"),
    )

    gamma, log_likelihoods = model.forward_backward_batch(
        responses,
        skill_assignments,
    )

    assert gamma.shape == (*responses.shape, 2)
    _assert_matches_scalar_oracle(
        model, responses, skill_assignments, gamma, log_likelihoods
    )


def test_identical_person_layouts_use_shared_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, shared_skills = _batch()
    skill_assignments = np.tile(shared_skills, (len(responses), 1))
    model = _model()
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_native(
        response_values: np.ndarray,
        skill_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((response_values.shape, skill_values.shape))
        learned = np.full(response_values.shape, 0.6)
        return np.stack((1.0 - learned, learned), axis=2), np.full(
            response_values.shape[0], -3.0
        )

    monkeypatch.setattr(model, "_native_forward_backward_batch", fake_native)

    gamma, log_likelihoods = model.forward_backward_batch(
        responses,
        skill_assignments,
    )

    assert calls == [(responses.shape, shared_skills.shape)]
    assert_allclose(gamma[..., 1], 0.6)
    assert_allclose(log_likelihoods, -3.0)


def _repeated_layout_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle one layout shared by many learners among small layout groups."""
    _, shared_skills = _batch()
    min_learners = dynamic_module._BKT_NATIVE_LAYOUT_MIN_LEARNERS
    layouts = (
        [shared_skills] * (min_learners + 3)
        + [np.roll(shared_skills, 1)] * (min_learners - 1)
        + [np.roll(shared_skills, 2)]
    )
    rng = np.random.default_rng(17)
    skill_assignments = np.asarray(layouts, dtype=np.int32)[
        rng.permutation(len(layouts))
    ]
    responses = rng.integers(-1, 2, size=skill_assignments.shape).astype(np.int32)
    return responses, skill_assignments, shared_skills


@pytest.mark.parametrize("native_succeeds", [True, False])
def test_repeated_person_layouts_use_grouped_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    native_succeeds: bool,
) -> None:
    responses, skill_assignments, shared_skills = _repeated_layout_batch()
    model = _model()
    monkeypatch.setattr(model, "_can_use_native_inference", lambda: True)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_native(
        response_values: np.ndarray,
        skill_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        calls.append((response_values.copy(), skill_values.copy()))
        if not native_succeeds:
            return None
        learned = np.full(response_values.shape, 0.6)
        return np.stack((1.0 - learned, learned), axis=2), np.full(
            response_values.shape[0], -3.0
        )

    monkeypatch.setattr(model, "_native_forward_backward_batch", fake_native)
    monkeypatch.setattr(
        model,
        "_forward_backward_python",
        lambda *args, **kwargs: pytest.fail("layouts must not loop over learners"),
    )

    gamma, log_likelihoods = model.forward_backward_batch(
        responses,
        skill_assignments,
    )

    grouped = np.all(skill_assignments == shared_skills, axis=1)
    assert len(calls) == 1
    assert_array_equal(calls[0][0], responses[grouped])
    assert_array_equal(calls[0][1], shared_skills)
    vectorized = ~grouped
    if native_succeeds:
        assert_allclose(gamma[grouped, :, 1], 0.6)
        assert_allclose(log_likelihoods[grouped], -3.0)
    else:
        vectorized = np.ones(len(responses), dtype=bool)
    _assert_matches_scalar_oracle(
        model,
        responses[vectorized],
        skill_assignments[vectorized],
        gamma[vectorized],
        log_likelihoods[vectorized],
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled backend is unavailable")
def test_real_native_grouped_layouts_match_scalar_oracle() -> None:
    responses, skill_assignments, _ = _repeated_layout_batch()
    set_backend_preference("rust")
    model = _model()

    gamma, log_likelihoods = model.forward_backward_batch(
        responses,
        skill_assignments,
    )

    for person_idx, person_responses in enumerate(responses):
        expected_gamma, expected_ll = _scalar_forward_backward(
            model, person_responses, skill_assignments[person_idx]
        )
        assert_allclose(gamma[person_idx], expected_gamma, rtol=1e-10, atol=1e-12)
        assert log_likelihoods[person_idx] == pytest.approx(expected_ll, rel=1e-10)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("person_layouts", [True, False])
def test_vectorized_layout_smoother_matches_scalar_oracle(
    seed: int,
    person_layouts: bool,
) -> None:
    rng = np.random.default_rng([seed, int(person_layouts)])
    n_skills = int(rng.integers(1, 5))
    n_persons = int(rng.integers(1, 30))
    n_trials = int(rng.integers(1, 20))
    p_slip = rng.uniform(0.05, 0.25, n_skills)
    p_guess = rng.uniform(0.1, 0.3, n_skills)
    p_init = rng.uniform(0.1, 0.7, n_skills)
    if seed % 2:
        p_slip[0] = 0.0
        p_guess[0] = 0.0
        p_init[0] = 0.0
    model = BKTModel(
        n_skills=n_skills,
        allow_forgetting=True,
        p_init=p_init,
        p_learn=rng.uniform(0.0, 0.3, n_skills),
        p_forget=rng.uniform(0.0, 0.15, n_skills),
        p_slip=p_slip,
        p_guess=p_guess,
        use_rust=False,
    )
    responses = rng.integers(-1, 2, size=(n_persons, n_trials)).astype(np.int32)
    shape = (n_persons, n_trials) if person_layouts else (n_trials,)
    skills = rng.integers(0, n_skills, size=shape).astype(np.int32)
    if person_layouts and n_skills > 1:
        skills[0] = 0

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    assert gamma.shape == (n_persons, n_trials, 2)
    _assert_matches_scalar_oracle(model, responses, skills, gamma, log_likelihoods)


def test_person_layouts_preserve_zero_scaling_chains() -> None:
    model = BKTModel(
        n_skills=2,
        p_init=np.array([0.0, 0.4]),
        p_learn=np.array([0.0, 0.2]),
        p_slip=np.array([0.0, 0.1]),
        p_guess=np.array([0.0, 0.2]),
        use_rust=False,
    )
    responses = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [-1, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    skills = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    _assert_matches_scalar_oracle(model, responses, skills, gamma, log_likelihoods)
    # A correct answer on an unlearnable, guess-free skill has zero likelihood
    # and the chain stays at zero for later opportunities of that skill.
    assert_array_equal(gamma[0, [0, 2]], 0.0)
    assert_array_equal(gamma[1, [1, 2]], 0.0)
    assert log_likelihoods[0] < 2 * np.log(1e-300) + 1.0
    for person_idx in range(len(responses)):
        expected_gamma, expected_ll = model._forward_backward_python(
            responses[person_idx], skills[person_idx]
        )
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_ll)


def test_single_and_batch_methods_dispatch_to_native_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _batch()
    calls: list[str] = []

    def fake_forward(person_responses: np.ndarray, *args: Any) -> tuple[Any, Any]:
        calls.append("forward")
        return (
            np.tile([0.25, 0.75], (len(person_responses), 1)),
            np.full(len(person_responses), 0.5),
        )

    def fake_backward(person_responses: np.ndarray, *args: Any) -> np.ndarray:
        calls.append("backward")
        return np.ones((len(person_responses), 2))

    def fake_batch(batch_responses: np.ndarray, *args: Any) -> tuple[Any, Any]:
        calls.append("batch")
        return (
            np.full(batch_responses.shape, 0.75),
            np.full(batch_responses.shape[0], -4.0),
        )

    def fake_viterbi(person_responses: np.ndarray, *args: Any) -> np.ndarray:
        calls.append("viterbi")
        return np.ones(len(person_responses), dtype=np.int32)

    monkeypatch.setattr(dynamic_module, "should_use_rust", lambda use_rust: use_rust)
    monkeypatch.setattr(dynamic_module, "bkt_forward", fake_forward)
    monkeypatch.setattr(dynamic_module, "bkt_backward", fake_backward)
    monkeypatch.setattr(dynamic_module, "bkt_forward_backward_batch", fake_batch)
    monkeypatch.setattr(dynamic_module, "bkt_viterbi", fake_viterbi)
    model = _model()

    alpha, scaling = model.forward(responses[0], skills)
    beta = model.backward(responses[0], skills, scaling)
    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)
    path = model.viterbi(responses[0], skills)

    assert calls == ["forward", "backward", "batch", "viterbi"]
    assert_allclose(alpha[:, 1], 0.75)
    assert_allclose(beta, 1.0)
    assert_allclose(gamma[..., 1], 0.75)
    assert_allclose(log_likelihoods, -4.0)
    assert_array_equal(path, 1)


def test_native_wrapper_reuses_canonical_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _batch()
    model = _model()
    captured: tuple[np.ndarray, ...] | None = None

    class FakeNative:
        def bkt_forward_backward_batch(self, *args: np.ndarray) -> tuple[Any, Any]:
            nonlocal captured
            captured = args
            return np.full(responses.shape, 0.5), np.zeros(responses.shape[0])

    monkeypatch.setattr(native_dynamic_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(native_dynamic_module, "mirt_rs", FakeNative())

    native_dynamic_module.bkt_forward_backward_batch(
        responses,
        skills,
        model.p_init,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )

    assert captured is not None
    assert captured[0] is responses
    assert captured[1] is skills
    assert captured[2] is model.p_init
    assert captured[3] is model.p_learn
    assert captured[4] is model.p_forget
    assert captured[5] is model.p_slip
    assert captured[6] is model.p_guess

    response_view = responses[:, ::-1]
    skill_view = skills[::-1]
    p_init_view = np.column_stack((model.p_init, model.p_init))[:, 0]
    native_dynamic_module.bkt_forward_backward_batch(
        response_view,
        skill_view,
        p_init_view,
        model.p_learn,
        model.p_forget,
        model.p_slip,
        model.p_guess,
    )

    assert captured is not None
    assert captured[0].flags.c_contiguous
    assert captured[1].flags.c_contiguous
    assert captured[2].flags.c_contiguous
    assert_array_equal(captured[0], response_view)
    assert_array_equal(captured[1], skill_view)
    assert_allclose(captured[2], p_init_view)


@pytest.mark.parametrize(
    ("backend", "use_rust"),
    [("numpy", True), ("auto", False)],
)
def test_backend_controls_disable_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    use_rust: bool,
) -> None:
    responses, skills = _batch()
    set_backend_preference(backend)
    monkeypatch.setattr(
        dynamic_module,
        "bkt_forward_backward_batch",
        lambda *args, **kwargs: pytest.fail("native inference should be disabled"),
    )

    gamma, log_likelihoods = _model(use_rust=use_rust).forward_backward_batch(
        responses, skills
    )

    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(log_likelihoods))


def test_malformed_native_output_falls_back_without_state_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _batch()
    monkeypatch.setattr(dynamic_module, "should_use_rust", lambda use_rust: use_rust)
    monkeypatch.setattr(
        dynamic_module,
        "bkt_forward_backward_batch",
        lambda *args, **kwargs: (np.full((1, 1), np.nan), np.array([np.nan])),
    )
    accelerated = _model()
    fallback = _model(use_rust=False)

    gamma, log_likelihoods = accelerated.forward_backward_batch(responses, skills)
    expected_gamma, expected_ll = fallback.forward_backward_batch(responses, skills)

    assert_allclose(gamma, expected_gamma)
    assert_allclose(log_likelihoods, expected_ll)
    assert_allclose(accelerated.p_init, fallback.p_init)


def test_degenerate_emissions_use_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, skills = _batch()
    monkeypatch.setattr(dynamic_module, "should_use_rust", lambda use_rust: use_rust)
    monkeypatch.setattr(
        dynamic_module,
        "bkt_forward_backward_batch",
        lambda *args, **kwargs: pytest.fail("degenerate model must use fallback"),
    )
    model = BKTModel(
        n_skills=3,
        p_slip=np.zeros(3),
        p_guess=np.zeros(3),
    )

    gamma, log_likelihoods = model.forward_backward_batch(responses, skills)

    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(log_likelihoods))


def test_use_rust_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="use_rust must be a boolean"):
        BKTModel(n_skills=1, use_rust=1)
    with pytest.raises(TypeError, match="use_rust must be a boolean"):
        BKTGibbsSampler(use_rust=1)


def test_sampler_propagates_native_preference() -> None:
    responses, skills = _batch()
    fallback = BKTGibbsSampler(
        n_iter=2,
        burnin=1,
        thin=1,
        seed=9,
        use_rust=False,
    ).fit(responses[:2], skills, n_skills=3, allow_forgetting=True)
    accelerated = BKTGibbsSampler(
        n_iter=2,
        burnin=1,
        thin=1,
        seed=9,
        use_rust=True,
    ).fit(responses[:2], skills, n_skills=3, allow_forgetting=True)

    assert fallback.model.use_rust is False
    assert accelerated.model.use_rust is True
    assert_allclose(accelerated.learning_curves, fallback.learning_curves)
    assert_allclose(accelerated.skill_mastery, fallback.skill_mastery)
    assert accelerated.log_likelihood == pytest.approx(fallback.log_likelihood)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="compiled backend is unavailable")
def test_real_native_batch_matches_numpy_with_missing_data() -> None:
    responses, skills = _batch()
    set_backend_preference("numpy")
    numpy_model = _model()
    expected_gamma, expected_ll = numpy_model.forward_backward_batch(responses, skills)
    expected_alpha, expected_scaling = numpy_model.forward(responses[0], skills)
    expected_path = numpy_model.viterbi(responses[0], skills)

    set_backend_preference("rust")
    native_model = _model()
    gamma, log_likelihoods = native_model.forward_backward_batch(responses, skills)
    alpha, scaling = native_model.forward(responses[0], skills)
    path = native_model.viterbi(responses[0], skills)

    assert_allclose(gamma, expected_gamma, rtol=1e-12, atol=1e-12)
    assert_allclose(log_likelihoods, expected_ll, rtol=1e-12, atol=1e-12)
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)
    assert_allclose(scaling, expected_scaling, rtol=1e-12, atol=1e-12)
    assert_array_equal(path, expected_path)
