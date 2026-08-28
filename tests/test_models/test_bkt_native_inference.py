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


def test_shared_numpy_filter_and_smoother_match_individual_fallbacks() -> None:
    responses, skills = _batch()
    model = _model(use_rust=False)
    skill_trials = model._skill_trials(skills)

    alpha, scaling = model._forward_batch_shared_python(
        responses,
        skills,
        skill_trials,
    )
    beta = model._backward_batch_shared_python(
        responses,
        skills,
        scaling,
        skill_trials,
    )
    gamma, log_likelihoods = model._forward_backward_batch_shared_python(
        responses,
        skills,
    )

    for person_idx, person_responses in enumerate(responses):
        expected_alpha, expected_scaling = model._forward_python(
            person_responses,
            skills,
        )
        expected_beta = model._backward_python(
            person_responses,
            skills,
            expected_scaling,
        )
        expected_gamma, expected_log_likelihood = model._forward_backward_python(
            person_responses,
            skills,
        )
        assert_allclose(alpha[person_idx], expected_alpha)
        assert_allclose(scaling[person_idx], expected_scaling)
        assert_allclose(beta[person_idx], expected_beta)
        assert_allclose(gamma[person_idx], expected_gamma)
        assert log_likelihoods[person_idx] == pytest.approx(expected_log_likelihood)


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
