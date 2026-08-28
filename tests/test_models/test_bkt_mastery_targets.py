"""Contracts for Bayesian knowledge-tracing mastery targets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.models.dynamic import (
    BKTBatchMasteryTargetResult,
    BKTMasteryTargetResult,
    BKTModel,
)


@pytest.fixture
def model() -> BKTModel:
    return BKTModel(
        n_skills=4,
        allow_forgetting=True,
        p_init=np.array([0.2, 0.55, 0.8, 0.35]),
        p_learn=np.array([0.2, 0.3, 0.1, 0.0]),
        p_forget=np.array([0.0, 0.1, 0.2, 0.0]),
        p_slip=np.array([0.08, 0.15, 0.22, 0.05]),
        p_guess=np.array([0.12, 0.25, 0.35, 0.4]),
        use_rust=False,
    )


def test_models_module_exports_mastery_target_results() -> None:
    from mirt.models import (
        BKTBatchMasteryTargetResult as PublicBatchMasteryTargetResult,
    )
    from mirt.models import BKTMasteryTargetResult as PublicMasteryTargetResult

    assert PublicMasteryTargetResult is BKTMasteryTargetResult
    assert PublicBatchMasteryTargetResult is BKTBatchMasteryTargetResult


def test_opportunities_match_closed_form_transition(model: BKTModel) -> None:
    priors = np.array([0.1, 0.2, 0.9, 0.4])
    targets = np.array([0.6, 0.7, 0.8, 0.5])
    original_priors = priors.copy()
    original_targets = targets.copy()

    result = model.opportunities_to_mastery(priors, targets)

    assert isinstance(result, BKTMasteryTargetResult)
    assert result.n_skills == model.n_skills
    assert_array_equal(result.opportunities, np.array([4.0, 5.0, 0.0, np.inf]))
    assert_array_equal(result.reachable, np.array([True, True, True, False]))
    assert not result.all_reachable
    assert_array_equal(result.target_mastery, targets)
    assert_array_equal(priors, original_priors)
    assert_array_equal(targets, original_targets)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "opportunities", np.empty(0))


def test_default_target_is_broadcast(model: BKTModel) -> None:
    result = model.opportunities_to_mastery(np.array([0.1, 0.2, 0.9, 0.4]))

    assert_allclose(result.target_mastery, 0.95)
    assert_array_equal(result.reachable, np.array([True, False, False, False]))


def test_boundary_and_alternating_transitions() -> None:
    model = BKTModel(
        n_skills=7,
        allow_forgetting=True,
        p_learn=np.array([0.0, 0.2, 0.3, 0.8, 0.8, 1.0, 0.2]),
        p_forget=np.array([0.0, 0.2, 0.7, 0.6, 0.6, 1.0, 0.0]),
        use_rust=False,
    )
    priors = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1])
    targets = np.array([0.5, 0.5, 0.3, 0.7, 0.8, 0.8, 1.0])

    result = model.opportunities_to_mastery(priors, targets)

    assert_array_equal(
        result.opportunities,
        np.array([np.inf, np.inf, 1.0, 1.0, np.inf, 1.0, np.inf]),
    )


def test_batch_accepts_shared_and_person_specific_targets(model: BKTModel) -> None:
    priors = np.array(
        [
            [0.1, 0.2, 0.9, 0.4],
            [0.4, 0.5, 0.7, 0.6],
            [0.8, 0.7, 0.5, 0.3],
        ]
    )
    shared_targets = np.array([0.6, 0.7, 0.8, 0.5])
    person_targets = np.array(
        [
            [0.6, 0.7, 0.8, 0.5],
            [0.7, 0.6, 0.75, 0.6],
            [0.85, 0.72, 0.5, 0.4],
        ]
    )

    shared = model.opportunities_to_mastery_batch(priors, shared_targets)
    person_specific = model.opportunities_to_mastery_batch(
        priors,
        person_targets,
    )

    assert isinstance(shared, BKTBatchMasteryTargetResult)
    assert shared.n_persons == len(priors)
    assert shared.n_skills == model.n_skills
    assert_array_equal(
        shared.target_mastery, np.broadcast_to(shared_targets, priors.shape)
    )
    assert_array_equal(shared.reachable_counts, np.sum(shared.reachable, axis=1))
    assert not shared.all_reachable
    assert_array_equal(person_specific.target_mastery, person_targets)
    for person_idx in range(len(priors)):
        expected_shared = model.opportunities_to_mastery(
            priors[person_idx],
            shared_targets,
        )
        expected_person = model.opportunities_to_mastery(
            priors[person_idx],
            person_targets[person_idx],
        )
        assert_array_equal(
            shared.opportunities[person_idx],
            expected_shared.opportunities,
        )
        assert_array_equal(
            person_specific.opportunities[person_idx],
            expected_person.opportunities,
        )


def test_closed_form_matches_brute_force_crossings() -> None:
    rng = np.random.default_rng(42)
    n_skills = 12
    learn = rng.uniform(0.02, 0.4, n_skills)
    forget = rng.uniform(0.0, 0.25, n_skills)
    model = BKTModel(
        n_skills=n_skills,
        allow_forgetting=True,
        p_learn=learn,
        p_forget=forget,
        use_rust=False,
    )
    priors = rng.random((80, n_skills))
    targets = rng.random((80, n_skills))

    result = model.opportunities_to_mastery_batch(priors, targets)

    expected = np.full(priors.shape, np.inf)
    current = priors.copy()
    unresolved = current < targets
    expected[~unresolved] = 0.0
    for opportunity in range(1, 5_001):
        current = current * (1.0 - forget) + (1.0 - current) * learn
        newly_reached = unresolved & (current >= targets)
        expected[newly_reached] = opportunity
        unresolved[newly_reached] = False
        if not np.any(unresolved):
            break

    assert_array_equal(result.opportunities, expected)


def test_finite_counts_are_minimal_forecast_crossings(model: BKTModel) -> None:
    priors = np.array([0.1, 0.2, 0.9, 0.4])
    targets = np.array([0.6, 0.7, 0.8, 0.5])

    result = model.opportunities_to_mastery(priors, targets)
    forecast = model.forecast_from_priors(priors, 7)

    for skill_idx, count_value in enumerate(result.opportunities):
        if not np.isfinite(count_value):
            continue
        count = int(count_value)
        if count == 0:
            assert priors[skill_idx] >= targets[skill_idx]
            continue
        assert forecast.mastery_probabilities[count, skill_idx] >= targets[skill_idx]
        assert forecast.mastery_probabilities[count - 1, skill_idx] < targets[skill_idx]


def test_history_targets_match_retained_priors(model: BKTModel) -> None:
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

    result = model.opportunities_to_mastery_batch(
        diagnostics.next_mastery_priors,
        target_mastery=0.8,
    )

    for person_idx in range(len(responses)):
        expected = model.opportunities_to_mastery(
            model.next_mastery_priors(responses[person_idx], skills),
            target_mastery=0.8,
        )
        assert_array_equal(result.opportunities[person_idx], expected.opportunities)


def test_target_calculation_does_not_allocate_forecast_horizon(
    model: BKTModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model,
        "_forecast_from_priors_batch",
        lambda *args, **kwargs: pytest.fail("target solving must be closed form"),
    )

    result = model.opportunities_to_mastery_batch(
        np.full((1_000, model.n_skills), 0.1),
        target_mastery=0.8,
    )

    assert result.opportunities.shape == (1_000, model.n_skills)


@pytest.mark.parametrize(
    "target_mastery",
    [
        True,
        0.5 + 0.1j,
        np.nan,
        -0.1,
        1.1,
        np.array([]),
        np.array([0.2, 0.4, 0.6]),
        np.ones((1, 4)),
    ],
)
def test_scalar_validates_targets(
    model: BKTModel,
    target_mastery: object,
) -> None:
    with pytest.raises(ValueError, match="target_mastery"):
        model.opportunities_to_mastery(
            np.full(model.n_skills, 0.2),
            target_mastery=target_mastery,
        )


@pytest.mark.parametrize(
    "target_mastery",
    [
        np.array([]),
        np.array([0.2, 0.4, 0.6]),
        np.ones((2, 3)),
        np.ones((1, 2, 4)),
        np.array([[0.2, 0.4, 0.6, 0.8], [0.3, 0.2, np.inf, 0.7]]),
    ],
)
def test_batch_validates_targets(
    model: BKTModel,
    target_mastery: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="target_mastery"):
        model.opportunities_to_mastery_batch(
            np.full((2, model.n_skills), 0.2),
            target_mastery=target_mastery,
        )


@pytest.mark.parametrize(
    ("batch", "prior_mastery"),
    [
        (False, np.array([0.2, 0.4, 0.6])),
        (False, np.array([True, False, True, False])),
        (True, np.array([0.2, 0.4, 0.6, 0.8])),
        (True, np.empty((0, 4))),
        (True, np.ones((2, 3))),
    ],
)
def test_target_methods_validate_priors(
    model: BKTModel,
    batch: bool,
    prior_mastery: np.ndarray,
) -> None:
    method = (
        model.opportunities_to_mastery_batch
        if batch
        else model.opportunities_to_mastery
    )
    with pytest.raises(ValueError, match="prior_mastery"):
        method(prior_mastery)
