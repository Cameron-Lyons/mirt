"""Rejected CAT answers must be recoverable without corrupting a session."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.cat import CATEngine, MCATEngine, SympsonHetter
from mirt.models import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    MultidimensionalModel,
    TwoParameterLogistic,
)


def _model(n_factors: int, *, polytomous: bool = False):
    if polytomous and n_factors == 2:
        # MCAT needs exact polytomous Fisher matrices, which GPCM defines.
        model = GeneralizedPartialCredit(n_items=3, n_factors=2, n_categories=[2, 3, 4])
        model.set_parameters(
            discrimination=np.array([[0.8, 1.1], [1.2, 0.7], [1.5, 1.3]]),
            steps=np.array([[0.1, 0.0, 0.0], [-0.8, 0.7, 0.0], [-1.0, 0.0, 1.0]]),
        )
    elif polytomous:
        model = GradedResponseModel(n_items=3, n_categories=[2, 3, 4])
        model.set_parameters(
            discrimination=np.array([0.8, 1.2, 1.5]),
            thresholds=np.array([[0.1, 0.0, 0.0], [-0.8, 0.7, 0.0], [-1.0, 0.0, 1.0]]),
        )
    elif n_factors == 1:
        model = TwoParameterLogistic(n_items=3)
        model.set_parameters(
            discrimination=np.array([0.8, 1.2, 1.5]),
            difficulty=np.array([-0.6, 0.2, 0.9]),
        )
    else:
        model = MultidimensionalModel(n_items=3, n_factors=2)
        model.set_parameters(
            slopes=np.array([[0.8, 1.1], [1.2, 0.7], [1.5, 1.3]]),
            intercepts=np.array([-0.6, 0.2, 0.9]),
        )
    model._is_fitted = True
    return model


def _engine(n_factors: int, *, polytomous: bool = False):
    model = _model(n_factors, polytomous=polytomous)
    exposure = SympsonHetter(np.ones(model.n_items), seed=17)
    engine_class = CATEngine if n_factors == 1 else MCATEngine
    engine = engine_class(
        model,
        min_items=3,
        max_items=3,
        n_quadpts=9,
        exposure_control=exposure,
        seed=11,
    )
    return engine, exposure


def _assert_same_state(left, right):
    assert_allclose(left.theta, right.theta, rtol=0.0, atol=0.0)
    assert_allclose(left.standard_error, right.standard_error, rtol=0.0, atol=0.0)
    if hasattr(left, "covariance"):
        assert_allclose(left.covariance, right.covariance, rtol=0.0, atol=0.0)
    assert left.items_administered == right.items_administered
    assert left.responses == right.responses
    assert left.n_items == right.n_items
    assert left.next_item == right.next_item
    assert left.is_complete == right.is_complete


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
@pytest.mark.parametrize(
    "response", [-1, -9, 2, 0.5, np.nan, np.inf, "1", [1], np.array([1]), 1j]
)
def test_invalid_dichotomous_answer_preserves_session_and_exposure(n_factors, response):
    engine, exposure = _engine(n_factors)
    engine.administer_item(0)
    before = engine.get_current_state()
    report = exposure.exposure_report(n_items=3)

    with pytest.raises(ValueError, match="response"):
        engine.administer_item(response)

    _assert_same_state(engine.get_current_state(), before)
    after_report = exposure.exposure_report(n_items=3)
    for field in ("selection_counts", "opportunity_counts", "eligibility_counts"):
        assert_array_equal(getattr(after_report, field), getattr(report, field))

    # Correcting the answer must use the same disclosed item, exactly once.
    after = engine.administer_item(np.int64(1))
    assert after.items_administered == before.items_administered + [before.next_item]
    assert after.responses == [0, 1]
    assert exposure.exposure_report(n_items=3).selection_counts.sum() == 2


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_rejecting_an_answer_does_not_change_final_estimate_or_adaptive_path(n_factors):
    clean, clean_exposure = _engine(n_factors)
    corrected, corrected_exposure = _engine(n_factors)

    for response in [1, 0, 1]:
        _assert_same_state(clean.get_current_state(), corrected.get_current_state())
        with pytest.raises(ValueError):
            corrected.administer_item(0.75)
        _assert_same_state(
            clean.administer_item(response), corrected.administer_item(response)
        )

    left, right = clean.get_result(), corrected.get_result()
    assert_allclose(left.theta_history, right.theta_history, rtol=0.0, atol=0.0)
    assert_allclose(left.se_history, right.se_history, rtol=0.0, atol=0.0)
    assert left.stopping_reason == right.stopping_reason
    assert_array_equal(
        clean_exposure.exposure_report(n_items=3).selection_counts,
        corrected_exposure.exposure_report(n_items=3).selection_counts,
    )


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_category_limit_uses_the_disclosed_polytomous_item(n_factors):
    engine, exposure = _engine(n_factors, polytomous=True)

    while not engine.get_current_state().is_complete:
        before = engine.get_current_state()
        n_categories = engine.model.n_categories[before.next_item]
        with pytest.raises(ValueError, match=f"item {before.next_item}"):
            engine.administer_item(n_categories)
        _assert_same_state(engine.get_current_state(), before)
        after = engine.administer_item(np.float64(n_categories - 1))
        assert after.responses[-1] == n_categories - 1
        assert type(after.responses[-1]) is int

    result = engine.get_result()
    assert result.n_items_administered == 3
    assert np.all(np.isfinite(result.theta))
    assert np.all(np.isfinite(result.standard_error))
    assert exposure.exposure_report(n_items=3).selection_counts.sum() == 3


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
@pytest.mark.parametrize("response", [0, 1, np.int64(1), 1.0, True, np.bool_(False)])
def test_accepts_and_normalizes_integer_valued_numeric_scalars(n_factors, response):
    engine, _ = _engine(n_factors)
    state = engine.administer_item(response)
    assert state.responses == [int(response)]
    assert type(state.responses[0]) is int
