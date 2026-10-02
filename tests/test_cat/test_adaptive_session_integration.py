"""Cached adaptive EAP must retain transactional answers and exposure sessions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, roots_hermite

from mirt.cat import CATEngine, MCATEngine, SympsonHetter
from mirt.models import MultidimensionalModel, TwoParameterLogistic


def _engine(n_factors):
    if n_factors == 1:
        model = TwoParameterLogistic(n_items=4)
        model.set_parameters(
            discrimination=np.array([0.8, 1.2, 1.5, 1.1]),
            difficulty=np.array([-0.6, 0.2, 0.9, -0.4]),
        )
        engine_type = CATEngine
    else:
        model = MultidimensionalModel(n_items=4, n_factors=2)
        model.set_parameters(
            slopes=np.array([[0.8, 1.1], [1.2, 0.7], [1.5, 1.3], [0.9, 1.4]]),
            intercepts=np.array([-0.6, 0.2, 0.9, -0.4]),
        )
        engine_type = MCATEngine
    model._is_fitted = True
    exposure = SympsonHetter(np.ones(4), seed=7)
    return (
        engine_type(
            model, exposure_control=exposure, n_quadpts=9, min_items=4, max_items=4
        ),
        exposure,
    )


def _reference(model, items, responses):
    nodes, weights = roots_hermite(9)
    points = np.column_stack(
        [
            values.ravel()
            for values in np.meshgrid(
                *([nodes * np.sqrt(2)] * model.n_factors), indexing="ij"
            )
        ]
    )
    mass = np.prod(
        np.stack(
            np.meshgrid(*([weights / np.sqrt(np.pi)] * model.n_factors), indexing="ij")
        ),
        axis=0,
    ).ravel()
    for item, response in zip(items, responses, strict=True):
        if model.n_factors == 1:
            logits = model.discrimination[item] * (
                points[:, 0] - model.difficulty[item]
            )
        else:
            logits = points @ model.slopes[item] + model.intercepts[item]
        probability = expit(logits)
        mass *= probability if response else 1 - probability
    mass /= mass.sum()
    mean = mass @ points
    residuals = points - mean
    covariance = np.einsum("q,qi,qj->ij", mass, residuals, residuals)
    return mean, covariance


def _assert_moments(engine, state):
    mean, covariance = _reference(
        engine.model, state.items_administered, state.responses
    )
    assert_allclose(np.atleast_1d(state.theta), mean, atol=2e-14)
    assert_allclose(
        np.atleast_1d(state.standard_error) ** 2, np.diag(covariance), atol=2e-14
    )
    if hasattr(state, "covariance"):
        assert_allclose(state.covariance, covariance, atol=2e-14)


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_cached_eap_handles_rejected_answers_and_resets_without_stale_evidence(
    n_factors,
):
    engine, exposure = _engine(n_factors)
    first = engine.administer_item(0)
    _assert_moments(engine, first)
    grid = engine._eap_quadrature
    before = exposure.exposure_report(n_items=4)

    for response in [np.array(0.25), 2, np.array(np.nan), "1"]:
        with pytest.raises(ValueError, match="response"):
            engine.administer_item(response)
        state = engine.get_current_state()
        assert state.items_administered == first.items_administered
        assert state.responses == [0]
        assert state.next_item == first.next_item
        assert engine._eap_quadrature is grid
        _assert_moments(engine, state)
        report = exposure.exposure_report(n_items=4)
        assert report.n_examinees == before.n_examinees == 1
        for field in ("selection_counts", "opportunity_counts", "eligibility_counts"):
            assert_array_equal(getattr(report, field), getattr(before, field))

    corrected = engine.administer_item(np.array(1, dtype=np.uint64))
    assert corrected.items_administered[-1] == first.next_item
    assert corrected.responses == [0, 1]
    assert type(corrected.responses[-1]) is int
    _assert_moments(engine, corrected)
    assert engine._eap_quadrature is grid

    engine.reset()
    engine.reset()
    assert exposure.n_examinees == 2
    replacement = engine.administer_item(np.array(1.0))
    assert replacement.responses == [1]
    assert replacement.n_items == 1
    assert engine._eap_quadrature is grid
    _assert_moments(engine, replacement)
    assert exposure.exposure_report(n_items=4).selection_counts.sum() == 3
