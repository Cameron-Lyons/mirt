"""Shared two-group calibration, linking and bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt
from mirt.diagnostics._utils import (
    fit_group_models,
    fit_linked_group_models,
    link_focal_to_reference,
    resolve_anchor_items,
    summarize_bootstrap,
    validate_two_group_inputs,
)
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel

SCALE, SHIFT = 1.3, -0.7
DISCRIMINATION = np.array([0.8, 1.2, 1.5, 2.0, 1.1])
LOCATION = np.array([-1.2, -0.4, 0.1, 0.6, 1.3])


def _calibration_pair(model_type: type) -> tuple[Any, Any]:
    """A reference calibration and the focal calibration of the same items.

    Focal abilities follow ``N(SHIFT, SCALE**2)``; calibrating the focal group
    alone gives slopes ``a * SCALE`` and locations ``(b - SHIFT) / SCALE``.
    """
    n_items = DISCRIMINATION.size
    if model_type in (GradedResponseModel, GeneralizedPartialCredit):
        locations = LOCATION[:, None] + np.array([-0.5, 0.5])
        name = "thresholds" if model_type is GradedResponseModel else "steps"
        reference = model_type(n_items=n_items, n_categories=3)
        focal = model_type(n_items=n_items, n_categories=3)
        reference.set_parameters(discrimination=DISCRIMINATION, **{name: locations})
        focal.set_parameters(
            discrimination=DISCRIMINATION * SCALE,
            **{name: (locations - SHIFT) / SCALE},
        )
        return reference, focal
    reference = model_type(n_items=n_items)
    focal = model_type(n_items=n_items)
    if model_type is OneParameterLogistic:
        reference.set_parameters(difficulty=LOCATION)
        focal.set_parameters(difficulty=LOCATION - SHIFT)
        return reference, focal
    reference.set_parameters(discrimination=DISCRIMINATION, difficulty=LOCATION)
    focal.set_parameters(
        discrimination=DISCRIMINATION * SCALE, difficulty=(LOCATION - SHIFT) / SCALE
    )
    if model_type is ThreeParameterLogistic:
        guessing = np.full(n_items, 0.15)
        reference.set_parameters(guessing=guessing)
        focal.set_parameters(guessing=guessing)
    return reference, focal


@pytest.mark.parametrize(
    "model_type",
    [
        TwoParameterLogistic,
        ThreeParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
    ],
)
def test_linking_recovers_the_reference_scale(model_type: type) -> None:
    reference, focal = _calibration_pair(model_type)

    linked, A, B = link_focal_to_reference(reference, focal, range(5))

    assert A == pytest.approx(SCALE, rel=1e-4)
    assert B == pytest.approx(SHIFT, abs=1e-4)
    for name, values in reference.parameters.items():
        assert_allclose(linked.parameters[name], values, atol=1e-4)
    assert_allclose(focal.parameters["discrimination"], DISCRIMINATION * SCALE)


def test_one_parameter_linking_shifts_locations_only() -> None:
    reference, focal = _calibration_pair(OneParameterLogistic)

    linked, A, B = link_focal_to_reference(reference, focal, [0, 1, 2, 3, 4])

    assert A == 1.0
    assert B == pytest.approx(SHIFT)
    assert_allclose(linked.parameters["difficulty"], LOCATION)
    assert_allclose(linked.parameters["discrimination"], 1.0)


def test_fit_linked_group_models_links_over_requested_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, focal = _calibration_pair(TwoParameterLogistic)
    biased = focal.copy()
    difficulty = biased.parameters["difficulty"].copy()
    difficulty[0] += 1.0
    biased.set_parameters(difficulty=difficulty)
    options: list[dict[str, Any]] = []

    def fake_fit(ref_data: Any, focal_data: Any, model: str = "2PL", **kwargs: Any):
        options.append(kwargs)
        return SimpleNamespace(model=reference), SimpleNamespace(model=biased)

    monkeypatch.setattr("mirt.diagnostics._utils.fit_group_models", fake_fit)
    data = np.zeros((4, 5), dtype=np.int64)

    anchored = fit_linked_group_models(
        data, data, anchor_items=[4, 1, 2, 3], compute_standard_errors=True
    )
    all_items = fit_linked_group_models(data, data)

    assert anchored.anchor_items == [1, 2, 3, 4]
    assert anchored.A == pytest.approx(SCALE, rel=1e-4)
    assert anchored.B == pytest.approx(SHIFT, abs=1e-4)
    assert anchored.focal_on_reference.parameters["difficulty"][0] == pytest.approx(
        LOCATION[0] + SCALE * 1.0, abs=1e-3
    )
    assert all_items.anchor_items == [0, 1, 2, 3, 4]
    assert abs(all_items.B - SHIFT) > 1e-3
    assert options[0]["compute_standard_errors"] is True


def test_group_fits_share_pooled_polytomous_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fit(data: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(model=None)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    reference = np.array([[0, 2, 1], [1, 1, 0]])
    focal = np.array([[0, 1, 3], [1, 0, -1]])

    fit_group_models(reference, focal, model="GRM")
    fit_group_models(reference, focal, model="GRM", n_categories=5)
    fit_group_models(reference, focal, model="2PL")

    assert [call["n_categories"] for call in calls[:2]] == [[2, 3, 4], [2, 3, 4]]
    assert [call["n_categories"] for call in calls[2:4]] == [5, 5]
    assert "n_categories" not in calls[4]


def test_pooled_categories_treat_nan_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: NaN-coded missing responses produced NaN category counts."""
    calls: list[dict[str, Any]] = []

    def fake_fit(data: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(model=None)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    reference = np.array([[0.0, 2.0], [np.nan, 1.0]])
    focal = np.array([[1.0, np.nan], [2.0, 0.0]])

    fit_group_models(reference, focal, model="GPCM")

    assert [call["n_categories"] for call in calls] == [[3, 3], [3, 3]]


@pytest.mark.parametrize(
    ("anchors", "message"),
    [
        ([1], "at least 2"),
        ([1, 1], "duplicate"),
        ([0, 5], r"\[0, 5\)"),
        ([0, True], "integer"),
        ("01", "sequence"),
    ],
)
def test_anchor_items_are_validated(anchors: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_anchor_items(anchors, 5)


def test_anchor_items_are_sorted_and_none_passes_through() -> None:
    assert resolve_anchor_items(np.array([3, 0, 2]), 5) == [0, 2, 3]
    assert resolve_anchor_items(None, 5) is None
    assert resolve_anchor_items([4], 5, minimum=1) == [4]


def test_bootstrap_summary_drops_failed_replicates() -> None:
    summary = summarize_bootstrap(
        [0.1, np.nan, 0.3, 0.2, np.inf],
        observed=0.4,
        n_requested=5,
        confidence_level=0.9,
    )

    assert summary.n_successful == 3
    assert summary.n_failed == 2
    assert summary.standard_error == pytest.approx(0.1)
    assert summary.p_value == pytest.approx(2 * 0.0000316712418, rel=1e-4)
    assert summary.confidence_interval == pytest.approx((0.11, 0.29))


def test_bootstrap_summary_needs_two_replicates() -> None:
    summary = summarize_bootstrap(
        [0.1], observed=0.2, n_requested=3, confidence_level=0.95
    )

    assert np.isnan(summary.standard_error) and np.isnan(summary.p_value)
    assert (summary.n_successful, summary.n_failed) == (1, 2)


def test_two_group_validation_rejects_missing_float_labels_in_objects() -> None:
    with pytest.raises(ValueError, match="missing labels"):
        validate_two_group_inputs(
            np.zeros((3, 2)), np.array(["a", "b", np.nan], dtype=object), (-1, 1)
        )
