"""Diagnostics match DataFrame columns to the model's item names."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.itemfit import compute_itemfit
from mirt.diagnostics.ld import compute_ld_chi2, compute_q3
from mirt.diagnostics.modelfit import compute_m2
from mirt.diagnostics.residuals import compute_residuals
from mirt.models.dichotomous import TwoParameterLogistic


def _named_fit() -> tuple[TwoParameterLogistic, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    names = [f"Q{index}" for index in range(6)]
    model = TwoParameterLogistic(6, item_names=names).set_parameters(
        discrimination=np.linspace(0.7, 1.8, 6), difficulty=np.linspace(-1, 1, 6)
    )
    model._is_fitted = True
    theta = rng.standard_normal((200, 1))
    responses = model.simulate(theta, seed=3)
    return model, responses, theta


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_diagnostics_read_reordered_frames_by_item_name(library: str) -> None:
    frames = pytest.importorskip(library)
    model, responses, theta = _named_fit()
    columns = {name: responses[:, j] for j, name in enumerate(model.item_names)}
    reordered = dict(reversed(list(columns.items())))
    reordered["respondent"] = np.arange(len(responses))
    frame = frames.DataFrame(reordered)

    expected_fit = compute_itemfit(model, responses, theta=theta)
    actual_fit = compute_itemfit(model, frame, theta=theta)
    for name in ("infit", "outfit"):
        assert_allclose(actual_fit[name], expected_fit[name])
    assert_allclose(
        compute_residuals(model, frame, theta=theta),
        compute_residuals(model, responses, theta=theta),
    )
    assert_allclose(
        compute_q3(model, frame, theta=theta), compute_q3(model, responses, theta)
    )
    actual_ld, expected_ld = (
        compute_ld_chi2(model, data)[0] for data in (frame, responses)
    )
    assert_allclose(actual_ld, expected_ld, equal_nan=True)
    assert compute_m2(model, frame)["M2"] == pytest.approx(
        compute_m2(model, responses)["M2"]
    )
