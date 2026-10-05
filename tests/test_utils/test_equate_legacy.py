"""Regression tests for the deprecated equate() and transform_theta helpers."""

import numpy as np
import pytest

from mirt.equating.linking import link
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.utils.calibration import EquatingResult, equate, transform_theta

METHODS = ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]


def _forms(noise: float = 0.0, seed: int = 3):
    """Old and new calibrations with theta_old = 1.3 * theta_new + 0.4."""
    rng = np.random.default_rng(seed)
    a_old = np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 0.8, 1.2, 1.6, 1.0])
    b_old = np.array([-1.6, -1.1, -0.6, -0.2, 0.1, 0.5, 0.9, 1.2, 1.6, 0.0])
    old = TwoParameterLogistic(10)
    new = TwoParameterLogistic(10)
    old.set_parameters(discrimination=a_old, difficulty=b_old)
    new.set_parameters(
        discrimination=1.3 * a_old * np.exp(rng.normal(0.0, noise, 10)),
        difficulty=(b_old - 0.4) / 1.3 + rng.normal(0.0, noise, 10),
    )
    return old, new


@pytest.mark.parametrize("method", METHODS)
def test_exact_anchors_recover_legacy_constants(method):
    old, new = _forms()

    with pytest.warns(DeprecationWarning, match="mirt.equating.link"):
        result = equate(old, new, list(range(10)), list(range(10)), method=method)

    # Legacy convention: theta_new = A * theta_old + B.
    assert result.A == pytest.approx(1.0 / 1.3, abs=1e-7)
    assert result.B == pytest.approx(-0.4 / 1.3, abs=1e-7)
    assert result.rmse == pytest.approx(0.0, abs=1e-6)
    assert result.method == method
    assert result.anchor_items == list(range(10))


@pytest.mark.parametrize("method", METHODS)
def test_transform_theta_places_new_scores_on_old_scale(method):
    old, new = _forms()
    theta_new = np.array([-2.0, -0.5, 0.0, 1.0, 2.5])

    with pytest.warns(DeprecationWarning):
        result = equate(old, new, list(range(10)), list(range(10)), method=method)

    np.testing.assert_allclose(
        transform_theta(theta_new, result), 1.3 * theta_new + 0.4, atol=1e-6
    )


@pytest.mark.parametrize("method", METHODS)
def test_matches_link_up_to_the_documented_inversion(method):
    old, new = _forms(noise=0.15)
    anchors = list(range(10))

    with pytest.warns(DeprecationWarning):
        result = equate(old, new, anchors, anchors, method=method)

    linked = link(old, new, anchors, anchors, method=method)
    assert result.A == pytest.approx(1.0 / linked.constants.A, rel=1e-12)
    assert result.B == pytest.approx(
        -linked.constants.B / linked.constants.A, rel=1e-12
    )
    assert result.rmse == pytest.approx(linked.fit_statistics.weighted_rmse)


def test_stocking_lord_and_haebara_are_distinct_estimators():
    old, new = _forms(noise=0.15)
    anchors = list(range(10))

    with pytest.warns(DeprecationWarning):
        stocking_lord = equate(old, new, anchors, anchors, method="stocking_lord")
        haebara = equate(old, new, anchors, anchors, method="haebara")

    assert abs(stocking_lord.A - haebara.A) > 1e-4


def test_rejects_unknown_method():
    old, new = _forms()

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Unknown equating method"):
            equate(old, new, [0, 1], [0, 1], method="tcc")


@pytest.mark.parametrize("A", [0.0, -1.0, np.nan])
def test_transform_theta_rejects_invalid_slopes(A):
    result = EquatingResult(A=A, B=0.0, method="mean_mean", anchor_items=[0], rmse=0.0)

    with pytest.raises(MirtValidationError, match="A must"):
        transform_theta(np.zeros(3), result)
