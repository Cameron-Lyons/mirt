"""Cross-model regression coverage for information and reliability utilities."""

import numpy as np
import pytest
from scipy import stats

import mirt
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils import information as information_utils
from mirt.utils.information import areainfo, expected_score, iteminfo
from mirt.utils.reliability import empirical_rxx, marginal_rxx, sem


@pytest.fixture(params=["dichotomous", "polytomous"])
def model(request):
    if request.param == "dichotomous":
        return TwoParameterLogistic(n_items=3)
    return GeneralizedPartialCredit(n_items=3, n_categories=4)


def test_information_utilities_return_consistent_item_and_test_values(model):
    theta = np.array([-1.0, 0.0, 1.0])
    theta_2d = theta[:, None]
    expected_items = np.column_stack(
        [model.information(theta_2d, item_idx=idx) for idx in range(model.n_items)]
    )

    actual_items = iteminfo(model, theta)

    assert actual_items.shape == (3, 3)
    np.testing.assert_allclose(actual_items, expected_items)
    np.testing.assert_allclose(
        information_utils.testinfo(model, theta), expected_items.sum(axis=1)
    )
    np.testing.assert_allclose(iteminfo(model, theta, 1), expected_items[:, 1])
    np.testing.assert_allclose(
        iteminfo(model, theta, [2, 0]), expected_items[:, [2, 0]]
    )


def test_expected_score_honors_item_selection(model):
    theta = np.array([-1.0, 0.0, 1.0])
    theta_2d = theta[:, None]
    expected_items = np.column_stack(
        [model.expected_score(theta_2d, item_idx=idx) for idx in range(model.n_items)]
    )

    np.testing.assert_allclose(expected_score(model, theta, 1), expected_items[:, 1])
    np.testing.assert_allclose(
        expected_score(model, theta, [2, 0]), expected_items[:, [2, 0]]
    )
    np.testing.assert_allclose(expected_score(model, theta), expected_items.sum(axis=1))
    assert expected_score(model, theta, []).shape == (3, 0)


def test_area_and_sem_use_total_polytomous_information():
    model = GeneralizedPartialCredit(n_items=4, n_categories=4)
    theta = np.linspace(-2.0, 2.0, 31)
    information = information_utils.testinfo(model, theta)

    np.testing.assert_allclose(sem(model, theta), 1.0 / np.sqrt(information))
    assert areainfo(model, (-2.0, 2.0), n_points=31) == pytest.approx(
        np.trapezoid(information, theta)
    )


def test_marginal_reliability_uses_the_selected_latent_range_variance():
    model = GeneralizedPartialCredit(n_items=20, n_categories=4)
    theta = np.linspace(-1.0, 1.0, 41)
    weights = stats.norm.pdf(theta)
    weights[[0, -1]] *= 0.5
    weights /= weights.sum()
    latent_mean = np.sum(weights * theta)
    latent_variance = np.sum(weights * (theta - latent_mean) ** 2)
    information = information_utils.testinfo(model, theta)
    local_reliability = information / (information + 1.0 / latent_variance)
    expected = np.clip(np.sum(weights * local_reliability), 0.0, 1.0)

    assert marginal_rxx(model, (-1.0, 1.0), n_points=41) == pytest.approx(expected)


def test_empirical_reliability_supports_polytomous_information():
    model = GeneralizedPartialCredit(n_items=20, n_categories=4)
    theta = np.linspace(-2.0, 2.0, 21)
    observed_variance = np.var(theta, ddof=1)
    error_variance = np.mean(1.0 / information_utils.testinfo(model, theta))
    expected = np.clip(
        observed_variance / (observed_variance + error_variance),
        0.0,
        1.0,
    )

    assert empirical_rxx(model, theta) == pytest.approx(expected)
    assert empirical_rxx(model, theta, method="information") == pytest.approx(expected)


def test_reliability_rejects_unsupported_options():
    model = TwoParameterLogistic(n_items=3)

    with pytest.raises(ValueError, match="density must be"):
        marginal_rxx(model, density="other")

    with pytest.raises(ValueError, match="method must be"):
        empirical_rxx(model, np.zeros(3), method="unknown")


def test_score_utilities_are_available_from_the_top_level_api():
    assert mirt.expected_score is expected_score
    assert callable(mirt.expected_test_score)
    assert callable(mirt.theta_for_score)
