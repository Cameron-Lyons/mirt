"""Tests for ability-grouped X2, G2 and PV-Q1 item fit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt.diagnostics.itemfit_binned as binned_module
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.utils.empirical import _build_theta_bins


def _two_item_model() -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=2)
    model.set_parameters(discrimination=np.ones(2), difficulty=np.zeros(2))
    return model


def _reference_grouped_fit(model, responses, theta, n_groups):
    """Scalar X2/G2 reference with the shared ability bins."""
    group, group_theta = _build_theta_bins(theta, n_groups)
    probabilities = np.asarray(model.probability(group_theta[:, None]))
    if probabilities.ndim == 2:
        probabilities = np.stack((1.0 - probabilities, probabilities), axis=2)
    categories = (
        np.asarray(model.n_categories)
        if model.is_polytomous
        else np.full(model.n_items, 2)
    )
    x2 = np.zeros(model.n_items)
    g2 = np.zeros(model.n_items)
    df = np.zeros(model.n_items, dtype=int)
    for item in range(model.n_items):
        for g in range(n_groups):
            members = (group == g) & (responses[:, item] >= 0)
            size = members.sum()
            if size == 0:
                continue
            df[item] += categories[item] - 1
            for category in range(categories[item]):
                observed = np.sum(responses[members, item] == category)
                expected = size * probabilities[g, item, category]
                x2[item] += (observed - expected) ** 2 / expected
                if observed > 0:
                    g2[item] += 2.0 * observed * np.log(observed / expected)
    return x2, g2, df


class TestGroupedStatistics:
    def test_hand_computed_two_group_example(self) -> None:
        model = _two_item_model()
        theta = np.repeat([-1.0, 1.0], 4)
        responses = np.array(
            [[0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [1, 0], [0, 1]]
        )

        result = compute_itemfit(
            model,
            responses,
            statistics=["X2", "G2"],
            theta=theta,
            n_groups=2,
            item_parameter_counts=np.zeros(2, dtype=int),
        )

        low = 1.0 / (1.0 + np.e)
        high = 1.0 - low
        # Item 0: 1 of 4 positive at theta=-1 and 3 of 4 at theta=1.
        # Item 1: 1 of 4 positive at theta=-1 and 3 of 4 at theta=1.
        expected_x2 = 2 * (4 * (0.25 - low) ** 2 / (low * high))
        expected_g2 = 2 * (
            2.0 * (1 * np.log(1 / (4 * low)) + 3 * np.log(3 / (4 * high)))
        )
        assert_allclose(result["X2"], [expected_x2, expected_x2], rtol=1e-12)
        assert_allclose(result["G2"], [expected_g2, expected_g2], rtol=1e-12)
        assert_allclose(result["X2_df"], [2, 2])
        assert_allclose(result["X2_p"], stats.chi2.sf(result["X2"], 2), rtol=1e-12)
        assert_allclose(result["G2_p"], stats.chi2.sf(result["G2"], 2), rtol=1e-12)

    def test_default_degrees_of_freedom_subtract_free_parameters(self) -> None:
        model = _two_item_model()
        theta = np.repeat([-1.0, 1.0], 4)
        responses = np.tile([[0, 1], [1, 0]], (4, 1))

        result = compute_itemfit(
            model, responses, statistics=["X2"], theta=theta, n_groups=2
        )

        np.testing.assert_array_equal(result["X2_df"], [0, 0])
        assert np.all(np.isnan(result["X2_p"]))

    @pytest.mark.parametrize("chunk_elements", [1, 7, 1_000_000])
    def test_polytomous_missing_data_matches_scalar_reference(
        self, monkeypatch, chunk_elements
    ) -> None:
        rng = np.random.default_rng(42)
        model = GradedResponseModel(4, n_categories=[2, 3, 5, 4])
        theta = rng.normal(size=300)
        probabilities = model.probability(theta[:, None])
        draws = rng.random((300, 4, 1))
        responses = (draws > probabilities.cumsum(axis=2)).sum(axis=2)
        responses = np.minimum(responses, np.array([1, 2, 4, 3]))
        responses = responses.astype(float)
        responses[rng.random(responses.shape) < 0.1] = np.nan
        responses[rng.random(responses.shape) < 0.05] = -1
        monkeypatch.setattr(
            binned_module, "_BINNED_TARGET_CHUNK_ELEMENTS", chunk_elements
        )

        result = compute_itemfit(
            model,
            responses,
            statistics=["X2", "G2"],
            theta=theta,
            n_groups=6,
            item_parameter_counts=np.ones(4, dtype=int),
        )

        coded = np.where(np.isnan(responses), -1, responses).astype(int)
        x2, g2, df = _reference_grouped_fit(model, coded, theta, 6)
        assert_allclose(result["X2"], x2, rtol=1e-12)
        assert_allclose(result["G2"], g2, rtol=1e-12)
        np.testing.assert_array_equal(result["X2_df"], df - 1)

    def test_adjusted_p_values_follow_requested_method(self) -> None:
        from mirt.diagnostics.multiple_testing import adjust_p_values

        rng = np.random.default_rng(3)
        model = _two_item_model()
        theta = rng.normal(size=200)
        responses = (rng.random((200, 2)) < 0.5).astype(int)

        result = compute_itemfit(
            model,
            responses,
            statistics=["X2", "G2"],
            theta=theta,
            p_adjust="holm",
            item_parameter_counts=np.zeros(2, dtype=int),
        )

        for name in ("X2", "G2"):
            assert_allclose(
                result[f"{name}_p_adjusted"],
                adjust_p_values(result[f"{name}_p"], "holm"),
            )
        assert "p_value_adjusted" not in result

    def test_unobserved_category_with_zero_probability_is_infinite(self) -> None:
        model = _two_item_model()
        model.set_parameters(discrimination=np.array([60.0, 1.0]))
        theta = np.repeat([-1.0, 1.0], 4)
        responses = np.zeros((8, 2), dtype=int)
        responses[0, 0] = 1

        result = compute_itemfit(
            model,
            responses,
            statistics=["X2", "G2"],
            theta=theta,
            n_groups=2,
            item_parameter_counts=np.zeros(2, dtype=int),
        )

        assert result["X2"][0] == np.inf
        assert result["G2"][0] == np.inf
        assert np.isfinite(result["X2"][1])


class TestPlausibleValueQ1:
    def test_seeded_draws_are_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        model = TwoParameterLogistic(n_items=6)
        model.set_parameters(
            discrimination=rng.uniform(0.8, 2.0, 6), difficulty=rng.normal(size=6)
        )
        model._is_fitted = True
        theta = rng.normal(size=(400, 1))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)

        first = compute_itemfit(
            model, responses, statistics=["PV_Q1"], n_plausible=5, seed=11
        )
        second = compute_itemfit(
            model, responses, statistics=["PV_Q1"], n_plausible=5, seed=11
        )

        assert set(first) == {"PV_Q1", "PV_Q1_df", "PV_Q1_p"}
        assert_allclose(first["PV_Q1"], second["PV_Q1"])
        assert np.all(first["PV_Q1_df"] == 10 - 2)

        # Plausible values come from the EAP posterior, not supplied abilities.
        supplied = compute_itemfit(
            model,
            responses,
            statistics=["PV_Q1", "X2"],
            theta=np.full(400, 3.0),
            n_plausible=5,
            seed=11,
        )
        assert_allclose(supplied["PV_Q1"], first["PV_Q1"])
        # Identical supplied abilities form one group: 1 contrast - 2 < 0.
        assert np.all(supplied["X2_df"] == 0)

    def test_true_model_rejects_less_often_than_naive_x2(self) -> None:
        rng = np.random.default_rng(100)
        n_items = 20
        model = TwoParameterLogistic(n_items=n_items)
        model.set_parameters(
            discrimination=rng.uniform(0.8, 2.0, n_items),
            difficulty=rng.normal(size=n_items),
        )
        model._is_fitted = True
        theta = rng.normal(size=(1500, 1))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)

        result = compute_itemfit(
            model,
            responses,
            statistics=["X2", "PV_Q1"],
            item_parameter_counts=np.zeros(n_items, dtype=int),
            n_plausible=30,
            seed=0,
        )

        # X2 groups on shrunken EAP scores and is liberal for short tests.
        assert np.mean(result["PV_Q1_p"] < 0.05) <= 0.1
        assert np.mean(result["X2_p"] < 0.05) > np.mean(result["PV_Q1_p"] < 0.05)

    def test_flags_guessing_items_fit_with_a_2pl(self) -> None:
        from mirt import fit_mirt
        from mirt.models import ThreeParameterLogistic

        rng = np.random.default_rng(0)
        n_items = 20
        generator = ThreeParameterLogistic(n_items=n_items)
        guessing = np.zeros(n_items)
        guessing[:2] = 0.4
        difficulty = rng.normal(size=n_items)
        difficulty[:2] = 1.5
        discrimination = rng.uniform(1.0, 2.0, n_items)
        discrimination[:2] = 2.5
        generator.set_parameters(
            discrimination=discrimination,
            difficulty=difficulty,
            guessing=guessing,
        )
        theta = rng.normal(size=(2000, 1))
        probabilities = generator.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)
        fitted = fit_mirt(responses, model="2PL", verbose=False)

        result = compute_itemfit(
            fitted.model, responses, statistics=["PV_Q1"], n_plausible=50, seed=1
        )

        assert np.all(result["PV_Q1_p"][:2] < 0.05)
        assert np.all(result["PV_Q1_p"][2:] > 0.05)


class TestBinnedOptions:
    def test_n_groups_is_not_deprecated_when_it_groups_x2(self) -> None:
        rng = np.random.default_rng(5)
        model = _two_item_model()
        theta = rng.normal(size=60)
        responses = (rng.random((60, 2)) < 0.5).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = compute_itemfit(
                model,
                responses,
                statistics=["S_X2", "X2"],
                theta=theta,
                n_groups=4,
                item_parameter_counts=np.zeros(2, dtype=int),
            )
        assert np.all(result["X2_df"] == 4)

        with pytest.warns(DeprecationWarning, match="exact total scores"):
            compute_s_x2(
                model,
                responses,
                n_groups=4,
                item_parameter_counts=np.zeros(2, dtype=int),
            )

    @pytest.mark.parametrize("n_groups", [True, 1, 2.5])
    def test_rejects_invalid_group_counts(self, n_groups) -> None:
        model = _two_item_model()
        with pytest.raises(ValueError, match="n_groups"):
            compute_itemfit(
                model,
                np.zeros((4, 2), dtype=int),
                statistics=["X2"],
                theta=np.zeros(4),
                n_groups=n_groups,
            )

    @pytest.mark.parametrize("n_plausible", [0, True, 2.0])
    def test_rejects_invalid_plausible_draw_counts(self, n_plausible) -> None:
        model = _two_item_model()
        with pytest.raises(ValueError, match="n_plausible"):
            compute_itemfit(
                model,
                np.zeros((4, 2), dtype=int),
                statistics=["PV_Q1"],
                n_plausible=n_plausible,
            )

    def test_rejects_multidimensional_models(self) -> None:
        model = TwoParameterLogistic(n_items=3, n_factors=2)
        with pytest.raises(ValueError, match="unidimensional"):
            compute_itemfit(
                model,
                np.zeros((5, 3), dtype=int),
                statistics=["G2"],
                theta=np.zeros((5, 2)),
            )

    def test_rejects_out_of_range_categories(self) -> None:
        model = _two_item_model()
        responses = np.zeros((4, 2), dtype=int)
        responses[0, 1] = 2
        with pytest.raises(ValueError, match="category codes"):
            compute_itemfit(model, responses, statistics=["X2"], theta=np.zeros(4))
