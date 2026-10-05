"""Tests for item fit statistics."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.diagnostics.itemfit as itemfit_module
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2
from mirt.utils.numeric import compute_fit_stats, compute_probability_moments


class FixedProbabilityModel:
    """Minimal model returning fixed all-item probabilities."""

    def __init__(self, probabilities, n_categories=None):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.n_items = self.probabilities.shape[1]
        self._n_categories = n_categories
        self.probability_calls = 0

    @property
    def is_polytomous(self):
        return self._n_categories is not None

    @property
    def n_categories(self):
        if self._n_categories is None:
            raise AttributeError("dichotomous models do not have categories")
        return list(self._n_categories)

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        if len(theta) != len(self.probabilities):
            raise ValueError("theta length must match fixed probabilities")
        if item_idx is None:
            return self.probabilities.copy()
        return self.probabilities[:, item_idx].copy()


class IndexedProbabilityModel(FixedProbabilityModel):
    """Use ability values as row indices to test arbitrary probability blocks."""

    def __init__(self, probabilities, n_categories=None):
        super().__init__(probabilities, n_categories)
        self.batch_sizes = []

    def probability(self, theta, item_idx=None):
        assert item_idx is None
        self.batch_sizes.append(len(theta))
        return self.probabilities[np.asarray(theta[:, 0], dtype=np.intp)].copy()


@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("block_rows", [1, 7, 1000])
@pytest.mark.parametrize("statistics", [["infit"], ["outfit"], ["infit", "outfit"]])
def test_itemfit_blocks_preserve_population_statistics(
    monkeypatch, polytomous, block_rows, statistics
):
    rng = np.random.default_rng(4218)
    n_persons, n_items = 103, 4
    categories = [2, 4, 3, 5] if polytomous else None
    if polytomous:
        probabilities = rng.uniform(0.1, 1.0, size=(n_persons, n_items, 5))
        probabilities *= np.arange(5) < np.asarray(categories)[:, None]
        probabilities /= probabilities.sum(axis=2, keepdims=True)
    else:
        probabilities = rng.uniform(0.05, 0.95, size=(n_persons, n_items))
        probabilities[:10, 1] = 0.0
        probabilities[10:20, 1] = 1.0
    responses = rng.integers(
        0, categories if polytomous else 2, size=(n_persons, n_items)
    )
    responses[rng.random(responses.shape) < 0.2] = -9
    responses[0] = -1
    responses[:, 3] = -1
    responses.setflags(write=False)
    theta = np.arange(n_persons, dtype=float)[:, None]
    model = IndexedProbabilityModel(probabilities, categories)
    _, expected, variance = compute_probability_moments(model, theta, n_items)
    infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)
    reference = {"infit": infit, "outfit": outfit}
    model.batch_sizes.clear()
    monkeypatch.setattr(
        itemfit_module,
        "_ITEMFIT_TARGET_CHUNK_ELEMENTS",
        block_rows * n_items * (5 if polytomous else 1),
    )
    monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", 9)

    actual = compute_itemfit(
        model,
        responses,
        theta=theta,
        statistics=statistics,
    )

    assert max(model.batch_sizes) <= block_rows
    assert sum(model.batch_sizes) == n_persons
    for name in statistics:
        assert_allclose(actual[name], reference[name], rtol=1e-12, atol=1e-12)
    assert set(actual) == set(statistics)


def test_itemfit_accumulates_small_variances_before_infit_threshold(monkeypatch):
    from mirt.constants import PROB_EPSILON

    model = IndexedProbabilityModel(np.full((9, 1), PROB_EPSILON / 2))
    responses = np.ones((9, 1))
    theta = np.arange(9)[:, None]
    monkeypatch.setattr(itemfit_module, "_ITEMFIT_TARGET_CHUNK_ELEMENTS", 1)

    result = compute_itemfit(model, responses, theta=theta)

    p = PROB_EPSILON / 2
    assert_allclose(result["infit"], [(1 - p) / p])
    assert np.isnan(result["outfit"][0])


@pytest.mark.parametrize(
    "theta", [np.empty((0, 1)), np.zeros((2, 1)), np.zeros((3, 1, 1))]
)
@pytest.mark.parametrize("compute", [compute_itemfit, compute_s_x2])
def test_itemfit_rejects_unaligned_theta(theta, compute):
    model = FixedProbabilityModel(np.full((3, 2), 0.5))
    with pytest.raises(ValueError, match="one row per person"):
        compute(model, np.ones((3, 2)), theta=theta)
    assert model.probability_calls == 0


class TestComputeItemfit:
    """Tests for compute_itemfit function."""

    def test_basic_itemfit(self, fitted_2pl_model, dichotomous_responses):
        """Test basic item fit computation."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result
        assert len(result["infit"]) == dichotomous_responses["n_items"]
        assert len(result["outfit"]) == dichotomous_responses["n_items"]

    def test_itemfit_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that fit statistics are positive."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] > 0)
        assert np.all(result["outfit"] > 0)

    def test_itemfit_values_reasonable_range(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that fit statistics are in reasonable range."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] < 3.0)
        assert np.all(result["outfit"] < 3.0)

    def test_itemfit_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test item fit with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in result
        assert "outfit" in result

    def test_itemfit_statistics_subset(self, fitted_2pl_model, dichotomous_responses):
        """Test computing only subset of statistics."""
        model = fitted_2pl_model.model
        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            statistics=["infit"],
        )

        assert "infit" in result
        assert "outfit" not in result

    def test_itemfit_no_responses_raises_error(self, fitted_2pl_model):
        """Test that missing responses raises error."""
        model = fitted_2pl_model.model
        with pytest.raises(ValueError, match="responses required"):
            compute_itemfit(model, responses=None)

    def test_itemfit_default_statistics(self, fitted_2pl_model, dichotomous_responses):
        """Test default statistics are infit and outfit."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result

    def test_itemfit_supports_s_x2(self, fitted_2pl_model, dichotomous_responses):
        """Test the documented S-X2 statistic through the item-fit API."""
        model = fitted_2pl_model.model

        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            statistics=["S_X2"],
        )

        assert set(result) == {"S_X2", "df", "p_value"}
        assert len(result["S_X2"]) == dichotomous_responses["n_items"]

    def test_public_itemfit_exposes_s_x2(
        self,
        fitted_2pl_model,
        dichotomous_responses,
    ):
        """Test the top-level interface exposes conditional S-X2."""
        from mirt import itemfit

        result = itemfit(
            fitted_2pl_model,
            dichotomous_responses["responses"],
            statistics=["S_X2"],
        )

        assert {"S_X2", "df", "p_value"}.issubset(result.columns)


class TestComputeSX2:
    """Tests for compute_s_x2 function."""

    def test_basic_s_x2(self, fitted_2pl_model, dichotomous_responses):
        """Test basic S-X2 computation."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert "S_X2" in result
        assert "df" in result
        assert "p_value" in result
        assert len(result["S_X2"]) == dichotomous_responses["n_items"]

    def test_s_x2_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that S-X2 values are non-negative."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["S_X2"] >= 0)

    def test_s_x2_df_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that degrees of freedom are positive."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["df"] >= 1)

    def test_s_x2_p_values_in_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that p-values are in [0, 1]."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["p_value"] >= 0)
        assert np.all(result["p_value"] <= 1)

    def test_s_x2_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test S-X2 with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "S_X2" in result

    def test_s_x2_standard_normal_quadrature(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test S-X2 with a denser latent quadrature."""
        model = fitted_2pl_model.model
        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
        )

        assert "S_X2" in result

    @pytest.mark.parametrize("n_groups", [True, 1, 0, -1, 2.5])
    def test_s_x2_rejects_invalid_group_count(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        n_groups,
    ):
        """Test score-group validation fails clearly and consistently."""
        with pytest.raises(ValueError, match="n_groups"):
            compute_s_x2(
                fitted_2pl_model.model,
                dichotomous_responses["responses"],
                theta=np.zeros(dichotomous_responses["n_persons"]),
                n_groups=n_groups,
            )


class TestItemfitWithPolytomousModel:
    """Tests for item fit with polytomous models."""

    def test_itemfit_polytomous(self, polytomous_responses):
        """Test item fit with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        fit_result = compute_itemfit(
            result.model, responses=polytomous_responses["responses"]
        )

        assert "infit" in fit_result
        assert "outfit" in fit_result
        assert len(fit_result["infit"]) == polytomous_responses["n_items"]


class TestItemfitEdgeCases:
    """Tests for edge cases in item fit computation."""

    def test_itemfit_perfect_fit(self, fitted_2pl_model):
        """Test item fit when responses match model expectations perfectly."""
        model = fitted_2pl_model.model
        n_persons = 30
        responses = np.zeros((n_persons, model.n_items), dtype=int)

        from mirt.scoring import fscores

        theta = fscores(model, responses, method="EAP").theta

        result = compute_itemfit(model, responses=responses, theta=theta)

        assert np.all(np.isfinite(result["infit"]))
        assert np.all(np.isfinite(result["outfit"]))

    def test_itemfit_consistency(self, fitted_2pl_model, dichotomous_responses):
        """Test that item fit is consistent across calls."""
        model = fitted_2pl_model.model
        result1 = compute_itemfit(model, responses=dichotomous_responses["responses"])
        result2 = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert_allclose(result1["infit"], result2["infit"])
        assert_allclose(result1["outfit"], result2["outfit"])


def _reference_standardized_fit(responses, probabilities, axis):
    """Scalar Wright-Masters mean squares and Wilson-Hilferty z statistics."""
    from mirt.constants import PROB_EPSILON

    if probabilities.ndim == 2:
        probabilities = np.stack((1.0 - probabilities, probabilities), axis=2)
    scores = np.arange(probabilities.shape[2])
    n_groups = responses.shape[1 - axis]
    reference = {
        name: np.full(n_groups, np.nan)
        for name in ("infit", "outfit", "z_infit", "z_outfit")
    }
    for group in range(n_groups):
        squared, variance, fourth = [], [], []
        for other in range(responses.shape[axis]):
            person, item = (other, group) if axis == 0 else (group, other)
            if responses[person, item] < 0:
                continue
            cell = probabilities[person, item]
            mean = np.sum(scores * cell)
            squared.append((responses[person, item] - mean) ** 2)
            variance.append(np.sum((scores - mean) ** 2 * cell))
            fourth.append(np.sum((scores - mean) ** 4 * cell))
        squared, variance, fourth = map(np.asarray, (squared, variance, fourth))
        eligible = variance > PROB_EPSILON
        count = eligible.sum()
        if count:
            outfit = np.mean(squared[eligible] / variance[eligible])
            q2 = np.sum(fourth[eligible] / variance[eligible] ** 2) / count**2
            q2 -= 1.0 / count
            reference["outfit"][group] = outfit
            if q2 > 0:
                q = np.sqrt(q2)
                reference["z_outfit"][group] = (outfit ** (1 / 3) - 1) * 3 / q + q / 3
        if variance.sum() > PROB_EPSILON:
            infit = squared.sum() / variance.sum()
            q2 = np.sum(fourth - variance**2) / variance.sum() ** 2
            reference["infit"][group] = infit
            if q2 > 0:
                q = np.sqrt(q2)
                reference["z_infit"][group] = (infit ** (1 / 3) - 1) * 3 / q + q / 3
    return reference


class TestStandardizedMeanSquares:
    """Wilson-Hilferty z_infit and z_outfit."""

    def test_hand_computed_binary_item(self):
        model = FixedProbabilityModel(np.array([[0.2], [0.5], [0.8]]))
        responses = np.array([[1], [0], [1]])

        result = compute_itemfit(
            model,
            responses,
            statistics=["infit", "outfit", "z_infit", "z_outfit"],
            theta=np.zeros(3),
        )

        # Outfit: (0.64/0.16 + 0.25/0.25 + 0.04/0.16) / 3 = 1.75 with
        # q^2 = sum((1 - 3W) / W) / 9 - 1/3 = 7.5 / 9 - 1 / 3 = 0.5.
        q_out = np.sqrt(0.5)
        # Infit: 0.93 / 0.57 with q^2 = sum(W - 4W^2) / 0.57^2.
        q_in = np.sqrt(0.1152 / 0.57**2)
        assert_allclose(result["outfit"], [1.75])
        assert_allclose(result["infit"], [0.93 / 0.57])
        assert_allclose(
            result["z_outfit"], [(1.75 ** (1 / 3) - 1) * 3 / q_out + q_out / 3]
        )
        assert_allclose(
            result["z_infit"],
            [((0.93 / 0.57) ** (1 / 3) - 1) * 3 / q_in + q_in / 3],
        )

    @pytest.mark.parametrize("polytomous", [False, True])
    @pytest.mark.parametrize("block_rows", [1, 7, 1000])
    def test_blocks_match_scalar_reference(self, monkeypatch, polytomous, block_rows):
        rng = np.random.default_rng(4219)
        n_persons, n_items = 61, 4
        categories = [2, 4, 3, 5] if polytomous else None
        if polytomous:
            probabilities = rng.uniform(0.1, 1.0, size=(n_persons, n_items, 5))
            probabilities *= np.arange(5) < np.asarray(categories)[:, None]
            probabilities /= probabilities.sum(axis=2, keepdims=True)
        else:
            probabilities = rng.uniform(0.05, 0.95, size=(n_persons, n_items))
            probabilities[:10, 1] = 0.0
        responses = rng.integers(
            0, categories if polytomous else 2, size=(n_persons, n_items)
        )
        if not polytomous:
            responses[:10, 1] = 0
        responses[rng.random(responses.shape) < 0.2] = -9
        responses[:, 3] = -1
        theta = np.arange(n_persons, dtype=float)[:, None]
        model = IndexedProbabilityModel(probabilities, categories)
        monkeypatch.setattr(
            itemfit_module,
            "_ITEMFIT_TARGET_CHUNK_ELEMENTS",
            block_rows * n_items * (5 if polytomous else 1),
        )

        actual = compute_itemfit(
            model,
            responses,
            theta=theta,
            statistics=["z_infit", "z_outfit", "infit", "outfit"],
        )

        reference = _reference_standardized_fit(responses, probabilities, axis=0)
        assert max(model.batch_sizes) <= block_rows
        assert list(actual) == ["outfit", "z_outfit", "infit", "z_infit"]
        for name, values in reference.items():
            assert_allclose(actual[name], values, rtol=1e-11, equal_nan=True)
        assert np.isnan(actual["z_infit"][3])

    def test_approximately_standard_normal_under_true_rasch_model(self):
        from mirt.models import OneParameterLogistic

        rng = np.random.default_rng(1)
        n_items = 60
        model = OneParameterLogistic(n_items=n_items)
        model.set_parameters(difficulty=rng.normal(size=n_items))
        theta = rng.normal(size=(1000, 1))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)

        result = compute_itemfit(
            model, responses, statistics=["z_infit", "z_outfit"], theta=theta
        )

        for name in ("z_infit", "z_outfit"):
            assert abs(np.mean(result[name])) < 0.3
            assert 0.7 < np.std(result[name]) < 1.3

    @pytest.mark.filterwarnings("ignore:z_infit and z_outfit with EAP")
    def test_nan_codes_missing_responses_like_negative_codes(self):
        from mirt.models import TwoParameterLogistic

        rng = np.random.default_rng(12)
        model = TwoParameterLogistic(n_items=5)
        model.set_parameters(
            discrimination=rng.uniform(0.8, 2.0, 5), difficulty=rng.normal(size=5)
        )
        model._is_fitted = True
        responses = rng.integers(0, 2, size=(120, 5)).astype(float)
        missing = rng.random(responses.shape) < 0.1
        statistics = ["infit", "outfit", "z_infit", "z_outfit", "X2", "PV_Q1"]

        responses[missing] = np.nan
        with_nan = compute_itemfit(model, responses, statistics, n_plausible=3, seed=0)
        responses[missing] = -1
        with_negative = compute_itemfit(
            model, responses, statistics, n_plausible=3, seed=0
        )

        assert with_nan.keys() == with_negative.keys()
        for name, values in with_negative.items():
            assert_allclose(with_nan[name], values, equal_nan=True)

    def test_constant_half_probabilities_have_undefined_z(self):
        model = FixedProbabilityModel(np.full((4, 2), 0.5))
        responses = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])

        result = compute_itemfit(
            model, responses, statistics=["z_infit", "z_outfit"], theta=np.zeros(4)
        )

        assert np.all(np.isnan(result["z_infit"]))
        assert np.all(np.isnan(result["z_outfit"]))


class TestStatisticNames:
    """Unknown statistic names fail instead of being silently dropped."""

    @pytest.mark.parametrize(
        "statistics", [["X3"], ["infit", "s_x2"], ["Zh"], ["infit", "lz"], []]
    )
    def test_compute_itemfit_rejects_unknown_names(self, statistics):
        from mirt.exceptions import MirtValidationError

        model = FixedProbabilityModel(np.full((3, 2), 0.5))
        with pytest.raises(MirtValidationError, match="statistic"):
            compute_itemfit(model, np.ones((3, 2)), statistics, theta=np.zeros(3))
        assert model.probability_calls == 0

    def test_single_name_string_is_accepted(self):
        model = FixedProbabilityModel(np.full((3, 2), 0.5))

        result = compute_itemfit(
            model, np.array([[0, 1], [1, 0], [1, 1]]), "outfit", theta=np.zeros(3)
        )

        assert set(result) == {"outfit"}

    def test_top_level_itemfit_rejects_unknown_names(
        self, fitted_2pl_model, dichotomous_responses
    ):
        from mirt import itemfit

        with pytest.raises(ValueError, match="Unknown fit statistic"):
            itemfit(
                fitted_2pl_model,
                dichotomous_responses["responses"],
                statistics=["infit", "X2*"],
            )

    def test_statistic_literal_lists_every_supported_name(self):
        from typing import get_args

        from mirt.typing import ItemFitStatistic

        assert set(get_args(ItemFitStatistic)) == {
            "infit",
            "outfit",
            "z_infit",
            "z_outfit",
            "S_X2",
            "X2",
            "G2",
            "PV_Q1",
        }


def _two_parameter_sample(seed=21, n_items=20, n_persons=1000):
    from mirt.models import TwoParameterLogistic

    rng = np.random.default_rng(seed)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.uniform(0.8, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    theta = rng.normal(size=(n_persons, 1))
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(int)
    return model, responses, theta


def test_standardized_fit_with_eap_abilities_warns_about_its_bias():
    model, responses, theta = _two_parameter_sample()

    with pytest.warns(UserWarning, match="biased toward overfit"):
        eap = compute_itemfit(model, responses, ["z_infit", "z_outfit"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        known = compute_itemfit(model, responses, ["z_infit", "z_outfit"], theta=theta)
        compute_itemfit(model, responses, ["infit", "outfit"])

    # The documented calibration: centered with true abilities, strongly
    # negative with EAP abilities from the same responses.
    assert abs(np.mean(known["z_infit"])) < 0.4
    assert np.mean(eap["z_infit"]) < -1.0
    assert np.mean(eap["z_infit"] < -1.96) > 0.25


def test_top_level_itemfit_accepts_external_abilities():
    import mirt
    from mirt.results.fit_result import FitResult

    model, responses, theta = _two_parameter_sample(n_persons=300)
    result = FitResult(
        model=model,
        log_likelihood=-1.0,
        n_iterations=1,
        converged=True,
        standard_errors={},
        aic=0.0,
        bic=0.0,
        n_observations=responses.shape[0],
        n_parameters=40,
    )
    statistics = ["infit", "z_infit"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = mirt.itemfit(result, responses, statistics, theta=theta)
    expected = compute_itemfit(model, responses, statistics, theta=theta)

    for name in statistics:
        assert_allclose(np.asarray(frame[name]), expected[name])
