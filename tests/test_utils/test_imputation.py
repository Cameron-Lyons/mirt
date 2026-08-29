"""Tests for missing data imputation."""

from types import SimpleNamespace

import numpy as np
import pytest

import mirt
import mirt.scoring as scoring_module
import mirt.utils.imputation as imputation_module
from mirt import analyze_missing, averageMI, impute_responses, listwise_deletion
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.imputation import (
    LARGE_DF,
    MIResult,
    _draw_categorical,
    pairwise_available,
)


class TestImputeResponses:
    """Tests for response imputation."""

    def test_impute_mean(self, responses_with_missing):
        """Test mean imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="mean")

        assert np.all(imputed >= 0)

        assert imputed.shape == responses.shape

    def test_impute_mean_uses_rounded_item_values(self):
        responses = np.array([[0, -1], [1, 1], [2, 2], [3, 2]])

        imputed = impute_responses(responses, method="mean")

        np.testing.assert_array_equal(
            imputed,
            np.array([[0, 2], [1, 1], [2, 2], [3, 2]]),
        )

    def test_impute_median_supports_ordered_categories(self):
        responses = np.array(
            [
                [0, -1, 2],
                [1, 1, -1],
                [4, 2, 4],
                [4, 3, 4],
            ]
        )

        imputed = impute_responses(responses, method="median")

        np.testing.assert_array_equal(
            imputed,
            np.array(
                [
                    [0, 2, 2],
                    [1, 1, 4],
                    [4, 2, 4],
                    [4, 3, 4],
                ]
            ),
        )

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("median", np.array([[0, 3], [1000, 5], [1000, 5], [1000, 7]])),
            ("mode", np.array([[0, 3], [1000, 3], [1000, 5], [1000, 7]])),
        ],
    )
    def test_simple_imputation_supports_sparse_large_category_codes(
        self, method, expected
    ):
        responses = np.array([[0, 3], [1000, -1], [-1, 5], [1000, 7]])

        imputed = impute_responses(responses, method=method)

        np.testing.assert_array_equal(imputed, expected)

    def test_impute_mode(self, responses_with_missing):
        """Test mode imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="mode")

        assert np.all(imputed >= 0)

        assert set(imputed.flatten()).issubset({0, 1})

    def test_impute_random(self, responses_with_missing, rng):
        """Test random imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="random", seed=42)

        assert np.all(imputed >= 0)

        assert set(imputed.flatten()).issubset({0, 1})

    def test_impute_em(self, responses_with_missing):
        """Test EM imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="EM")

        assert np.all(imputed >= 0)

    def test_impute_multiple(self, responses_with_missing):
        """Test multiple imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(
            responses,
            method="multiple",
            n_imputations=3,
            seed=42,
        )

        assert isinstance(imputed, list)
        assert len(imputed) == 3

        for imp in imputed:
            assert np.all(imp >= 0)

    def test_no_missing_data(self, dichotomous_responses):
        """Test imputation when no data is missing."""
        responses = dichotomous_responses["responses"]

        imputed = impute_responses(responses, method="mean")

        np.testing.assert_array_equal(imputed, responses)

    def test_complete_multiple_returns_independent_copies(self):
        responses = np.array([[0, 1], [1, 0]])

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=2,
        )

        assert len(imputations) == 2
        np.testing.assert_array_equal(imputations[0], responses)
        assert imputations[0] is not imputations[1]

    @pytest.mark.parametrize(
        "missing_code",
        [1.5, True, np.iinfo(np.int_).max + 1],
    )
    def test_missing_code_must_be_a_supported_integer(self, missing_code):
        with pytest.raises(MirtValidationError, match="missing_code"):
            impute_responses(
                np.array([[0, -1], [1, 1]]),
                method="mean",
                missing_code=missing_code,
            )

    def test_invalid_method_is_rejected_without_missing_data(self):
        responses = np.array([[0, 1], [1, 0]])

        with pytest.raises(MirtValidationError, match="Unknown imputation method"):
            impute_responses(responses, method="unknown")

    @pytest.mark.parametrize("n_imputations", [0, -1, 1.5, True])
    def test_multiple_requires_positive_integer_count(self, n_imputations):
        responses = np.array([[0, 1], [1, 0]])

        with pytest.raises(MirtValidationError, match="positive integer"):
            impute_responses(
                responses,
                method="multiple",
                n_imputations=n_imputations,
            )

    @pytest.mark.parametrize(
        "responses",
        [np.array([0, -1]), np.empty((0, 2)), np.empty((2, 0))],
    )
    def test_invalid_response_shapes_are_rejected(self, responses):
        with pytest.raises(MirtDataError):
            impute_responses(responses, method="mean")

    @pytest.mark.parametrize(
        "method", ["mean", "median", "mode", "random", "EM", "multiple"]
    )
    def test_fully_missing_item_is_rejected(self, method):
        responses = np.array([[-1, 0], [-1, 1]])

        with pytest.raises(MirtDataError, match="no observed responses"):
            impute_responses(responses, method=method, n_imputations=2)

    def test_custom_missing_code_is_imputed(self):
        responses = np.array([[0, 99], [1, 1]])

        imputed = impute_responses(
            responses,
            method="mean",
            missing_code=99,
        )

        np.testing.assert_array_equal(imputed, np.array([[0, 1], [1, 1]]))

    def test_imputation_does_not_mutate_source(self):
        responses = np.array([[0, -1], [1, 1]])
        original = responses.copy()

        impute_responses(responses, method="mode")

        np.testing.assert_array_equal(responses, original)

    def test_multiple_imputation_reuses_fit_and_batches_each_copy(self, monkeypatch):
        responses = np.array([[-1, 0], [1, -1], [0, 1]])
        calls = {"fit": 0, "score": 0, "probability": 0}

        class FakeModel:
            is_polytomous = False

            def probability_pairs(self, theta, item_indices):
                calls["probability"] += 1
                return np.full(len(theta), 0.5)

        fake_model = FakeModel()

        def fake_fit(*args, **kwargs):
            calls["fit"] += 1
            return SimpleNamespace(model=fake_model)

        def fake_scores(*args, **kwargs):
            calls["score"] += 1
            return SimpleNamespace(
                theta=np.zeros(len(responses)),
                standard_error=np.full(len(responses), 0.1),
            )

        monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
        monkeypatch.setattr(scoring_module, "fscores", fake_scores)

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=4,
            seed=42,
        )

        assert calls == {"fit": 1, "score": 1, "probability": 4}
        assert len(imputations) == 4
        assert all(np.all(imputation >= 0) for imputation in imputations)

    def test_multiple_imputation_supports_models_without_paired_api(self, monkeypatch):
        responses = np.zeros((20, 12), dtype=int)
        responses[0, 1] = -1
        responses[1, 9] = -1
        calls: list[int | None] = []

        class FakeModel:
            is_polytomous = False

            def probability(self, theta, item=None):
                calls.append(item)
                if item is None:
                    return np.full((len(theta), responses.shape[1]), 0.5)
                return np.full(len(theta), 0.5)

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            scoring_module,
            "fscores",
            lambda *args, **kwargs: SimpleNamespace(
                theta=np.zeros(len(responses)),
                standard_error=np.full(len(responses), 0.1),
            ),
        )

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=3,
            seed=42,
        )

        assert calls == [1, 9] * 3
        assert all(np.all(imputation >= 0) for imputation in imputations)

    def test_paired_model_batches_respect_probability_memory_limit(self, monkeypatch):
        responses = np.zeros((10, 4), dtype=int)
        rows, columns = np.indices(responses.shape)
        responses[(rows + columns) % 2 == 0] = -1
        batch_sizes: list[int] = []

        class FakeModel:
            is_polytomous = False

            def probability_pairs(self, theta, item_indices):
                batch_sizes.append(len(theta))
                return np.full(len(theta), 0.5)

        monkeypatch.setattr(imputation_module, "_MODEL_DRAW_TARGET_ELEMENTS", 8)
        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            scoring_module,
            "fscores",
            lambda *args, **kwargs: SimpleNamespace(
                theta=np.zeros(len(responses)),
                standard_error=np.full(len(responses), 0.1),
            ),
        )

        impute_responses(
            responses,
            method="multiple",
            n_imputations=2,
            seed=42,
        )

        assert batch_sizes == [4] * 10

    def test_dense_polytomous_imputation_batches_all_items(self, monkeypatch):
        responses = np.array(
            [
                [-1, 0, -1],
                [1, -1, 2],
                [-1, 2, 0],
                [2, -1, -1],
            ]
        )
        calls = 0

        class FakeModel:
            is_polytomous = True
            n_categories = [3, 3, 3]

            def probability_pairs(self, theta, item_indices):
                nonlocal calls
                calls += 1
                probabilities = np.zeros((len(theta), 3))
                probabilities[:, 2] = 1.0
                return probabilities

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            scoring_module,
            "fscores",
            lambda *args, **kwargs: SimpleNamespace(
                theta=np.zeros(len(responses)),
                standard_error=np.full(len(responses), 0.1),
            ),
        )

        imputations = impute_responses(
            responses,
            method="multiple",
            model="GRM",
            n_imputations=3,
            seed=42,
        )

        assert calls == 3
        for imputed in imputations:
            np.testing.assert_array_equal(imputed[responses == -1], 2)

    def test_multiple_imputation_falls_back_when_scoring_fails(self, monkeypatch):
        responses = np.array([[-1, 0], [1, -1], [0, 1]])

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=object()),
        )

        def fail_scoring(*args, **kwargs):
            raise RuntimeError("scoring failed")

        monkeypatch.setattr(scoring_module, "fscores", fail_scoring)

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=3,
            seed=42,
        )

        assert len(imputations) == 3
        assert all(np.all(imputation >= 0) for imputation in imputations)


class TestAnalyzeMissing:
    """Tests for missing data analysis."""

    def test_analyze_missing(self, responses_with_missing):
        """Test missing data analysis."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        assert (
            "total_missing" in analysis
            or "total_missing_rate" in analysis
            or "n_missing" in analysis
        )

    def test_missing_by_item(self, responses_with_missing):
        """Test missing data by item."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        if "item_missing_rate" in analysis:
            n_items = responses.shape[1]
            assert len(analysis["item_missing_rate"]) == n_items

    def test_missing_by_person(self, responses_with_missing):
        """Test missing data by person."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        if "person_missing_rate" in analysis:
            n_persons = responses.shape[0]
            assert len(analysis["person_missing_rate"]) == n_persons

    def test_exact_summary_and_custom_missing_code(self):
        responses = np.array([[0, 99, 1], [1, 0, 99], [1, 1, 1]])

        analysis = analyze_missing(responses, missing_code=99)

        assert analysis["total_missing_rate"] == pytest.approx(2 / 9)
        np.testing.assert_allclose(analysis["item_missing_rate"], [0, 1 / 3, 1 / 3])
        np.testing.assert_allclose(analysis["person_missing_rate"], [1 / 3, 1 / 3, 0])
        assert analysis["n_complete_cases"] == 1
        assert analysis["n_complete_items"] == 1


class TestListwiseDeletion:
    """Tests for listwise deletion."""

    def test_listwise_deletion(self, responses_with_missing):
        """Test listwise deletion."""
        responses = responses_with_missing["responses"]

        clean = listwise_deletion(responses)

        assert np.all(clean >= 0)

        assert clean.shape[0] <= responses.shape[0]

        assert clean.shape[1] == responses.shape[1]

    def test_listwise_preserves_complete(self, dichotomous_responses):
        """Test that listwise preserves complete data."""
        responses = dichotomous_responses["responses"]

        clean = listwise_deletion(responses)

        assert clean.shape[0] == responses.shape[0]


class TestPairwiseAvailable:
    def test_counts_available_responses_and_pairs(self):
        responses = np.array([[1, -1, 0], [0, 1, -1], [-1, 1, 1]])

        available, joint = pairwise_available(responses)

        np.testing.assert_array_equal(available, [2, 2, 2])
        np.testing.assert_array_equal(
            joint,
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
        )

    def test_blocked_counts_do_not_overflow(self):
        responses = np.ones((600, 3), dtype=int)
        responses[:17, 1] = -1
        responses[20:49, 2] = -1

        available, joint = pairwise_available(responses)

        np.testing.assert_array_equal(available, [600, 583, 571])
        np.testing.assert_array_equal(
            joint,
            [[600, 583, 571], [583, 583, 554], [571, 554, 571]],
        )
        assert joint.dtype == np.dtype(np.int_)


def test_categorical_draws_respect_degenerate_probabilities():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.0]])

    draws = _draw_categorical(probabilities, np.random.default_rng(42))

    np.testing.assert_array_equal(draws, [0, 1, 2])


def test_categorical_draws_require_a_probability_matrix():
    with pytest.raises(MirtDataError, match="probabilities"):
        _draw_categorical(np.array([0.5, 0.5]), np.random.default_rng(42))


class TestAverageMI:
    def test_scalar_rubins_rules(self):
        result = averageMI([1.0, 3.0], variances=[1.0, 1.0])

        assert isinstance(result, MIResult)
        assert result.estimate == pytest.approx(2.0)
        assert result.within_variance == pytest.approx(1.0)
        assert result.between_variance == pytest.approx(2.0)
        assert result.total_variance == pytest.approx(4.0)
        assert result.standard_error == pytest.approx(2.0)
        assert result.lambda_hat == pytest.approx(0.75)

    def test_array_standard_errors(self):
        result = averageMI(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            standard_errors=[np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        )

        np.testing.assert_allclose(result.estimate, [2.0, 3.0])
        np.testing.assert_allclose(result.within_variance, [1.0, 4.0])
        np.testing.assert_allclose(result.total_variance, [4.0, 7.0])

    def test_zero_between_variance_is_stable(self):
        with np.errstate(all="raise"):
            result = averageMI([1.0, 1.0], variances=[1.0, 1.0])

        assert result.df == LARGE_DF
        assert result.lambda_hat == 0.0
        assert np.isfinite(result.fmi)

    @pytest.mark.parametrize(
        ("variances", "standard_errors"),
        [(None, None), ([1.0, 1.0], [1.0, 1.0])],
    )
    def test_requires_exactly_one_uncertainty_source(self, variances, standard_errors):
        with pytest.raises(MirtValidationError, match="exactly one"):
            averageMI(
                [1.0, 2.0],
                variances=variances,
                standard_errors=standard_errors,
            )

    def test_uncertainty_count_must_match_imputations(self):
        with pytest.raises(MirtValidationError, match="number"):
            averageMI([1.0, 2.0, 3.0], variances=[1.0, 1.0])

    def test_at_least_two_imputations_are_required(self):
        with pytest.raises(MirtValidationError, match="at least 2"):
            averageMI([1.0], variances=[1.0])

    def test_all_shapes_must_match(self):
        with pytest.raises(MirtValidationError, match="same shape"):
            averageMI(
                [np.array([1.0]), np.array([2.0, 3.0])],
                variances=[np.array([1.0]), np.array([1.0])],
            )

        with pytest.raises(MirtValidationError, match="estimate shape"):
            averageMI(
                [np.array([1.0]), np.array([2.0])],
                variances=[np.array([1.0, 2.0]), np.array([1.0, 2.0])],
            )

    @pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
    def test_uncertainty_must_be_finite_and_nonnegative(self, invalid):
        with pytest.raises(MirtValidationError):
            averageMI([1.0, 2.0], variances=[1.0, invalid])

    def test_estimates_must_be_finite(self):
        with pytest.raises(MirtValidationError, match="finite"):
            averageMI([1.0, np.nan], variances=[1.0, 1.0])

    def test_inputs_must_be_numeric(self):
        with pytest.raises(MirtValidationError, match="estimates"):
            averageMI([1.0, "invalid"], variances=[1.0, 1.0])

        with pytest.raises(MirtValidationError, match="Uncertainty"):
            averageMI([1.0, 2.0], variances=[1.0, "invalid"])
