"""Tests for missing data imputation."""

import numpy as np

from mirt import analyze_missing, impute_responses, listwise_deletion


class TestImputeResponses:
    """Tests for response imputation."""

    def test_impute_mean(self, responses_with_missing):
        """Test mean imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="mean")

        assert np.all(imputed >= 0)

        assert imputed.shape == responses.shape

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
