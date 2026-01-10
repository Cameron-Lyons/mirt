"""Integration tests using real datasets.

Tests end-to-end workflows with LSAT6, LSAT7, SAT12, and verbal_aggression datasets.
"""

import numpy as np
import pytest

from mirt import (
    compare_models,
    fit_mirt,
    fscores,
    itemfit,
    list_datasets,
    load_dataset,
    personfit,
)
from mirt.utils.batch import fit_models
from mirt.utils.cv import KFold, LogLikelihoodScorer, cross_validate


class TestLSAT6Workflow:
    """End-to-end tests with LSAT6 dataset."""

    @pytest.fixture(scope="class")
    def lsat6_data(self):
        return load_dataset("LSAT6")

    def test_load_dataset(self, lsat6_data):
        """Test dataset loading."""
        assert "data" in lsat6_data
        assert lsat6_data["n_items"] == 5
        assert lsat6_data["n_persons"] > 0
        assert lsat6_data["data"].shape == (lsat6_data["n_persons"], 5)

    def test_fit_2pl(self, lsat6_data):
        """Test fitting 2PL model."""
        result = fit_mirt(lsat6_data["data"], model="2PL", max_iter=100)

        assert result.converged
        assert result.model.n_items == 5
        assert "discrimination" in result.model.parameters
        assert "difficulty" in result.model.parameters
        assert result.log_likelihood < 0

    def test_full_workflow(self, lsat6_data):
        """Test complete workflow: fit -> score -> diagnostics."""
        responses = lsat6_data["data"]

        result = fit_mirt(responses, model="2PL", max_iter=100)
        assert result.converged

        scores = fscores(result, responses, method="EAP")
        assert scores.theta.shape[0] == responses.shape[0]
        assert np.all(np.isfinite(scores.theta))
        assert np.all(scores.standard_error > 0)

        item_fit = itemfit(result, responses)
        assert len(item_fit) == 5 or item_fit.shape[0] == 5

        person_fit = personfit(result, responses)
        assert (
            len(person_fit) == responses.shape[0]
            or person_fit.shape[0] == responses.shape[0]
        )

    def test_model_comparison(self, lsat6_data):
        """Test comparing 1PL vs 2PL."""
        responses = lsat6_data["data"]

        result_1pl = fit_mirt(responses, model="1PL", max_iter=100)
        result_2pl = fit_mirt(responses, model="2PL", max_iter=100)

        comparison = compare_models([result_1pl, result_2pl])
        assert comparison is not None

        assert result_2pl.log_likelihood >= result_1pl.log_likelihood


class TestLSAT7Workflow:
    """End-to-end tests with LSAT7 dataset."""

    @pytest.fixture(scope="class")
    def lsat7_data(self):
        return load_dataset("LSAT7")

    def test_full_workflow(self, lsat7_data):
        """Test complete workflow."""
        responses = lsat7_data["data"]

        result = fit_mirt(responses, model="2PL", max_iter=200)
        assert result.log_likelihood < 0

        scores = fscores(result, responses, method="EAP")
        assert scores.theta.shape[0] == responses.shape[0]

    def test_cross_validation(self, lsat7_data):
        """Test cross-validation with real data."""
        responses = lsat7_data["data"]

        cv_result = cross_validate(
            model_type="2PL",
            responses=responses,
            splitter=KFold(n_splits=3, random_state=42),
            scorers=[LogLikelihoodScorer()],
            max_iter=50,
        )

        assert cv_result.n_folds == 3
        assert "log_likelihood" in cv_result.mean_scores
        assert len(cv_result.scores["log_likelihood"]) == 3


class TestSAT12Workflow:
    """End-to-end tests with SAT12 dataset."""

    @pytest.fixture(scope="class")
    def sat12_data(self):
        return load_dataset("SAT12")

    def test_load_dataset(self, sat12_data):
        """Test SAT12 loading."""
        assert sat12_data["n_items"] == 12
        assert sat12_data["n_persons"] == 500

    def test_fit_and_score(self, sat12_data):
        """Test fitting and scoring."""
        responses = sat12_data["data"]

        result = fit_mirt(responses, model="2PL", max_iter=100)
        assert result.model.n_items == 12

        scores = fscores(result, responses, method="EAP")
        assert len(scores.theta) == 500

    def test_batch_fitting(self, sat12_data):
        """Test batch model fitting."""
        responses = sat12_data["data"]

        batch_result = fit_models(
            models=["1PL", "2PL"],
            responses=responses,
            max_iter=50,
        )

        assert "1PL" in batch_result.results
        assert "2PL" in batch_result.results
        assert batch_result.best_model in ["1PL", "2PL"]

        summary = batch_result.summary()
        assert "1PL" in summary
        assert "2PL" in summary


class TestVerbalAggressionWorkflow:
    """End-to-end tests with verbal_aggression polytomous dataset."""

    @pytest.fixture(scope="class")
    def va_data(self):
        return load_dataset("verbal_aggression")

    def test_load_dataset(self, va_data):
        """Test verbal_aggression loading."""
        assert va_data["n_items"] == 24
        assert va_data["n_categories"] == 3
        assert va_data["data"].max() == 2
        assert va_data["data"].min() == 0

    def test_fit_grm(self, va_data):
        """Test fitting Graded Response Model."""
        responses = va_data["data"]

        result = fit_mirt(
            responses,
            model="GRM",
            n_categories=3,
            max_iter=100,
        )

        assert result.model.n_items == 24
        assert "discrimination" in result.model.parameters

    def test_polytomous_workflow(self, va_data):
        """Test complete polytomous workflow."""
        responses = va_data["data"]

        result = fit_mirt(
            responses,
            model="GRM",
            n_categories=3,
            max_iter=100,
        )

        scores = fscores(result, responses, method="EAP")
        assert scores.theta.shape[0] == responses.shape[0]


class TestCrossDatasetConsistency:
    """Tests for consistency across datasets."""

    def test_all_datasets_load(self):
        """Test that all datasets load successfully."""
        dataset_names = list_datasets()

        for name in dataset_names:
            data = load_dataset(name)
            assert "data" in data
            assert "n_persons" in data
            assert "n_items" in data
            assert data["data"].shape == (data["n_persons"], data["n_items"])

    def test_fit_dichotomous_datasets(self):
        """Test fitting 2PL to dichotomous datasets."""
        dichotomous = ["LSAT6", "LSAT7"]

        for name in dichotomous:
            data = load_dataset(name)
            result = fit_mirt(data["data"], model="2PL", max_iter=50)
            assert result.model.n_items == data["n_items"]


class TestCrossValidationIntegration:
    """Integration tests for cross-validation framework."""

    def test_cv_lsat7(self):
        """Test CV with LSAT7."""
        data = load_dataset("LSAT7")

        cv_result = cross_validate(
            model_type="2PL",
            responses=data["data"],
            splitter=KFold(n_splits=3, random_state=42),
            max_iter=30,
        )

        assert cv_result.n_folds == 3
        assert all(ll < 0 for ll in cv_result.scores["log_likelihood"])

        summary = cv_result.summary()
        assert "log_likelihood" in summary

    def test_cv_return_models(self):
        """Test CV with return_models=True."""
        data = load_dataset("LSAT6")

        cv_result = cross_validate(
            model_type="2PL",
            responses=data["data"],
            splitter=KFold(n_splits=3, random_state=42),
            max_iter=30,
            return_models=True,
        )

        assert cv_result.fold_results is not None
        assert len(cv_result.fold_results) == 3


class TestBatchFittingIntegration:
    """Integration tests for batch fitting."""

    def test_batch_lsat6(self):
        """Test batch fitting with LSAT6."""
        data = load_dataset("LSAT6")

        batch_result = fit_models(
            models=["1PL", "2PL"],
            responses=data["data"],
            max_iter=50,
        )

        assert len(batch_result.results) == 2

        assert (
            batch_result.results["2PL"].log_likelihood
            >= batch_result.results["1PL"].log_likelihood
        )

    def test_batch_comparison_output(self):
        """Test that batch comparison produces valid DataFrame."""
        data = load_dataset("LSAT7")

        batch_result = fit_models(
            models=["1PL", "2PL"],
            responses=data["data"],
            max_iter=30,
        )

        assert batch_result.comparison is not None

        summary = batch_result.summary()
        assert "1PL" in summary
        assert "2PL" in summary

    def test_get_best_result(self):
        """Test getting the best result."""
        data = load_dataset("LSAT6")

        batch_result = fit_models(
            models=["1PL", "2PL"],
            responses=data["data"],
            max_iter=50,
        )

        best = batch_result.get_best_result()
        assert best is not None
        assert best == batch_result[batch_result.best_model]


class TestCVSplitters:
    """Test cross-validation splitters."""

    @pytest.fixture
    def sample_data(self):
        return load_dataset("LSAT6")["data"]

    def test_kfold_split(self, sample_data):
        """Test KFold splitting."""
        from mirt.utils.cv import KFold

        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        all_test_indices = []
        for train_idx, test_idx in splitter.split(sample_data):
            assert len(train_idx) + len(test_idx) == len(sample_data)
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            all_test_indices.extend(test_idx)

        assert sorted(all_test_indices) == list(range(len(sample_data)))

    def test_stratified_kfold_split(self, sample_data):
        """Test StratifiedKFold splitting."""
        from mirt.utils.cv import StratifiedKFold

        splitter = StratifiedKFold(n_splits=5, n_bins=5, random_state=42)

        fold_count = 0
        for train_idx, test_idx in splitter.split(sample_data):
            assert len(train_idx) + len(test_idx) == len(sample_data)
            fold_count += 1

        assert fold_count == 5

    def test_leave_one_out_split(self, sample_data):
        """Test LeaveOneOut splitting."""
        from mirt.utils.cv import LeaveOneOut

        small_data = sample_data[:20]
        splitter = LeaveOneOut()

        fold_count = 0
        for train_idx, test_idx in splitter.split(small_data):
            assert len(test_idx) == 1
            assert len(train_idx) == len(small_data) - 1
            fold_count += 1

        assert fold_count == len(small_data)
