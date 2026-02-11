"""Tests for cross-validation module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.utils.cv import (
    AICScorer,
    BICScorer,
    CVResult,
    KFold,
    LeaveOneOut,
    LogLikelihoodScorer,
    StratifiedKFold,
)


@pytest.fixture(scope="module")
def response_matrix():
    """Binary response matrix for splitter tests."""
    rng = np.random.default_rng(42)
    n_persons, n_items = 50, 10
    theta = rng.standard_normal(n_persons)
    difficulty = rng.normal(0, 1, n_items)
    p = 1 / (1 + np.exp(-(theta[:, None] - difficulty)))
    return (rng.random((n_persons, n_items)) < p).astype(int)


def _assert_valid_splits(splitter, response_matrix, expected_n_splits):
    """Shared assertions for any splitter: fold count, no overlap, full coverage, complementary."""
    n_persons = response_matrix.shape[0]
    folds = list(splitter.split(response_matrix))

    assert len(folds) == expected_n_splits

    all_test = np.concatenate([test for _, test in folds])
    assert len(all_test) == len(np.unique(all_test))
    assert_allclose(np.sort(all_test), np.arange(n_persons))

    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == n_persons
        assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestKFold:
    def test_default_splits(self):
        kf = KFold()
        assert kf.n_splits == 5

    def test_custom_splits(self):
        kf = KFold(n_splits=10)
        assert kf.n_splits == 10

    def test_valid_splits(self, response_matrix):
        kf = KFold(n_splits=5, random_state=42)
        _assert_valid_splits(kf, response_matrix, expected_n_splits=5)

    def test_shuffle_false(self, response_matrix):
        kf = KFold(n_splits=5, shuffle=False)
        first_train, first_test = next(iter(kf.split(response_matrix)))
        assert_allclose(first_test, np.arange(len(first_test)))

    def test_reproducible(self, response_matrix):
        kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
        folds1 = list(kf1.split(response_matrix))
        folds2 = list(kf2.split(response_matrix))
        for (tr1, te1), (tr2, te2) in zip(folds1, folds2):
            assert_allclose(tr1, tr2)
            assert_allclose(te1, te2)

    def test_balanced_fold_sizes(self, response_matrix):
        n_persons = response_matrix.shape[0]
        kf = KFold(n_splits=5, random_state=42)
        sizes = [len(test) for _, test in kf.split(response_matrix)]
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == n_persons


class TestStratifiedKFold:
    def test_default_splits(self):
        skf = StratifiedKFold()
        assert skf.n_splits == 5

    def test_valid_splits(self, response_matrix):
        skf = StratifiedKFold(n_splits=5, random_state=42)
        _assert_valid_splits(skf, response_matrix, expected_n_splits=5)

    def test_stratified_similar_scores(self, response_matrix):
        skf = StratifiedKFold(n_splits=5, n_bins=5, random_state=42)
        sum_scores = np.sum(response_matrix, axis=1)
        global_mean = np.mean(sum_scores)
        for _, test_idx in skf.split(response_matrix):
            fold_mean = np.mean(sum_scores[test_idx])
            assert abs(fold_mean - global_mean) < 2.0


class TestLeaveOneOut:
    def test_n_splits_set_after_split(self):
        loo = LeaveOneOut()
        n_persons = 10
        responses = np.zeros((n_persons, 3), dtype=int)
        list(loo.split(responses))
        assert loo.n_splits == n_persons

    def test_correct_fold_count(self):
        loo = LeaveOneOut()
        n_persons = 8
        responses = np.zeros((n_persons, 3), dtype=int)
        folds = list(loo.split(responses))
        assert len(folds) == n_persons

    def test_single_test(self):
        loo = LeaveOneOut()
        responses = np.zeros((5, 3), dtype=int)
        for _, test_idx in loo.split(responses):
            assert len(test_idx) == 1

    def test_full_coverage(self):
        loo = LeaveOneOut()
        n_persons = 6
        responses = np.zeros((n_persons, 3), dtype=int)
        all_test = [test_idx[0] for _, test_idx in loo.split(responses)]
        assert_allclose(sorted(all_test), np.arange(n_persons))

    def test_train_size(self):
        loo = LeaveOneOut()
        n_persons = 10
        responses = np.zeros((n_persons, 3), dtype=int)
        for train_idx, _ in loo.split(responses):
            assert len(train_idx) == n_persons - 1


class TestLogLikelihoodScorer:
    def test_name(self):
        scorer = LogLikelihoodScorer()
        assert scorer.name == "log_likelihood"


class TestAICScorer:
    def test_name(self):
        scorer = AICScorer()
        assert scorer.name == "aic"


class TestBICScorer:
    def test_name(self):
        scorer = BICScorer()
        assert scorer.name == "bic"


class TestCVResult:
    def test_dataclass_fields(self):
        result = CVResult(
            scores={"log_likelihood": [-10.0, -12.0, -11.0]},
            mean_scores={"log_likelihood": -11.0},
            std_scores={"log_likelihood": 1.0},
            n_folds=3,
        )
        assert result.n_folds == 3
        assert result.fold_results is None

    def test_summary(self):
        result = CVResult(
            scores={"log_likelihood": [-10.0, -12.0, -11.0]},
            mean_scores={"log_likelihood": -11.0},
            std_scores={"log_likelihood": 1.0},
            n_folds=3,
        )
        summary = result.summary()
        assert "Cross-Validation" in summary
        assert "log_likelihood" in summary
        assert "3" in summary

    def test_summary_multiple_metrics(self):
        result = CVResult(
            scores={
                "log_likelihood": [-10.0, -12.0],
                "aic": [-20.0, -22.0],
            },
            mean_scores={"log_likelihood": -11.0, "aic": -21.0},
            std_scores={"log_likelihood": 1.0, "aic": 1.0},
            n_folds=2,
        )
        summary = result.summary()
        assert "log_likelihood" in summary
        assert "aic" in summary
