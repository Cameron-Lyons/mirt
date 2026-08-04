"""Tests for cross-validation module."""

from types import SimpleNamespace

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
    cross_validate,
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

    @pytest.mark.parametrize("n_splits", [0, 1, -1, True, 2.5])
    def test_invalid_split_count(self, n_splits):
        with pytest.raises(ValueError, match="n_splits"):
            KFold(n_splits=n_splits)

    def test_rejects_more_folds_than_persons(self):
        responses = np.zeros((3, 2), dtype=int)
        with pytest.raises(ValueError, match="cannot exceed"):
            list(KFold(n_splits=4).split(responses))


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

    def test_sparse_strata_still_produce_balanced_folds(self):
        responses = np.tri(6, 5, k=-1, dtype=int)
        splitter = StratifiedKFold(n_splits=3, n_bins=6, random_state=42)

        sizes = [len(test_idx) for _, test_idx in splitter.split(responses)]

        assert sizes == [2, 2, 2]

    @pytest.mark.parametrize("n_bins", [0, -1, True, 2.5])
    def test_invalid_bin_count(self, n_bins):
        with pytest.raises(ValueError, match="n_bins"):
            StratifiedKFold(n_bins=n_bins)


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

    def test_requires_two_persons(self):
        with pytest.raises(ValueError, match="at least 2"):
            list(LeaveOneOut().split(np.zeros((1, 3), dtype=int)))


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


class TestCrossValidate:
    @staticmethod
    def _small_responses() -> np.ndarray:
        rng = np.random.default_rng(2026)
        theta = rng.normal(size=24)
        difficulty = np.linspace(-1.0, 1.0, 6)
        probabilities = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty)))
        return (rng.random(probabilities.shape) < probabilities).astype(int)

    def test_parallel_matches_sequential_results(self):
        responses = self._small_responses()
        common = {
            "model_type": "1PL",
            "responses": responses,
            "splitter": KFold(n_splits=2, shuffle=False),
            "n_quadpts": 11,
            "max_iter": 2,
            "return_models": True,
        }

        sequential = cross_validate(**common, n_jobs=1)
        parallel = cross_validate(**common, n_jobs=2)

        assert parallel.n_folds == 2
        assert parallel.fold_results is not None
        assert len(parallel.fold_results) == 2
        assert_allclose(
            parallel.scores["log_likelihood"],
            sequential.scores["log_likelihood"],
        )

    @pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
    def test_rejects_invalid_job_counts(self, n_jobs):
        with pytest.raises(ValueError, match="n_jobs"):
            cross_validate("1PL", self._small_responses(), n_jobs=n_jobs)

    def test_rejects_invalid_response_shapes(self):
        with pytest.raises(ValueError, match="responses"):
            cross_validate("1PL", np.zeros((1, 3), dtype=int))
        with pytest.raises(ValueError, match="responses"):
            cross_validate("1PL", np.zeros(5, dtype=int))

    def test_rejects_empty_or_duplicate_scorers(self):
        responses = self._small_responses()
        with pytest.raises(ValueError, match="at least one"):
            cross_validate("1PL", responses, scorers=[])
        with pytest.raises(ValueError, match="unique"):
            cross_validate(
                "1PL",
                responses,
                scorers=[LogLikelihoodScorer(), LogLikelihoodScorer()],
            )

    def test_rejects_overlapping_custom_split(self):
        class OverlappingSplitter:
            n_splits = 1

            def split(self, responses):
                _ = responses
                yield np.array([0, 1]), np.array([1, 2])

        with pytest.raises(ValueError, match="overlap"):
            cross_validate(
                "1PL",
                self._small_responses(),
                splitter=OverlappingSplitter(),
            )

    def test_rejects_non_integer_custom_split_indices(self):
        class FloatIndexSplitter:
            n_splits = 1

            def split(self, responses):
                _ = responses
                yield np.array([0.0, 1.0]), np.array([2.0])

        with pytest.raises(ValueError, match="integers"):
            cross_validate(
                "1PL",
                self._small_responses(),
                splitter=FloatIndexSplitter(),
            )

    def test_uses_materialized_fold_count(self, monkeypatch):
        class TwoFoldSplitter:
            n_splits = 99

            def split(self, responses):
                midpoint = len(responses) // 2
                yield np.arange(midpoint), np.arange(midpoint, len(responses))
                yield np.arange(midpoint, len(responses)), np.arange(midpoint)

        monkeypatch.setattr(
            "mirt.fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(aic=10.0),
        )
        result = cross_validate(
            "1PL",
            self._small_responses(),
            splitter=TwoFoldSplitter(),
            scorers=[AICScorer()],
        )

        assert result.n_folds == 2
        assert result.scores == {"aic": [-10.0, -10.0]}
