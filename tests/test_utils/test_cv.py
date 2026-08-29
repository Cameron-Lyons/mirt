"""Tests for cross-validation module."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.utils.cv import (
    AICScorer,
    BICScorer,
    CVResult,
    GroupKFold,
    KFold,
    LeaveOneOut,
    LogLikelihoodScorer,
    StratifiedGroupKFold,
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
        for (tr1, te1), (tr2, te2) in zip(folds1, folds2, strict=True):
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

    def test_rejects_invalid_shuffle_and_response_shape(self):
        with pytest.raises(ValueError, match="shuffle"):
            KFold(shuffle=1)
        with pytest.raises(ValueError, match="two-dimensional"):
            list(KFold(n_splits=2).split(np.zeros(4, dtype=int)))


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

    def test_vectorized_assignments_match_scalar_reference(self):
        rng = np.random.default_rng(20260827)
        responses = rng.integers(0, 2, size=(1_000, 12), dtype=np.int8)
        splitter = StratifiedKFold(n_splits=7, n_bins=9, random_state=123)

        sum_scores = np.sum(np.maximum(responses, 0), axis=1)
        bins = np.unique(
            np.percentile(sum_scores, np.linspace(0, 100, splitter.n_bins + 1))
        )
        strata = np.clip(np.digitize(sum_scores, bins[:-1]) - 1, 0, len(bins) - 2)
        expected_assignments = np.zeros(len(responses), dtype=np.intp)
        reference_rng = np.random.default_rng(splitter.random_state)
        next_fold = 0
        for stratum in range(strata.max() + 1):
            indices = np.flatnonzero(strata == stratum)
            reference_rng.shuffle(indices)
            for offset, index in enumerate(indices):
                expected_assignments[index] = (next_fold + offset) % splitter.n_splits
            next_fold = (next_fold + indices.size) % splitter.n_splits

        folds = list(splitter.split(responses))

        for fold, (_, test_indices) in enumerate(folds):
            np.testing.assert_array_equal(
                test_indices, np.flatnonzero(expected_assignments == fold)
            )

    def test_missing_scores_do_not_change_strata(self):
        responses = np.array(
            [
                [1.0, np.nan, 0.0],
                [1.0, -1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        negative = responses.copy()
        negative[0, 1] = -1.0
        splitter = StratifiedKFold(n_splits=3, n_bins=3, shuffle=False)

        with_nan = list(splitter.split(responses))
        with_negative = list(splitter.split(negative))

        for (_, nan_test), (_, negative_test) in zip(
            with_nan, with_negative, strict=True
        ):
            np.testing.assert_array_equal(nan_test, negative_test)

    def test_shuffle_can_be_disabled(self, response_matrix):
        splitter = StratifiedKFold(n_splits=5, n_bins=5, shuffle=False)

        first = list(splitter.split(response_matrix))
        second = list(splitter.split(response_matrix))

        for (first_train, first_test), (second_train, second_test) in zip(
            first, second, strict=True
        ):
            np.testing.assert_array_equal(first_train, second_train)
            np.testing.assert_array_equal(first_test, second_test)

    def test_rejects_invalid_shuffle_and_non_numeric_scores(self):
        with pytest.raises(ValueError, match="shuffle"):
            StratifiedKFold(shuffle="yes")
        with pytest.raises(ValueError, match="numeric"):
            list(
                StratifiedKFold(n_splits=2).split(
                    np.array([["yes"], ["no"]], dtype=object)
                )
            )


class TestGroupKFold:
    def test_keeps_groups_together_and_balances_rows(self):
        group_sizes = np.array([50, 40, 30, 20, 10, 5])
        groups = np.repeat(np.arange(group_sizes.size), group_sizes)
        responses = np.zeros((groups.size, 3), dtype=int)
        splitter = GroupKFold(groups, n_splits=3)

        folds = list(splitter.split(responses))

        _assert_valid_splits(splitter, responses, expected_n_splits=3)
        assert sorted(test.size for _, test in folds) == [50, 50, 55]
        for train_indices, test_indices in folds:
            assert not set(groups[train_indices]) & set(groups[test_indices])

    def test_equal_sized_groups_use_fast_balanced_assignment(self):
        groups = np.repeat(np.arange(20), 3)
        responses = np.zeros((groups.size, 2), dtype=int)

        sizes = [
            test.size for _, test in GroupKFold(groups, n_splits=5).split(responses)
        ]

        assert sizes == [12, 12, 12, 12, 12]

    def test_shuffling_is_reproducible(self):
        groups = np.repeat(np.arange(30), 2)
        responses = np.zeros((groups.size, 2), dtype=int)
        first = list(
            GroupKFold(groups, n_splits=5, shuffle=True, random_state=42).split(
                responses
            )
        )
        second = list(
            GroupKFold(groups, n_splits=5, shuffle=True, random_state=42).split(
                responses
            )
        )

        for (_, first_test), (_, second_test) in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_test, second_test)

    @pytest.mark.parametrize(
        ("groups", "responses", "message"),
        [
            (np.array([[0], [1]]), np.zeros((2, 1)), "one-dimensional"),
            (np.array([0, 1]), np.zeros((3, 1)), "same number of rows"),
            (np.array([0.0, np.nan]), np.zeros((2, 1)), "non-finite"),
            (np.array([None, "a"], dtype=object), np.zeros((2, 1)), "missing"),
            (
                np.array([float("nan"), "a"], dtype=object),
                np.zeros((2, 1)),
                "missing",
            ),
        ],
    )
    def test_validates_groups(self, groups, responses, message):
        with pytest.raises(ValueError, match=message):
            list(GroupKFold(groups, n_splits=2).split(responses))

    def test_requires_at_least_one_group_per_fold(self):
        responses = np.zeros((4, 1), dtype=int)
        with pytest.raises(ValueError, match="cannot exceed n_groups"):
            list(GroupKFold(np.array([0, 0, 1, 1]), n_splits=3).split(responses))

    def test_is_exported_at_top_level(self):
        import mirt

        assert mirt.GroupKFold is GroupKFold


class TestStratifiedGroupKFold:
    @staticmethod
    def _clustered_score_problem():
        group_scores = np.tile([0, 4, 2], 3)
        group_size = 8
        groups = np.repeat(np.arange(group_scores.size), group_size)
        responses = np.zeros((groups.size, 4), dtype=int)
        for group, score in enumerate(group_scores):
            rows = groups == group
            responses[rows, :score] = 1
        return responses, groups

    def test_keeps_groups_together_and_covers_every_row(self):
        responses, groups = self._clustered_score_problem()
        splitter = StratifiedGroupKFold(groups, n_splits=3, n_bins=3)

        folds = list(splitter.split(responses))

        _assert_valid_splits(splitter, responses, expected_n_splits=3)
        assert [test.size for _, test in folds] == [24, 24, 24]
        for train_indices, test_indices in folds:
            assert not set(groups[train_indices]) & set(groups[test_indices])

    def test_improves_score_balance_over_size_only_grouping(self):
        responses, groups = self._clustered_score_problem()
        sum_scores = responses.sum(axis=1)

        grouped = GroupKFold(groups, n_splits=3)
        stratified = StratifiedGroupKFold(groups, n_splits=3, n_bins=3)
        grouped_means = [
            np.mean(sum_scores[test]) for _, test in grouped.split(responses)
        ]
        stratified_means = [
            np.mean(sum_scores[test]) for _, test in stratified.split(responses)
        ]

        assert np.std(stratified_means) < np.std(grouped_means)
        np.testing.assert_allclose(stratified_means, np.mean(sum_scores))

    def test_shuffling_is_reproducible(self):
        rng = np.random.default_rng(20260828)
        group_sizes = rng.integers(2, 9, size=40)
        groups = np.repeat(np.arange(group_sizes.size), group_sizes)
        responses = rng.integers(0, 2, size=(groups.size, 8))
        first = list(
            StratifiedGroupKFold(
                groups,
                n_splits=5,
                n_bins=4,
                shuffle=True,
                random_state=42,
            ).split(responses)
        )
        second = list(
            StratifiedGroupKFold(
                groups,
                n_splits=5,
                n_bins=4,
                shuffle=True,
                random_state=42,
            ).split(responses)
        )

        for (_, first_test), (_, second_test) in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_test, second_test)

    def test_missing_scores_match_negative_missing_codes(self):
        responses, groups = self._clustered_score_problem()
        with_nan = responses.astype(float)
        with_nan[::7, 0] = np.nan
        with_negative = with_nan.copy()
        with_negative[np.isnan(with_negative)] = -1.0
        splitter = StratifiedGroupKFold(groups, n_splits=3, n_bins=3)

        nan_folds = list(splitter.split(with_nan))
        negative_folds = list(splitter.split(with_negative))

        for (_, nan_test), (_, negative_test) in zip(
            nan_folds, negative_folds, strict=True
        ):
            np.testing.assert_array_equal(nan_test, negative_test)

    def test_sparse_quantile_bins_are_compacted(self):
        counts = np.array([1, 11, 22, 28, 36, 54, 42, 25, 13, 9, 3, 1])
        sum_scores = np.repeat(np.arange(1, 13), counts)
        responses = np.zeros((sum_scores.size, 12), dtype=int)
        for row, score in enumerate(sum_scores):
            responses[row, :score] = 1
        groups = np.arange(len(responses))

        with np.errstate(divide="raise", invalid="raise"):
            folds = list(
                StratifiedGroupKFold(groups, n_splits=5, n_bins=5).split(responses)
            )

        assert len(folds) == 5
        assert all(test.size > 0 for _, test in folds)

    @pytest.mark.parametrize("n_bins", [0, -1, True, 2.5])
    def test_rejects_invalid_bin_count(self, n_bins):
        with pytest.raises(ValueError, match="n_bins"):
            StratifiedGroupKFold(np.arange(4), n_splits=2, n_bins=n_bins)

    def test_reuses_group_and_score_validation(self):
        with pytest.raises(ValueError, match="same number of rows"):
            list(
                StratifiedGroupKFold(np.arange(3), n_splits=2).split(
                    np.zeros((4, 2), dtype=int)
                )
            )
        with pytest.raises(ValueError, match="numeric"):
            list(
                StratifiedGroupKFold(np.arange(4), n_splits=2).split(
                    np.array([["a"], ["b"], ["c"], ["d"]], dtype=object)
                )
            )

    def test_is_exported_from_public_namespaces(self):
        import mirt
        import mirt.utils as utils

        assert mirt.StratifiedGroupKFold is StratifiedGroupKFold
        assert utils.StratifiedGroupKFold is StratifiedGroupKFold
        assert (
            "StratifiedGroupKFold"
            in __import__("mirt.utils.cv", fromlist=["__all__"]).__all__
        )


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
