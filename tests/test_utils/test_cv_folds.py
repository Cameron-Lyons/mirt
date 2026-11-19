"""Regressions for fold data, fit options, and cross-validation scheduling."""

from types import SimpleNamespace

import numpy as np
import pytest

from mirt.utils.cv import (
    AICScorer,
    GroupKFold,
    KFold,
    LeaveOneOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)


@pytest.mark.parametrize("model_type", ["GRM", ["GRM", "2PL", "2PL", "2PL"]])
def test_fold_models_share_item_categories_and_names(model_type):
    pd = pytest.importorskip("pandas")
    responses = np.random.default_rng(20).integers(0, 2, (36, 4))
    responses[:3, 0] = 2  # The first training fold never observes category 2.
    responses[5, 1] = -1
    names = ["rare", "reading", "science", "math"]
    frame = pd.DataFrame(responses, columns=names, dtype="Int64")
    frame.iloc[5, 1] = pd.NA
    options = dict(splitter=KFold(3, shuffle=False), n_quadpts=7, max_iter=2)

    inferred = cross_validate(model_type, frame, return_models=True, **options)
    explicit = cross_validate(
        model_type, responses, n_categories=[3, 2, 2, 2], **options
    )

    assert np.all(np.isfinite(inferred.scores["log_likelihood"]))
    np.testing.assert_allclose(
        inferred.scores["log_likelihood"], explicit.scores["log_likelihood"]
    )
    assert all(fold.model.item_names == names for fold in inferred.fold_results)


def test_custom_splitter_can_refill_index_buffers(monkeypatch):
    class ReusingSplitter:
        n_splits = 2

        def split(self, responses):
            train = np.array([2, 3])
            test = np.array([0, 1])
            yield train, test
            train[:] = [0, 1]
            test[:] = [2, 3]
            yield train, test

    class RowScorer(AICScorer):
        def __call__(self, result, train_responses, test_responses, test_indices=None):
            np.testing.assert_array_equal(test_responses[:, 0], test_indices)
            np.testing.assert_array_equal(result.train, train_responses)
            return float(test_indices[0])

    monkeypatch.setattr(
        "mirt.fit_mirt", lambda data, **kwargs: SimpleNamespace(train=data)
    )
    cv = cross_validate("1PL", np.arange(4)[:, None], ReusingSplitter(), [RowScorer()])
    assert cv.scores == {"aic": [0.0, 2.0]}


@pytest.mark.parametrize(
    "return_models,custom,expected",
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_standard_errors_remain_available_to_custom_scorers_and_returned_models(
    monkeypatch, return_models, custom, expected
):
    class CustomScorer(AICScorer):
        def __call__(self, result, *args):
            return len(result.standard_errors)

    compute_se = []

    def fit(data, *, compute_standard_errors, **kwargs):
        compute_se.append(compute_standard_errors)
        return SimpleNamespace(
            aic=3.0, standard_errors={"slope": 0.5} if compute_standard_errors else {}
        )

    monkeypatch.setattr("mirt.fit_mirt", fit)
    cv = cross_validate(
        "2PL",
        np.zeros((6, 2), dtype=int),
        KFold(3, shuffle=False),
        [CustomScorer() if custom else AICScorer()],
        return_models=return_models,
    )
    assert compute_se == [expected] * 3
    assert cv.scores["aic"] == ([1.0] if custom else [-3.0]) * 3
    if return_models:
        assert all(fold.standard_errors for fold in cv.fold_results)


def test_replaced_builtin_scorer_receives_standard_errors(monkeypatch):
    def fit(data, *, compute_standard_errors, **options):
        return SimpleNamespace(
            standard_errors={"slope": 0.5} if compute_standard_errors else {}
        )

    monkeypatch.setattr("mirt.fit_mirt", fit)
    monkeypatch.setattr(
        AICScorer,
        "__call__",
        lambda self, result, *args: result.standard_errors["slope"],
    )
    cv = cross_validate("2PL", np.zeros((6, 2), dtype=int), KFold(2), [AICScorer()])
    assert cv.scores["aic"] == [0.5, 0.5]


@pytest.mark.parametrize("workers", [1, 2])
def test_leave_one_out_scores_while_folds_are_still_being_generated(
    monkeypatch, workers
):
    from concurrent.futures import ThreadPoolExecutor

    from mirt.utils.cv import _BUILTIN_SPLITS

    n_persons = 32
    generated = 0
    scored = 0
    max_pending = 0
    original_split = LeaveOneOut.split

    def recorded_split(self, responses):
        nonlocal generated, max_pending
        for fold in original_split(self, responses):
            generated += 1
            max_pending = max(max_pending, generated - scored)
            yield fold

    class CheckOrder(AICScorer):
        def __call__(self, result, train_responses, test_responses, test_indices=None):
            nonlocal scored
            assert test_indices.tolist() == [scored]
            scored += 1
            return result.aic

    monkeypatch.setattr(LeaveOneOut, "split", recorded_split)
    monkeypatch.setitem(_BUILTIN_SPLITS, LeaveOneOut, recorded_split)
    monkeypatch.setattr(
        "mirt.fit_mirt", lambda *args, **kwargs: SimpleNamespace(aic=0.0)
    )
    monkeypatch.setattr(
        "mirt.utils._parallel._process_pool",
        lambda count, context: ThreadPoolExecutor(count),
    )
    cv = cross_validate(
        "1PL",
        np.zeros((n_persons, 2), dtype=int),
        LeaveOneOut(),
        [CheckOrder()],
        n_jobs=workers,
    )
    assert cv.n_folds == n_persons
    assert scored == generated == n_persons
    assert max_pending <= 2 * workers


@pytest.mark.parametrize("override_class", [False, True])
def test_split_override_on_builtin_still_validates_folds(monkeypatch, override_class):
    splitter = KFold(n_splits=3)
    monkeypatch.setattr(
        KFold if override_class else splitter,
        "split",
        lambda *args: iter([(np.array([0, 1]), np.array([1, 2]))]),
    )
    with pytest.raises(ValueError, match="overlap"):
        cross_validate("1PL", np.zeros((6, 2), dtype=int), splitter)


@pytest.mark.parametrize(
    "n_factors, expected", [(1, 21), (3, 21), (4, 9), (5, 7), (6, 5)]
)
def test_default_cv_grid_is_bounded_across_dimensions(monkeypatch, n_factors, expected):
    points = []

    def fit(*args, n_quadpts, **kwargs):
        points.append(n_quadpts)
        return SimpleNamespace(aic=0.0)

    monkeypatch.setattr("mirt.fit_mirt", fit)
    cross_validate(
        "2PL", np.zeros((6, 8), dtype=int), KFold(2), [AICScorer()], n_factors=n_factors
    )
    assert points == [expected, expected]


def test_default_cv_rejects_impractical_grids_before_fitting():
    with pytest.raises(ValueError, match="pass n_quadpts explicitly"):
        cross_validate("2PL", np.zeros((6, 9), dtype=int), n_factors=7)


@pytest.mark.parametrize("splitter_type", [GroupKFold, StratifiedGroupKFold])
@pytest.mark.parametrize("label_dtype", ["datetime64[D]", object])
def test_missing_time_labels_rejected_for_grouped_folds(splitter_type, label_dtype):
    groups = np.array(
        [
            np.datetime64("2020-01-01"),
            np.datetime64("NaT"),
            np.datetime64("2020-01-03"),
        ],
        dtype=label_dtype,
    )
    with pytest.raises(ValueError, match="missing"):
        list(
            splitter_type(groups, n_splits=2).split(
                np.ones((len(groups), 2), dtype=int)
            )
        )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("rare_score", [0, 1])
def test_quantile_ties_do_not_collapse_distinct_score_groups(grouped, rare_score):
    responses = np.full((12, 1), 1 - rare_score)
    responses[::3] = rare_score
    splitter = (
        StratifiedGroupKFold(np.arange(12), 3, n_bins=5)
        if grouped
        else StratifiedKFold(3, n_bins=5, shuffle=False)
    )
    counts = [
        np.count_nonzero(responses[test] == rare_score)
        for _, test in splitter.split(responses)
    ]
    assert sorted(counts) == [1, 1, 2]
