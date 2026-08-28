from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics.dif import (
    _compute_grdif_statistics,
    _compute_robust_scale,
    _expected_response_matrix,
    _score_grdif_responses,
    compute_grdif,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel


class _DichotomousModel:
    n_factors = 1
    is_fitted = True

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.difficulty = np.linspace(-1.5, 1.5, n_items)
        self.discrimination = np.linspace(0.7, 1.8, n_items)
        self.calls: list[int | None] = []

    def probability(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        self.calls.append(item_idx)
        theta_values = np.asarray(theta, dtype=np.float64)[:, 0]
        logits = self.discrimination[None, :] * (
            theta_values[:, None] - self.difficulty[None, :]
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        if item_idx is not None:
            return probabilities[:, item_idx]
        return probabilities


def _scalar_grdif_reference(
    data: np.ndarray,
    theta: np.ndarray,
    model: _DichotomousModel,
    group_masks: dict[Any, np.ndarray],
    unique_groups: np.ndarray,
    scaling_method: str,
) -> tuple[np.ndarray, ...]:
    n_items = data.shape[1]
    n_groups = len(unique_groups)
    grdif_r = np.zeros(n_items)
    grdif_s = np.zeros(n_items)

    for item_index in range(n_items):
        mrr_values: list[float] = []
        msr_values: list[float] = []
        var_mrr_values: list[float] = []
        var_msr_values: list[float] = []
        counts: list[int] = []
        for group in unique_groups:
            mask = group_masks[group]
            responses = data[mask, item_index]
            group_theta = theta[mask]
            valid = responses >= 0
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 2:
                mrr_values.append(0.0)
                msr_values.append(0.0)
                var_mrr_values.append(1.0)
                var_msr_values.append(1.0)
                counts.append(1)
                continue

            expected = model.probability(
                group_theta[valid], item_idx=item_index
            ).ravel()
            residuals = responses[valid] - expected
            squared = residuals**2
            mrr_values.append(float(np.mean(residuals)))
            msr_values.append(float(np.mean(squared)))
            var_mrr_values.append(
                max(
                    _compute_robust_scale(residuals, scaling_method) / n_valid,
                    PROB_EPSILON,
                )
            )
            var_msr_values.append(
                max(
                    _compute_robust_scale(squared, scaling_method) / n_valid,
                    PROB_EPSILON,
                )
            )
            counts.append(n_valid)

        mrr = np.asarray(mrr_values)
        msr = np.asarray(msr_values)
        var_mrr = np.asarray(var_mrr_values)
        var_msr = np.asarray(var_msr_values)
        weights = np.asarray(counts) / np.sum(counts)
        grdif_r[item_index] = np.sum((mrr - np.sum(weights * mrr)) ** 2 / var_mrr)
        grdif_s[item_index] = np.sum((msr - np.sum(weights * msr)) ** 2 / var_msr)

    grdif_rs = grdif_r + grdif_s
    return (
        grdif_r,
        grdif_s,
        grdif_rs,
        stats.chi2.sf(grdif_r, df=n_groups - 1),
        stats.chi2.sf(grdif_s, df=n_groups - 1),
        stats.chi2.sf(grdif_rs, df=2 * (n_groups - 1)),
    )


@pytest.mark.parametrize("scaling_method", ["mean", "mad", "iqr"])
def test_vectorized_statistics_match_independent_scalar_reference(
    scaling_method: str,
) -> None:
    rng = np.random.default_rng(7201)
    n_people = 173
    n_items = 19
    groups = np.repeat(np.arange(4), [41, 43, 44, 45])
    theta = rng.normal(size=(n_people, 1))
    data = rng.integers(0, 2, size=(n_people, n_items), dtype=np.int64)
    data[rng.random(data.shape) < 0.17] = -1
    data[groups == 0, 0] = -1
    data[np.flatnonzero(groups == 0)[0], 0] = 1
    masks = {group: groups == group for group in np.unique(groups)}
    labels = np.unique(groups)
    vectorized_model = _DichotomousModel(n_items)
    scalar_model = _DichotomousModel(n_items)

    actual = _compute_grdif_statistics(
        data,
        theta,
        vectorized_model,
        masks,
        labels,
        scaling_method,
    )
    expected = _scalar_grdif_reference(
        data,
        theta,
        scalar_model,
        masks,
        labels,
        scaling_method,
    )

    for actual_values, expected_values in zip(actual, expected, strict=True):
        assert_allclose(actual_values, expected_values, rtol=2e-13, atol=2e-13)
    assert vectorized_model.calls == [None]
    assert len(scalar_model.calls) > n_items


@pytest.mark.parametrize("model_type", [GradedResponseModel, GeneralizedPartialCredit])
def test_expected_response_matrix_supports_polytomous_models(
    model_type: type[GradedResponseModel] | type[GeneralizedPartialCredit],
) -> None:
    model = model_type(n_items=3, n_categories=[3, 4, 5])
    theta = np.linspace(-2.0, 2.0, 13)[:, None]

    actual = _expected_response_matrix(model, theta, n_items=3)

    expected = np.column_stack(
        [
            model.probability(theta, item_idx=item)
            @ np.arange(model.n_categories[item])
            for item in range(model.n_items)
        ]
    )
    assert_allclose(actual, expected)


def test_statistics_reuse_precomputed_expected_responses() -> None:
    model = _DichotomousModel(5)
    data = np.zeros((8, 5), dtype=np.int64)
    theta = np.linspace(-1.0, 1.0, 8)[:, None]
    groups = np.repeat([0, 1], 4)
    masks = {group: groups == group for group in np.unique(groups)}
    expected = model.probability(theta)
    model.calls.clear()

    result = _compute_grdif_statistics(
        data,
        theta,
        model,
        masks,
        np.unique(groups),
        expected_responses=expected,
    )

    assert all(values.shape == (5,) for values in result)
    assert model.calls == []


@pytest.mark.parametrize(
    ("anchor_items", "message"),
    [
        (np.ones(3, dtype=np.bool_), "number of response columns"),
        (np.zeros(2, dtype=np.bool_), "at least one anchor"),
    ],
)
def test_scoring_rejects_invalid_anchor_sets(
    anchor_items: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _score_grdif_responses(
            _DichotomousModel(2),
            np.zeros((4, 2), dtype=np.int64),
            anchor_items,
            scoring_method="EAP",
            n_quadpts=31,
        )


@pytest.mark.parametrize(
    ("theta", "message"),
    [
        (np.zeros((3, 1)), "one row per response record"),
        (np.full((4, 1), np.nan), "finite values"),
    ],
)
def test_scoring_rejects_invalid_ability_estimates(
    monkeypatch: pytest.MonkeyPatch,
    theta: np.ndarray,
    message: str,
) -> None:
    monkeypatch.setattr(
        "mirt.scoring.fscores",
        lambda *args, **kwargs: SimpleNamespace(theta=theta),
    )

    with pytest.raises(ValueError, match=message):
        _score_grdif_responses(
            _DichotomousModel(2),
            np.zeros((4, 2), dtype=np.int64),
            np.ones(2, dtype=np.bool_),
            scoring_method="EAP",
            n_quadpts=31,
        )


@pytest.mark.parametrize(
    ("data", "theta", "scaling_method", "expected", "message"),
    [
        (np.zeros(4), np.zeros((4, 1)), "mean", None, "two-dimensional"),
        (
            np.zeros((4, 2)),
            np.zeros((3, 1)),
            "mean",
            None,
            "one row per person",
        ),
        (
            np.zeros((4, 2)),
            np.zeros((4, 1)),
            "unknown",
            None,
            "scaling_method",
        ),
        (
            np.zeros((4, 2)),
            np.zeros((4, 1)),
            "mean",
            np.zeros((4, 3)),
            "expected_responses",
        ),
        (
            np.zeros((4, 2)),
            np.zeros((4, 1)),
            "mean",
            np.full((4, 2), np.nan),
            "expected_responses",
        ),
    ],
)
def test_statistics_validate_direct_inputs(
    data: np.ndarray,
    theta: np.ndarray,
    scaling_method: str,
    expected: np.ndarray | None,
    message: str,
) -> None:
    groups = np.array([0, 0, 1, 1])
    masks = {group: groups == group for group in np.unique(groups)}

    with pytest.raises(ValueError, match=message):
        _compute_grdif_statistics(
            data,
            theta,
            _DichotomousModel(2),
            masks,
            np.unique(groups),
            scaling_method,  # type: ignore[arg-type]
            expected_responses=expected,
        )


def test_purification_rescores_with_only_current_anchor_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    groups = np.repeat(["reference", "focal"], 3)
    model = _DichotomousModel(data.shape[1])
    scoring_inputs: list[np.ndarray] = []
    statistics_calls = 0

    def fake_fit(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(model=model)

    def fake_scores(
        fitted_model: Any,
        responses: np.ndarray,
        **kwargs: Any,
    ) -> SimpleNamespace:
        assert fitted_model is model
        scoring_inputs.append(responses.copy())
        active = responses >= 0
        theta = np.sum(np.where(active, responses, 0), axis=1) / np.maximum(
            np.count_nonzero(active, axis=1), 1
        )
        return SimpleNamespace(theta=theta[:, None])

    def fake_statistics(*args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        nonlocal statistics_calls
        statistics_calls += 1
        zeros = np.zeros(data.shape[1])
        p_values = np.array([0.001, 0.5, 0.5, 0.5])
        return zeros, zeros, zeros, p_values, p_values, p_values

    monkeypatch.setattr("mirt.fit_mirt", fake_fit)
    monkeypatch.setattr("mirt.scoring.fscores", fake_scores)
    monkeypatch.setattr(
        "mirt.diagnostics.dif._compute_grdif_statistics", fake_statistics
    )

    result = compute_grdif(
        data,
        groups,
        purify=True,
        max_purify_iter=5,
    )

    assert len(scoring_inputs) == 2
    assert_array_equal(scoring_inputs[0], data)
    assert_array_equal(scoring_inputs[1][:, 0], -1)
    assert_array_equal(scoring_inputs[1][:, 1:], data[:, 1:])
    assert_array_equal(result["anchor_items"], [False, True, True, True])
    assert result["purification_complete"] is True
    assert result["purification_stop_reason"] == "converged"
    assert [entry["n_anchors"] for entry in result["purification_history"]] == [
        3,
        3,
    ]
    assert_allclose(
        result["theta"],
        np.mean(data[:, 1:], axis=1)[:, None],
    )
    assert statistics_calls == 3


def test_purification_reports_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.tile([0, 1, 0], (8, 1)).astype(np.int64)
    groups = np.repeat([0, 1], 4)
    model = _DichotomousModel(data.shape[1])

    monkeypatch.setattr(
        "mirt.fit_mirt", lambda *args, **kwargs: SimpleNamespace(model=model)
    )
    monkeypatch.setattr(
        "mirt.scoring.fscores",
        lambda fitted_model, responses, **kwargs: SimpleNamespace(
            theta=np.mean(np.where(responses >= 0, responses, 0), axis=1)[:, None]
        ),
    )

    def always_flags_first(*args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        zeros = np.zeros(data.shape[1])
        p_values = np.array([0.001, 0.5, 0.5])
        return zeros, zeros, zeros, p_values, p_values, p_values

    monkeypatch.setattr(
        "mirt.diagnostics.dif._compute_grdif_statistics", always_flags_first
    )

    result = compute_grdif(data, groups, purify=True, max_purify_iter=1)

    assert result["purification_complete"] is False
    assert result["purification_stop_reason"] == "max_iterations"
    assert_array_equal(result["anchor_items"], [False, True, True])


def test_purification_rejects_an_unusable_anchor_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.tile([0, 1, 0], (8, 1)).astype(np.int64)
    groups = np.repeat([0, 1], 4)
    model = _DichotomousModel(data.shape[1])
    score_calls = 0

    monkeypatch.setattr(
        "mirt.fit_mirt", lambda *args, **kwargs: SimpleNamespace(model=model)
    )

    def fake_scores(*args: Any, **kwargs: Any) -> SimpleNamespace:
        nonlocal score_calls
        score_calls += 1
        return SimpleNamespace(theta=np.zeros((len(data), 1)))

    def flags_everything(*args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        zeros = np.zeros(data.shape[1])
        p_values = np.full(data.shape[1], 0.001)
        return zeros, zeros, zeros, p_values, p_values, p_values

    monkeypatch.setattr("mirt.scoring.fscores", fake_scores)
    monkeypatch.setattr(
        "mirt.diagnostics.dif._compute_grdif_statistics", flags_everything
    )

    result = compute_grdif(data, groups, purify=True)

    assert score_calls == 1
    assert result["purification_complete"] is False
    assert result["purification_stop_reason"] == "insufficient_anchors"
    assert_array_equal(result["anchor_items"], [True, True, True])
    assert result["purification_history"][0]["n_anchors"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "NRM"}, "model"),
        ({"scoring_method": "EAPsum"}, "scoring_method"),
        ({"purify_by": "unknown"}, "purify_by"),
        ({"scaling_method": "unknown"}, "scaling_method"),
        ({"alpha": 0.0}, "alpha"),
        ({"purify": "yes"}, "purify"),
        ({"max_purify_iter": 0}, "max_purify_iter"),
        ({"n_quadpts": 1}, "n_quadpts"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_compute_grdif_validates_configuration_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    def unexpected_fit(*args: Any, **fit_kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.fit_mirt", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        compute_grdif(
            np.zeros((4, 2), dtype=np.int64),
            np.array([0, 0, 1, 1]),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("data", "groups", "message"),
    [
        (np.empty((0, 2)), np.array([]), "nonempty"),
        (np.zeros(4), np.array([0, 0, 1, 1]), "two-dimensional"),
        (np.full((4, 2), np.nan), np.array([0, 0, 1, 1]), "finite"),
        (np.full((4, 2), 0.5), np.array([0, 0, 1, 1]), "integer coded"),
        (np.full((4, 2), -2), np.array([0, 0, 1, 1]), "integer coded"),
        (np.zeros((4, 2)), np.array([[0], [0], [1], [1]]), "one-dimensional"),
        (np.zeros((4, 2)), np.array([0, 1]), "length"),
        (np.zeros((4, 2)), np.array([0, 0, 0, 0]), "at least 2 groups"),
        (np.zeros((4, 2)), np.array([0.0, 0.0, 1.0, np.nan]), "missing"),
        (
            np.zeros((4, 2)),
            np.array(["a", "a", None, "b"], dtype=object),
            "missing",
        ),
        (
            np.zeros((4, 2)),
            np.array([0, "a", 0, "a"], dtype=object),
            "mutually comparable",
        ),
    ],
)
def test_compute_grdif_validates_data_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
    data: np.ndarray,
    groups: np.ndarray,
    message: str,
) -> None:
    def unexpected_fit(*args: Any, **kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.fit_mirt", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        compute_grdif(data, groups)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.ones(4),
        np.ones((4, 3)),
        np.ones((4, 3, 2)),
        np.full((4, 2), np.nan),
        np.full((4, 2), 1.5),
        np.ones((4, 2, 3, 1)),
    ],
)
def test_expected_response_matrix_rejects_invalid_model_output(
    probabilities: np.ndarray,
) -> None:
    class InvalidModel:
        def probability(self, theta: np.ndarray) -> np.ndarray:
            return probabilities

    with pytest.raises(ValueError):
        _expected_response_matrix(InvalidModel(), np.zeros((4, 1)), n_items=2)
