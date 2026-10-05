"""Likelihood-ratio DIF through nested multiple-group fits."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest

from mirt.multigroup import (
    MultigroupEMEstimator,
    fit_multigroup,
    invariance_lrt,
    multigroup_dif,
    select_dif_anchors,
)
from mirt.multigroup import dif as dif_module
from mirt.multigroup.dif import _DIFTestRow, _DIFTestTable, _run_multigroup_dif
from mirt.multigroup.invariance import InvarianceSpec

FIT = {"n_quadpts": 9, "tol": 1e-3}


def _simulate(
    seed: int,
    *,
    n_per_group: int = 250,
    n_items: int = 8,
    shift: float = 1.2,
    impact: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """2PL responses with uniform DIF on item 0 and a lower focal mean."""
    rng = np.random.default_rng(seed)
    discrimination = np.linspace(0.9, 1.8, n_items)
    difficulty = np.linspace(-1.2, 1.0, n_items)
    focal_difficulty = difficulty.copy()
    focal_difficulty[0] += shift

    def responses(theta: np.ndarray, locations: np.ndarray) -> np.ndarray:
        logits = discrimination * (theta[:, None] - locations)
        return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

    data = np.vstack(
        [
            responses(rng.normal(0.0, 1.0, n_per_group), difficulty),
            responses(rng.normal(impact, 1.0, n_per_group), focal_difficulty),
        ]
    )
    return data, np.repeat([0, 1], n_per_group)


def _rows(table: _DIFTestTable) -> dict[int, _DIFTestRow]:
    return {row.item: row for row in table.rows}


@pytest.fixture(scope="module")
def dif_data() -> tuple[np.ndarray, np.ndarray]:
    return _simulate(1)


@pytest.fixture(scope="module")
def drop_table(dif_data: tuple[np.ndarray, np.ndarray]) -> _DIFTestTable:
    data, groups = dif_data
    return _run_multigroup_dif(data, groups, **FIT)


class TestDropScheme:
    def test_flags_only_the_dif_item_despite_impact(self, drop_table) -> None:
        rows = _rows(drop_table)

        assert drop_table.flagged() == [0]
        assert rows[0].p_value < 1e-4
        assert all(row.df == 2.0 for row in drop_table.rows)
        assert all(row.converged and row.round == 1 for row in drop_table.rows)

    def test_rows_match_cold_nested_fits(self, dif_data, drop_table) -> None:
        data, groups = dif_data
        constrained = fit_multigroup(data, groups, invariance="strict", **FIT)
        rows = _rows(drop_table)

        for item in (0, 1):
            free = fit_multigroup(
                data,
                groups,
                invariance=InvarianceSpec(
                    "strict", free_discrimination=[item], free_intercepts=[item]
                ),
                **FIT,
            )
            expected = invariance_lrt(constrained, free)
            assert rows[item].chi2 == pytest.approx(expected["chi2"], abs=0.05)
            assert rows[item].df == expected["df"]
            assert rows[item].delta_aic == pytest.approx(
                constrained.aic - free.aic, abs=0.05
            )

    def test_group_parameters_come_from_the_free_model(self, drop_table) -> None:
        reference, focal = _rows(drop_table)[0].group_parameters

        assert focal["difficulty"] - reference["difficulty"] == pytest.approx(
            1.2, abs=0.4
        )
        assert (
            _rows(drop_table)[3].group_parameters[0]
            != _rows(drop_table)[3].group_parameters[1]
        )

    def test_holm_adjustment_spans_the_tested_items(self, drop_table) -> None:
        from mirt.diagnostics.multiple_testing import adjust_p_values

        raw = np.array([row.p_value for row in drop_table.rows])
        adjusted = np.array([row.p_value_adjusted for row in drop_table.rows])

        np.testing.assert_allclose(adjusted, adjust_p_values(raw, "holm"))

    def test_dataframe_reports_one_row_per_tested_item(self, drop_table) -> None:
        frame = drop_table.to_dataframe()

        assert list(frame.columns) == [
            "item",
            "chi2",
            "df",
            "p_value",
            "p_value_adjusted",
            "delta_aic",
            "delta_bic",
            "flagged",
            "converged",
            "round",
        ]
        assert list(frame["item"]) == [f"Item_{index}" for index in range(8)]
        assert list(frame["flagged"]) == [True] + [False] * 7


def test_public_function_tests_selected_items(dif_data, drop_table) -> None:
    data, groups = dif_data

    frame = multigroup_dif(data, groups, items=["Item_1", 0], **FIT)

    assert list(frame["item"]) == ["Item_0", "Item_1"]
    rows = _rows(drop_table)
    np.testing.assert_allclose(
        list(frame["chi2"]), [rows[0].chi2, rows[1].chi2], atol=1e-6
    )


def test_add_scheme_constrains_one_studied_item_at_a_time(dif_data) -> None:
    data, groups = dif_data

    table = _run_multigroup_dif(data, groups, scheme="add", anchors=[5, 6, 7], **FIT)

    assert [row.item for row in table.rows] == [0, 1, 2, 3, 4]
    assert table.flagged() == [0]
    assert all(row.df == 2.0 for row in table.rows)
    # The anchored baseline frees the studied items, so their group-specific
    # estimates come from it.
    reference, focal = _rows(table)[0].group_parameters
    assert focal["difficulty"] > reference["difficulty"]


def test_drop_sequential_retests_with_flagged_items_free(dif_data) -> None:
    data, groups = dif_data

    table = _run_multigroup_dif(data, groups, scheme="drop_sequential", **FIT)
    rows = _rows(table)

    assert table.flagged() == [0]
    assert rows[0].round == 1
    assert all(rows[item].round == 2 for item in range(1, 8))


def test_initial_latent_warm_start_resumes_from_a_nested_fit(dif_data) -> None:
    data, groups = dif_data
    responses = [data[groups == group] for group in (0, 1)]
    constrained = fit_multigroup(data, groups, invariance="strict", **FIT)
    spec = InvarianceSpec("strict", free_discrimination=[0], free_intercepts=[0])
    estimator = MultigroupEMEstimator(**FIT)

    warm = estimator.fit(
        copy.deepcopy(constrained.model),
        responses,
        invariance=spec,
        initial_latent=constrained.latent_distributions,
    )
    cold = fit_multigroup(data, groups, invariance=spec, **FIT)

    # The constrained optimum is feasible in the freer model, so EM starts there.
    assert estimator.convergence_history[0] == pytest.approx(
        constrained.log_likelihood, abs=1e-6
    )
    assert warm.log_likelihood == pytest.approx(cold.log_likelihood, abs=0.02)
    assert warm.latent_distributions[0].mean[0] == 0.0
    with pytest.raises(ValueError, match="one distribution per group"):
        MultigroupEMEstimator(**FIT).fit(
            copy.deepcopy(constrained.model),
            responses,
            invariance=spec,
            initial_latent=constrained.latent_distributions[:1],
        )


def test_sequential_scheme_warns_when_rounds_run_out(dif_data) -> None:
    data, groups = dif_data

    with pytest.warns(UserWarning, match="did not settle within 1 rounds"):
        table = _run_multigroup_dif(
            data, groups, items=[0, 1], scheme="drop_sequential", max_rounds=1, **FIT
        )

    assert table.flagged() == [0]
    assert all(row.round == 1 for row in table.rows)


def test_add_sequential_moves_invariant_items_into_the_anchor_set(dif_data) -> None:
    data, groups = dif_data

    table = _run_multigroup_dif(
        data, groups, scheme="add_sequential", anchors=[5, 6, 7], **FIT
    )
    rows = _rows(table)

    assert table.flagged() == [0]
    assert rows[0].round == 2
    assert all(rows[item].round == 1 for item in (1, 2, 3, 4))


@pytest.mark.parametrize(
    ("model", "parameters", "expected_df"),
    [
        ("2PL", ("intercepts",), 1.0),
        ("1PL", ("discrimination", "intercepts"), 1.0),
        ("3PL", ("discrimination", "intercepts"), 2.0),
    ],
)
def test_df_counts_only_freed_free_parameters(
    dif_data, model: str, parameters: tuple[str, ...], expected_df: float
) -> None:
    data, groups = dif_data

    table = _run_multigroup_dif(
        data,
        groups,
        model,
        items=[1],
        parameters=parameters,
        n_quadpts=7,
        max_iter=2,
    )

    assert table.rows[0].df == expected_df


def test_failed_refit_reports_missing_statistics(
    dif_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, groups = dif_data
    original_fit = dif_module._fit

    def failing_fit(start_model: Any, responses: Any, free_items: Any, *args: Any):
        if tuple(free_items) == (1,):
            raise RuntimeError("synthetic divergence")
        return original_fit(start_model, responses, free_items, *args)

    monkeypatch.setattr(dif_module, "_fit", failing_fit)

    table = _run_multigroup_dif(data, groups, items=[0, 1], n_quadpts=7, max_iter=5)
    failed = _rows(table)[1]

    assert np.isnan(failed.chi2) and np.isnan(failed.p_value)
    assert np.isnan(failed.p_value_adjusted)
    assert not failed.converged
    assert failed.group_parameters is None
    assert 1 not in table.flagged()
    assert np.isfinite(_rows(table)[0].chi2)


def test_refits_run_through_the_parallel_task_runner(
    dif_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, groups = dif_data
    calls: list[tuple[int, int]] = []

    def run_inline(function: Any, tasks: list[Any], n_jobs: int) -> list[Any]:
        calls.append((len(tasks), n_jobs))
        pickle.loads(pickle.dumps(tasks[0]))
        return [function(task) for task in tasks]

    monkeypatch.setattr(dif_module, "_run_bootstrap_tasks", run_inline)

    _run_multigroup_dif(data, groups, items=[0, 2], n_jobs=3, n_quadpts=7, max_iter=3)

    assert calls == [(2, 3)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": "forward"}, "scheme"),
        ({"parameters": ("guessing",)}, "parameters"),
        ({"parameters": ()}, "parameters"),
        ({"p_adjust": "sidak"}, "p_adjust"),
        ({"alpha": 1.5}, "alpha"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"items": [9]}, "out of range"),
        ({"items": [1, 1]}, "duplicate"),
        ({"items": ["missing"]}, "unknown item name"),
        ({"items": [0.5]}, "item indices or names"),
        ({"items": [0, 1], "anchors": [1]}, "overlap"),
        ({"anchors": list(range(8))}, "at least one item"),
        ({"scheme": "add"}, "requires at least one anchor"),
        ({"reference_group": 5}, "reference_group"),
    ],
)
def test_options_are_validated_before_fitting(
    dif_data, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], message: str
) -> None:
    def unexpected_fit(*args: Any, **fit_kwargs: Any) -> None:
        pytest.fail("no model should be fitted for invalid options")

    monkeypatch.setattr(dif_module, "_fit", unexpected_fit)
    data, groups = dif_data
    with pytest.raises(ValueError, match=message):
        _run_multigroup_dif(data, groups, **kwargs)


def test_single_group_is_rejected(dif_data) -> None:
    data, _ = dif_data
    with pytest.raises(ValueError, match="At least 2 groups"):
        multigroup_dif(data, np.zeros(data.shape[0]))


def _fake_table(p_values: list[float], flagged: list[bool]) -> _DIFTestTable:
    rows = [
        _DIFTestRow(
            item=item,
            chi2=1.0 - p_value,
            df=2.0,
            p_value=p_value,
            p_value_adjusted=0.01 if is_flagged else 0.5,
            delta_aic=0.0,
            delta_bic=0.0,
            converged=True,
            round=1,
            group_parameters=None,
        )
        for item, (p_value, is_flagged) in enumerate(
            zip(p_values, flagged, strict=True)
        )
    ]
    return _DIFTestTable(
        rows=rows,
        item_names=[f"Item_{item}" for item in range(len(rows))],
        group_labels=["0", "1"],
        reference_group=0,
        n_categories=None,
        alpha=0.05,
        p_adjust="holm",
    )


@pytest.mark.parametrize(
    ("method", "n_anchors", "expected_scheme", "expected"),
    [
        ("rank", 2, "drop", [0, 2]),
        ("rank", None, "drop", [1, 2, 3]),
        ("aoaa_iterative", 2, "drop_sequential", [2, 3]),
        ("aoaa_iterative", 5, "drop_sequential", [1, 2, 3]),
    ],
)
def test_anchor_selection_ranks_unflagged_items(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    n_anchors: int | None,
    expected_scheme: str,
    expected: list[int],
) -> None:
    schemes: list[str] = []

    def fake_run(*args: Any, scheme: str, **kwargs: Any) -> _DIFTestTable:
        schemes.append(scheme)
        return _fake_table([0.9, 0.4, 0.8, 0.7], [True, False, False, False])

    monkeypatch.setattr(dif_module, "_run_multigroup_dif", fake_run)

    selected = select_dif_anchors(
        np.zeros((4, 4)), np.repeat([0, 1], 2), method=method, n_anchors=n_anchors
    )

    assert selected == expected
    assert schemes == [expected_scheme]


def test_anchor_selection_rank_may_return_flagged_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dif_module,
        "_run_multigroup_dif",
        lambda *args, **kwargs: _fake_table([0.9, 0.4], [True, False]),
    )

    assert select_dif_anchors(
        np.zeros((4, 2)), np.repeat([0, 1], 2), method="rank", n_anchors=1
    ) == [0]


def test_anchor_selection_requires_a_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dif_module,
        "_run_multigroup_dif",
        lambda *args, **kwargs: _fake_table([0.01, 0.02], [True, True]),
    )

    with pytest.raises(ValueError, match="no item qualified"):
        select_dif_anchors(np.zeros((4, 2)), np.repeat([0, 1], 2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"method": "forward"}, "method"), ({"n_anchors": 0}, "n_anchors")],
)
def test_anchor_selection_validates_options(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_dif_anchors(np.zeros((4, 2)), np.repeat([0, 1], 2), **kwargs)


@pytest.mark.slow
def test_aoaa_anchor_selection_excludes_the_dif_item(dif_data) -> None:
    data, groups = dif_data

    anchors = select_dif_anchors(data, groups, n_anchors=None, **FIT)

    assert anchors == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.slow
def test_impact_only_type_i_error_and_power() -> None:
    """Holm-adjusted false positives stay rare while large DIF is detected."""
    false_positives = 0
    for seed in range(5):
        data, groups = _simulate(100 + seed, n_per_group=500, shift=0.0)
        false_positives += len(_run_multigroup_dif(data, groups, **FIT).flagged())

        data, groups = _simulate(200 + seed, n_per_group=500)
        alternative = _run_multigroup_dif(data, groups, **FIT)
        assert _rows(alternative)[0].p_value < 0.01

    assert false_positives <= 1
