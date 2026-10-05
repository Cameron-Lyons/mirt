"""Assembly accepts verified feasible incumbents when a solver limit is hit."""

from itertools import combinations

import numpy as np
import pytest
import scipy.optimize
from scipy.optimize import OptimizeResult

from mirt.cat import (
    ContentArea,
    ContentBlueprint,
    assemble_form,
    assemble_parallel_forms,
)
from mirt.models import TwoParameterLogistic

_LIMIT_MESSAGE = "Time limit reached. (HiGHS Status 13: Time limit reached)"


def _pool(n_items: int = 60, seed: int = 0) -> TwoParameterLogistic:
    rng = np.random.default_rng(seed)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    return model


def _blueprint(n_items: int) -> ContentBlueprint:
    half = n_items // 2
    return ContentBlueprint(
        [
            ContentArea("A", items=set(range(half)), min_items=4, max_items=4),
            ContentArea("B", items=set(range(half, n_items)), min_items=4, max_items=6),
        ]
    )


def _patch_milp(monkeypatch, *, status=1, x_transform=None, gap=0.05):
    """Report every solve as stopped by a limit, optionally altering ``x``."""
    original = scipy.optimize.milp

    def limited(*args, **kwargs):
        solved = original(*args, **kwargs)
        x = solved.x if x_transform is None else x_transform(solved.x.copy())
        return OptimizeResult(
            x=x,
            fun=solved.fun,
            status=status,
            success=False,
            message=_LIMIT_MESSAGE,
            mip_gap=gap,
        )

    monkeypatch.setattr(scipy.optimize, "milp", limited)


def test_real_solver_limit_returns_constraint_satisfying_form():
    model = _pool(200)
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items=set(range(100)), min_items=8, max_items=10),
            ContentArea("B", items=set(range(100, 200)), min_items=8, max_items=12),
        ]
    )
    enemies = {(0, 1), (100, 101), (5, 150)}
    bundles = [{10, 11}, {120, 121, 122}]

    result = assemble_form(
        model,
        20,
        np.linspace(-2.0, 2.0, 11),
        target_information=5.0,
        blueprint=blueprint,
        enemy_pairs=enemies,
        item_bundles=bundles,
        solver_options={"node_limit": 1},
    )

    selected = set(result.selected_items.tolist())
    assert result.n_items == 20
    assert 8 <= result.content_counts["A"] <= 10
    assert 8 <= result.content_counts["B"] <= 12
    assert not any(
        first in selected and second in selected for first, second in enemies
    )
    assert all(bundle <= selected or not bundle & selected for bundle in bundles)
    if not result.is_optimal:
        assert result.mip_gap is not None and np.isfinite(result.mip_gap)
        assert "limit reached" in result.summary()


def test_limit_incumbent_is_returned_instead_of_raising(monkeypatch):
    model = _pool()
    expected = assemble_form(model, 8, [0.0], blueprint=_blueprint(60))
    _patch_milp(monkeypatch)

    result = assemble_form(model, 8, [0.0], blueprint=_blueprint(60))

    np.testing.assert_array_equal(result.selected_items, expected.selected_items)
    assert expected.is_optimal is True
    assert result.is_optimal is False
    assert result.mip_gap == pytest.approx(0.05)
    assert result.solver_message == _LIMIT_MESSAGE
    assert "Solver: limit reached" in result.summary()
    assert "MIP gap 0.05" in result.summary()


def test_optimal_summary_reports_proven_optimality():
    result = assemble_form(_pool(), 8, [0.0])

    assert result.is_optimal is True
    assert "Solver: optimal" in result.summary()


def test_require_optimal_restores_strict_failure(monkeypatch):
    model = _pool()
    _patch_milp(monkeypatch)

    with pytest.raises(RuntimeError, match=r"form assembly failed: Time limit"):
        assemble_form(model, 8, [0.0], require_optimal=True)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_require_optimal_must_be_boolean(value):
    with pytest.raises(TypeError, match="require_optimal"):
        assemble_form(_pool(), 8, [0.0], require_optimal=value)


def test_limit_without_incumbent_still_raises(monkeypatch):
    model = _pool()
    original = scipy.optimize.milp

    def no_incumbent(*args, **kwargs):
        original(*args, **kwargs)
        return OptimizeResult(
            x=None, fun=None, status=1, success=False, message=_LIMIT_MESSAGE
        )

    monkeypatch.setattr(scipy.optimize, "milp", no_incumbent)

    with pytest.raises(RuntimeError, match="form assembly failed: Time limit"):
        assemble_form(model, 8, [0.0])


def test_incumbent_violating_content_bounds_is_rejected(monkeypatch):
    model = _pool()

    def move_area_a_item_to_b(x):
        selected = np.flatnonzero(x[:60] > 0.5)
        unselected_b = [item for item in range(30, 60) if x[item] < 0.5]
        x[selected[selected < 30][0]] = 0.0
        x[unselected_b[0]] = 1.0
        return x

    _patch_milp(monkeypatch, x_transform=move_area_a_item_to_b)

    with pytest.raises(RuntimeError, match="violates the constraints"):
        assemble_form(model, 8, [0.0], blueprint=_blueprint(60))


def test_incumbent_with_wrong_form_size_is_rejected(monkeypatch):
    model = _pool()

    def add_item(x):
        x[int(np.flatnonzero(x[:60] < 0.5)[0])] = 1.0
        return x

    _patch_milp(monkeypatch, x_transform=add_item)

    with pytest.raises(RuntimeError, match="violates the constraints"):
        assemble_form(model, 8, [0.0])


def test_incumbent_within_solver_tolerance_is_rounded():
    model = _pool()
    reference = assemble_form(model, 8, [0.0], target_information=3.0)
    original = scipy.optimize.milp

    def fuzzy(*args, **kwargs):
        solved = original(*args, **kwargs)
        x = solved.x.copy()
        selected = x[:60] > 0.5
        x[:60] = np.where(selected, 1.0 - 4e-7, 3e-7)
        return OptimizeResult(
            x=x, fun=solved.fun, status=1, success=False, message=_LIMIT_MESSAGE
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scipy.optimize, "milp", fuzzy)
        result = assemble_form(model, 8, [0.0], target_information=3.0)

    np.testing.assert_array_equal(result.selected_items, reference.selected_items)
    assert result.objective_value == pytest.approx(reference.objective_value)
    assert result.mip_gap is None


@pytest.mark.parametrize("target", [None, 4.0])
def test_parallel_limit_incumbent_is_returned(monkeypatch, target):
    model = _pool()
    options = {
        "target_information": target,
        "blueprint": _blueprint(60),
        "max_pairwise_overlap": 1,
        "max_item_usage": 2,
    }
    expected = assemble_parallel_forms(model, 3, 8, [-1.0, 0.0, 1.0], **options)
    _patch_milp(monkeypatch, gap=0.2)

    result = assemble_parallel_forms(model, 3, 8, [-1.0, 0.0, 1.0], **options)

    for form, reference in zip(result.forms, expected.forms, strict=True):
        np.testing.assert_array_equal(form.selected_items, reference.selected_items)
        assert form.is_optimal is False
        assert form.mip_gap == pytest.approx(0.2)
    assert result.is_optimal is False
    assert result.mip_gap == pytest.approx(0.2)
    assert "limit reached" in result.summary()
    np.testing.assert_array_equal(result.overlap_matrix, expected.overlap_matrix)


def test_parallel_incumbent_violating_overlap_is_rejected(monkeypatch):
    model = _pool()

    def copy_first_form_to_second(x):
        x[60:120] = x[:60]
        return x

    _patch_milp(monkeypatch, x_transform=copy_first_form_to_second)

    with pytest.raises(RuntimeError, match="parallel form assembly failed: the"):
        assemble_parallel_forms(
            model, 2, 8, [0.0], max_item_usage=2, max_pairwise_overlap=2
        )


def test_parallel_require_optimal_restores_strict_failure(monkeypatch):
    model = _pool()
    _patch_milp(monkeypatch)

    with pytest.raises(RuntimeError, match="parallel form assembly failed: Time"):
        assemble_parallel_forms(model, 2, 8, [0.0], require_optimal=True)


def test_parallel_real_solver_limit_satisfies_joint_constraints():
    model = _pool(200)

    result = assemble_parallel_forms(
        model,
        3,
        20,
        np.linspace(-2.0, 2.0, 11),
        target_information=5.0,
        max_item_usage=2,
        max_pairwise_overlap=3,
        solver_options={"node_limit": 1},
    )

    sets = [set(form.selected_items.tolist()) for form in result.forms]
    assert all(len(items) == 20 for items in sets)
    assert all(len(a & b) <= 3 for a, b in combinations(sets, 2))
    assert max(result.item_usage.values()) <= 2
    if not result.is_optimal:
        assert result.mip_gap is not None and np.isfinite(result.mip_gap)
