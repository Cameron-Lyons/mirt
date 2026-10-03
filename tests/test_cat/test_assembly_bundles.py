"""Assembly contracts checked against exhaustive feasible-form enumeration."""

import tracemalloc
from itertools import combinations, product

import numpy as np
import pytest
from scipy.special import expit

from mirt.cat import (
    ContentArea,
    ContentBlueprint,
    assemble_form,
    assemble_parallel_forms,
)
from mirt.models import TwoParameterLogistic
from mirt.models.custom import CustomItemModel, create_item_type


class InformationPool:
    n_factors = 1

    def __init__(self, information):
        self.curves = np.asarray(information, dtype=float)
        self.n_items = self.curves.shape[1]
        self.evaluated = []

    def information(self, theta, item_idx=None):
        self.evaluated.append(item_idx)
        return self.curves.copy() if item_idx is None else self.curves[:, item_idx]


@pytest.mark.parametrize("target", [None, np.array([6.2, 5.0, 7.3])])
def test_bundle_content_security_and_budget_joint_optimum(target):
    rng = np.random.default_rng(905)
    curves = rng.uniform(0.1, 4.0, size=(3, 8))
    pool = InformationPool(curves)
    weights = np.array([0.2, 0.3, 0.5])
    costs = np.array([1, 1, 2, 2, 1, 3, 1, 2])
    bundles = [{0, 1}, {1, 2}, {3, 4}]
    core = {0, 3, 5, 6}
    blueprint = ContentBlueprint([ContentArea("core", core, min_items=1, max_items=2)])

    feasible = []
    for form in combinations(range(8), 3):
        selected = set(form)
        if any(
            bool(selected & bundle) and not bundle <= selected for bundle in bundles
        ):
            continue
        if {2, 5} <= selected or costs[list(form)].sum() > 5:
            continue
        if not 1 <= len(core & selected) <= 2:
            continue
        information = curves[:, form].sum(axis=1)
        value = weights @ (information if target is None else abs(information - target))
        feasible.append((value, form))
    expected = (max if target is None else min)(feasible)

    result = assemble_form(
        pool,
        3,
        [-1.0, 0.0, 1.0],
        theta_weights=weights,
        target_information=target,
        item_bundles=bundles,
        blueprint=blueprint,
        enemy_pairs=[(2, 5)],
        item_costs=costs,
        max_cost=5,
    )

    assert tuple(result.selected_items) == expected[1]
    assert result.objective_value == pytest.approx(expected[0], abs=1e-10)
    np.testing.assert_allclose(result.information, curves[:, expected[1]].sum(axis=1))


@pytest.mark.parametrize("use_target", [False, True])
def test_parallel_bundle_reuse_and_overlap_match_exhaustive_search(use_target):
    curves = np.random.default_rng(402).uniform(0.1, 5.0, size=(2, 7))
    weights = np.array([0.4, 0.6])
    bundles = [{0, 1}, {2, 3}]
    targets = np.array([[5.0, 6.0], [7.0, 4.0], [6.5, 5.5]])
    feasible_forms = [
        form
        for form in combinations(range(7), 3)
        if all(not set(form) & bundle or bundle <= set(form) for bundle in bundles)
    ]
    objectives = []
    for forms in product(feasible_forms, repeat=3):
        if any(sum(item in form for form in forms) > 2 for item in range(7)):
            continue
        if any(
            len(set(first) & set(second)) > 1
            for first, second in combinations(forms, 2)
        ):
            continue
        information = np.stack([curves[:, form].sum(axis=1) for form in forms])
        objectives.append(
            np.mean(abs(information - targets) @ weights)
            if use_target
            else np.min(information @ weights)
        )

    result = assemble_parallel_forms(
        InformationPool(curves),
        3,
        3,
        [-1.0, 1.0],
        item_bundles=bundles,
        theta_weights=weights,
        target_information=targets if use_target else None,
        max_item_usage=2,
        max_pairwise_overlap=1,
    )

    expected = (min if use_target else max)(objectives)
    assert result.objective_value == pytest.approx(expected, abs=1e-10)
    for form in result.forms:
        selected = set(form.selected_items)
        assert all(not selected & bundle or bundle <= selected for bundle in bundles)
    assert max(result.item_usage.values()) <= 2
    assert np.max(result.overlap_matrix[np.triu_indices(3, 1)]) <= 1


def test_required_overlapping_bundles_become_shared_anchors():
    pool = InformationPool([[1, 2, 3, 9, 8, 7]])
    result = assemble_parallel_forms(
        pool,
        3,
        4,
        [0.0],
        required_items=[0],
        item_bundles=[{0, 1}, {1, 2}],
        max_pairwise_overlap=3,
    )

    assert {item: result.item_usage[item] for item in (0, 1, 2)} == {0: 3, 1: 3, 2: 3}
    assert all({0, 1, 2} <= set(form.selected_items) for form in result.forms)
    np.testing.assert_array_equal(result.overlap_matrix, np.full((3, 3), 3) + np.eye(3))


@pytest.mark.parametrize("parallel", [False, True])
def test_unavailable_bundle_member_removes_entire_connected_bundle(parallel):
    pool = InformationPool([[100, 100, 100, 4, 3, 2, 1]])
    kwargs = dict(item_bundles=[{0, 1}, {1, 2}], excluded_items={2})
    if parallel:
        result = assemble_parallel_forms(pool, 2, 2, [0.0], **kwargs)
        assert set(result.item_usage) == {3, 4, 5, 6}
    else:
        result = assemble_form(pool, 2, [0.0], **kwargs)
        np.testing.assert_array_equal(result.selected_items, [3, 4])
    assert pool.evaluated == [None]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"item_bundles": [[0]]}, "at least two"),
        ({"item_bundles": [[0, 0]]}, "duplicate"),
        ({"item_bundles": [[0, 9]]}, "outside"),
        ({"item_bundles": [[0, True]]}, "integer"),
        ({"item_bundles": "01"}, "collections"),
        ({"item_bundles": 1}, "collections"),
        (
            {"item_bundles": [[0, 1]], "required_items": [0], "excluded_items": [1]},
            "unavailable",
        ),
        ({"item_bundles": [[0, 1, 2]], "required_items": [0]}, "smaller"),
    ],
)
def test_rejects_invalid_or_impossible_bundle_configuration(parallel, kwargs, message):
    pool = InformationPool([[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match=message):
        if parallel:
            assemble_parallel_forms(pool, 2, 2, [0.0], **kwargs)
        else:
            assemble_form(pool, 2, [0.0], **kwargs)


def test_bundle_and_enemy_constraint_conflict_reports_infeasibility():
    with pytest.raises(RuntimeError, match="form assembly failed"):
        assemble_form(
            InformationPool([[1, 2, 3, 4]]),
            2,
            [0.0],
            item_bundles=[{0, 1}],
            required_items=[0],
            enemy_pairs=[(0, 1)],
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_large_finite_weights_preserve_analytic_logistic_objective(parallel):
    model = TwoParameterLogistic(6)
    a = np.array([0.5, 0.8, 1.3, 2.0, 1.1, 0.9])
    b = np.array([-1, 0, 1, 0.5, -0.5, 2])
    model.set_parameters(discrimination=a, difficulty=b)
    theta = np.array([-1.0, 0.0, 1.0])
    p = expit(a * (theta[:, None] - b))
    information = a**2 * p * (1 - p)
    weights = np.array([0.25, 0.5, 0.25])
    large_weights = np.array([8e307, 1.6e308, 8e307])
    with np.errstate(over="raise", invalid="raise"):
        if parallel:
            result = assemble_parallel_forms(
                model, 2, 2, theta, theta_weights=large_weights
            )
            value = min(
                information[:, form.selected_items].sum(axis=1) @ weights
                for form in result.forms
            )
        else:
            result = assemble_form(model, 2, theta, theta_weights=large_weights)
            value = information[:, result.selected_items].sum(axis=1) @ weights
    assert result.objective_value == pytest.approx(value)


@pytest.mark.parametrize("parallel", [False, True])
def test_dense_candidate_subset_keeps_vectorized_information(parallel):
    pool = InformationPool(np.ones((3, 1_000)))
    kwargs = dict(excluded_items=[999])
    if parallel:
        assemble_parallel_forms(pool, 2, 2, [-1, 0, 1], **kwargs)
    else:
        assemble_form(pool, 2, [-1, 0, 1], **kwargs)
    assert pool.evaluated == [None]


@pytest.mark.parametrize("parallel", [False, True])
def test_sparse_custom_batch_information_remains_authoritative(parallel):
    calls = []

    def batch_information(theta, a):
        calls.append(a.copy())
        p = expit(theta[:, None] * a)
        return a**2 * p * (1 - p)

    item_type = create_item_type(
        "steep",
        lambda theta, a: expit(theta * a),
        par_defaults={"a": 1e6},
        batch_info_function=batch_information,
    )
    model = CustomItemModel(6, item_type=item_type)
    kwargs = dict(candidate_items={0, 1, 2, 3})
    if parallel:
        result = assemble_parallel_forms(model, 2, 2, [0.0], **kwargs)
        assert all(form.objective_value == pytest.approx(5e11) for form in result.forms)
    else:
        result = assemble_form(model, 2, [0.0], **kwargs)
        assert result.objective_value == pytest.approx(5e11)
    assert len(calls) == 1


@pytest.mark.parametrize("parallel", [False, True])
def test_itemwise_fallback_protects_reused_callback_buffer(parallel):
    class ScratchPool(InformationPool):
        def __init__(self):
            super().__init__([[1, 9, 4, 2], [2, 8, 3, 1]])
            self.scratch = np.empty(2)

        def information(self, theta, item_idx=None):
            if item_idx is None:
                return self.curves.sum(axis=1)
            self.scratch[:] = self.curves[:, item_idx]
            return self.scratch

    pool = ScratchPool()
    if parallel:
        result = assemble_parallel_forms(pool, 2, 1, [-1, 1], candidate_items={0, 1, 2})
        assert set(result.item_usage) == {1, 2}
        for form in result.forms:
            np.testing.assert_array_equal(
                form.information, pool.curves[:, form.selected_items[0]]
            )
    else:
        result = assemble_form(pool, 1, [-1, 1], candidate_items={0, 1, 2})
        np.testing.assert_array_equal(result.selected_items, [1])
        np.testing.assert_array_equal(result.information, [9, 8])


def test_one_candidate_does_not_confuse_total_information_with_item_information():
    class TotalColumnPool(InformationPool):
        def information(self, theta, item_idx=None):
            if item_idx is None:
                return self.curves.sum(axis=1, keepdims=True)
            return self.curves[:, item_idx]

    pool = TotalColumnPool([[1, 9, 4, 2], [2, 8, 3, 1]])
    result = assemble_form(pool, 1, [-1, 1], candidate_items={2})
    np.testing.assert_array_equal(result.selected_items, [2])
    np.testing.assert_array_equal(result.information, [4, 3])


def test_required_bundle_overlap_limit_includes_every_member():
    pool = InformationPool([[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match="smaller than required_items"):
        assemble_parallel_forms(
            pool,
            3,
            4,
            [0.0],
            required_items={0},
            item_bundles=[{0, 1}, {1, 2}],
            max_pairwise_overlap=2,
        )


def test_parallel_forms_can_consist_entirely_of_required_bundle():
    result = assemble_parallel_forms(
        InformationPool([[1, 2, 3, 4]]),
        3,
        3,
        [0.0],
        required_items={0},
        item_bundles=[{0, 1}, {1, 2}],
        max_pairwise_overlap=3,
    )
    assert result.item_usage == {0: 3, 1: 3, 2: 3}
    np.testing.assert_array_equal(result.overlap_matrix, np.full((3, 3), 3))


@pytest.mark.performance
def test_sparse_logistic_pool_assembly_bounds_traced_allocations():
    # Initialize the large pool and warm the optimizer before measuring scratch
    # space. A full theta-by-pool curve matrix alone would consume 13.2 MiB.
    model = TwoParameterLogistic(50_000)
    theta = np.linspace(-3, 3, 33)
    candidates = np.arange(0, 40_000, 1_000).tolist()
    assemble_form(model, 4, theta, candidate_items=candidates)
    tracemalloc.start()
    try:
        result = assemble_form(model, 4, theta, candidate_items=candidates)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.n_items == 4
    assert peak < 2 * 1024 * 1024, f"candidate scratch space used {peak:,} bytes"
