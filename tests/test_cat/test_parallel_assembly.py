"""Tests for exact joint parallel-form assembly."""

from itertools import combinations

import numpy as np
import pytest

from mirt.cat import (
    ContentArea,
    ContentBlueprint,
    ParallelFormAssemblyResult,
    assemble_parallel_forms,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


class ProfileInformationModel:
    """Small model exposing deterministic item-information profiles."""

    def __init__(self, profiles: np.ndarray, *, n_factors: int = 1) -> None:
        self.profiles = np.asarray(profiles, dtype=np.float64)
        self.n_items = self.profiles.shape[1]
        self.n_factors = n_factors
        self.calls = 0

    def information(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        self.calls += 1
        if len(theta) != len(self.profiles):
            raise ValueError("theta length does not match the profile grid")
        if item_idx is None:
            return self.profiles.copy()
        return self.profiles[:, item_idx].copy()


class ItemwiseInformationModel(ProfileInformationModel):
    """Model requiring itemwise information evaluation."""

    def information(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        self.calls += 1
        if item_idx is None:
            return np.sum(self.profiles, axis=1)
        return self.profiles[:, item_idx].copy()


@pytest.fixture
def theta() -> np.ndarray:
    return np.array([-1.0, 0.0, 1.0])


@pytest.fixture
def profiles() -> np.ndarray:
    return np.array(
        [
            [8.0, 1.0, 2.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [2.0, 8.0, 2.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [1.0, 1.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        ]
    )


def _selected_sets(result: ParallelFormAssemblyResult) -> list[set[int]]:
    return [set(form.selected_items.tolist()) for form in result.forms]


def test_max_min_objective_balances_strong_items() -> None:
    model = ProfileInformationModel(np.array([[10.0, 9.0, 1.0, 1.0]]))

    result = assemble_parallel_forms(model, 2, 2, theta=[0.0])

    assert result.method == "maximize_minimum_information"
    assert result.objective_value == pytest.approx(10.0)
    assert sorted(form.objective_value for form in result.forms) == [10.0, 11.0]
    assert all(len(items & {0, 1}) == 1 for items in _selected_sets(result))
    assert result.item_usage == {0: 1, 1: 1, 2: 1, 3: 1}
    np.testing.assert_array_equal(result.overlap_matrix, np.diag([2, 2]))
    assert model.calls == 1


def test_target_objective_matches_joint_brute_force(theta, profiles) -> None:
    model = ProfileInformationModel(profiles[:, :6])
    target = np.array([10.0, 10.0, 8.0])
    weights = np.array([0.2, 0.5, 0.3])

    result = assemble_parallel_forms(
        model,
        2,
        2,
        theta,
        theta_weights=weights,
        target_information=target,
    )

    reference = min(
        np.mean(
            [
                weights @ np.abs(np.sum(profiles[:, first], axis=1) - target),
                weights @ np.abs(np.sum(profiles[:, second], axis=1) - target),
            ]
        )
        for first in combinations(range(6), 2)
        for second in combinations(set(range(6)) - set(first), 2)
    )
    assert result.objective_value == pytest.approx(reference, abs=1e-10)
    assert result.method == "target_information"
    assert all(form.method == "target_information" for form in result.forms)
    assert _selected_sets(result)[0].isdisjoint(_selected_sets(result)[1])


def test_form_specific_targets_keep_form_labels() -> None:
    model = ProfileInformationModel(np.array([[1.0, 2.0, 3.0, 4.0]]))

    result = assemble_parallel_forms(
        model,
        2,
        1,
        theta=[0.0],
        target_information=np.array([[1.0], [4.0]]),
    )

    np.testing.assert_array_equal(result.forms[0].selected_items, np.array([0]))
    np.testing.assert_array_equal(result.forms[1].selected_items, np.array([3]))
    assert result.objective_value == pytest.approx(0.0, abs=1e-10)


def test_scalar_target_broadcasts_to_every_form() -> None:
    model = ProfileInformationModel(np.array([[1.0, 2.0, 3.0, 4.0]]))

    result = assemble_parallel_forms(
        model,
        2,
        1,
        theta=[0.0],
        target_information=3.5,
    )

    assert result.objective_value == pytest.approx(0.5)


def test_required_items_are_shared_anchors(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_parallel_forms(
        model,
        3,
        2,
        theta,
        required_items={7},
    )

    assert result.item_usage[7] == 3
    assert all(7 in items for items in _selected_sets(result))
    assert all(
        usage == 1 for item_idx, usage in result.item_usage.items() if item_idx != 7
    )
    expected_overlap = np.ones((3, 3), dtype=np.intp)
    np.fill_diagonal(expected_overlap, 2)
    np.testing.assert_array_equal(result.overlap_matrix, expected_overlap)


def test_max_item_usage_allows_controlled_reuse(theta, profiles) -> None:
    model = ProfileInformationModel(profiles[:, :4])

    result = assemble_parallel_forms(
        model,
        3,
        2,
        theta,
        max_item_usage=2,
    )

    assert sum(result.item_usage.values()) == 6
    assert max(result.item_usage.values()) <= 2


def test_pairwise_overlap_is_enforced(theta, profiles) -> None:
    model = ProfileInformationModel(profiles[:, :6])

    result = assemble_parallel_forms(
        model,
        3,
        2,
        theta,
        max_item_usage=3,
        max_pairwise_overlap=0,
    )

    np.testing.assert_array_equal(result.overlap_matrix, np.diag([2, 2, 2]))


def test_pairwise_overlap_counts_required_anchors(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_parallel_forms(
        model,
        3,
        2,
        theta,
        required_items={7},
        max_item_usage=3,
        max_pairwise_overlap=1,
    )

    off_diagonal = result.overlap_matrix[np.triu_indices(3, k=1)]
    np.testing.assert_array_equal(off_diagonal, np.ones(3, dtype=np.intp))


def test_applies_content_bounds_to_every_form(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items={0, 1, 2, 3}, min_items=1, max_items=1),
            ContentArea("B", items={4, 5, 6, 7}, min_items=1, max_items=1),
        ]
    )

    result = assemble_parallel_forms(
        model,
        3,
        2,
        theta,
        blueprint=blueprint,
    )

    assert all(form.content_counts == {"A": 1, "B": 1} for form in result.forms)


def test_enforces_enemy_pairs_within_each_form(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_parallel_forms(
        model,
        2,
        2,
        theta,
        enemy_pairs={(0, 3), (1, 4)},
    )

    for selected in _selected_sets(result):
        assert not {0, 3} <= selected
        assert not {1, 4} <= selected


def test_applies_cost_budget_to_every_form(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    costs = np.array([8.0, 8.0, 8.0, 5.0, 2.0, 1.0, 1.0, 1.0])

    result = assemble_parallel_forms(
        model,
        2,
        2,
        theta,
        item_costs=costs,
        max_cost=6.0,
    )

    assert all(form.total_cost is not None for form in result.forms)
    assert all(form.total_cost <= 6.0 for form in result.forms)  # type: ignore[operator]


def test_honors_candidate_and_excluded_items(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_parallel_forms(
        model,
        2,
        2,
        theta,
        candidate_items={0, 1, 2, 3, 4, 5},
        excluded_items={0, 1},
    )

    assert set(result.item_usage) <= {2, 3, 4, 5}


def test_information_is_evaluated_once_with_vectorized_model(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    assemble_parallel_forms(model, 2, 2, theta)

    assert model.calls == 1


def test_falls_back_to_itemwise_information(theta, profiles) -> None:
    model = ItemwiseInformationModel(profiles)

    assemble_parallel_forms(model, 2, 2, theta)

    assert model.calls == model.n_items + 1


def test_integrates_with_dichotomous_and_polytomous_models() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    dichotomous = TwoParameterLogistic(8)
    dichotomous.set_parameters(
        discrimination=np.linspace(0.6, 1.8, 8),
        difficulty=np.linspace(-1.2, 1.2, 8),
    )
    polytomous = GradedResponseModel(8, n_categories=3)
    polytomous.set_parameters(
        discrimination=np.linspace(0.6, 1.8, 8),
        thresholds=np.column_stack(
            (np.linspace(-1.4, -0.2, 8), np.linspace(0.2, 1.4, 8))
        ),
    )

    for model in (dichotomous, polytomous):
        result = assemble_parallel_forms(model, 2, 3, theta)

        assert all(form.n_items == 3 for form in result.forms)
        assert _selected_sets(result)[0].isdisjoint(_selected_sets(result)[1])
        assert all(np.all(form.information >= 0.0) for form in result.forms)


def test_summary_reports_joint_diagnostics(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_parallel_forms(model, 2, 2, theta)
    summary = result.summary()

    assert isinstance(result, ParallelFormAssemblyResult)
    assert "Forms: 2" in summary
    assert "Items per form: 2" in summary
    assert "minimum weighted information" in summary
    assert "Maximum pairwise overlap: 0" in summary


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_forms": 1}, "at least two"),
        ({"n_forms": 2.0}, "integer"),
        ({"n_forms": True}, "integer"),
        ({"max_item_usage": 0}, "positive"),
        ({"max_item_usage": 3}, "cannot exceed"),
        ({"max_item_usage": True}, "integer"),
        ({"max_pairwise_overlap": -1}, "non-negative"),
        ({"max_pairwise_overlap": 3}, "cannot exceed form_size"),
        ({"max_pairwise_overlap": True}, "integer"),
        ({"target_information": np.ones(2)}, "must be scalar or have shape"),
        ({"target_information": -1.0}, "non-negative"),
        ({"target_information": np.nan}, "finite"),
        ({"solver_options": []}, "mapping"),
    ],
)
def test_rejects_invalid_joint_configuration(
    theta,
    profiles,
    kwargs,
    message,
) -> None:
    model = ProfileInformationModel(profiles)
    arguments = {"n_forms": 2, "form_size": 2, "theta": theta}
    arguments.update(kwargs)

    error = TypeError if "mapping" in message else ValueError
    with pytest.raises(error, match=message):
        assemble_parallel_forms(model, **arguments)


def test_rejects_overlap_smaller_than_required_anchors(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    with pytest.raises(ValueError, match="smaller than required_items"):
        assemble_parallel_forms(
            model,
            2,
            2,
            theta,
            required_items={0, 1},
            max_pairwise_overlap=1,
        )


def test_rejects_insufficient_usage_capacity(theta, profiles) -> None:
    model = ProfileInformationModel(profiles[:, :5])

    with pytest.raises(ValueError, match="insufficient candidate capacity"):
        assemble_parallel_forms(model, 3, 2, theta)


def test_rejects_multidimensional_models(theta, profiles) -> None:
    model = ProfileInformationModel(profiles, n_factors=2)

    with pytest.raises(ValueError, match="unidimensional"):
        assemble_parallel_forms(model, 2, 2, theta)


def test_reports_jointly_infeasible_content_constraints(theta, profiles) -> None:
    model = ProfileInformationModel(profiles[:, :6])
    blueprint = ContentBlueprint(
        [ContentArea("Scarce", items={0, 1}, min_items=1, max_items=2)]
    )

    with pytest.raises(RuntimeError, match="parallel form assembly failed"):
        assemble_parallel_forms(
            model,
            3,
            2,
            theta,
            blueprint=blueprint,
        )
