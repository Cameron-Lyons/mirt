"""Tests for constrained fixed-form assembly."""

from itertools import combinations

import numpy as np
import pytest

from mirt.cat import ContentArea, ContentBlueprint, FormAssemblyResult, assemble_form
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
            [8.0, 1.0, 1.0, 6.0, 5.0, 4.0],
            [2.0, 7.0, 1.0, 6.0, 5.0, 4.0],
            [1.0, 1.0, 5.0, 6.0, 5.0, 4.0],
        ]
    )


def test_maximizes_weighted_information(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_form(model, 2, theta)

    np.testing.assert_array_equal(result.selected_items, np.array([3, 4]))
    np.testing.assert_allclose(result.information, profiles[:, 3] + profiles[:, 4])
    assert result.objective_value == pytest.approx(11.0)
    assert result.method == "maximize_information"
    assert result.n_items == 2
    assert model.calls == 1


def test_uses_caller_supplied_theta_weights(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_form(
        model,
        1,
        theta,
        theta_weights=np.array([0.95, 0.04, 0.01]),
    )

    np.testing.assert_array_equal(result.selected_items, np.array([0]))
    assert result.objective_value == pytest.approx(7.69)


def test_enforces_content_minimums_and_maximums(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items={0, 1, 2}, min_items=2, max_items=2),
            ContentArea("B", items={3, 4, 5}, min_items=1, max_items=1),
        ]
    )

    result = assemble_form(model, 3, theta, blueprint=blueprint)

    np.testing.assert_array_equal(result.selected_items, np.array([0, 1, 3]))
    assert result.content_counts == {"A": 2, "B": 1}


def test_honors_candidate_required_and_excluded_items(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_form(
        model,
        2,
        theta,
        candidate_items={0, 1, 2, 4, 5},
        required_items={1},
        excluded_items={2},
    )

    np.testing.assert_array_equal(result.selected_items, np.array([1, 4]))


def test_enemy_pairs_prevent_joint_selection(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_form(model, 2, theta, enemy_pairs={(3, 4), (4, 3)})

    np.testing.assert_array_equal(result.selected_items, np.array([3, 5]))


def test_budget_limits_total_cost(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    costs = np.array([1.0, 1.0, 1.0, 10.0, 4.0, 3.0])

    result = assemble_form(
        model,
        2,
        theta,
        item_costs=costs,
        max_cost=7.0,
    )

    np.testing.assert_array_equal(result.selected_items, np.array([4, 5]))
    assert result.total_cost == pytest.approx(7.0)


def test_reports_cost_without_requiring_budget(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    costs = np.arange(1.0, 7.0)

    result = assemble_form(model, 2, theta, item_costs=costs)

    assert result.total_cost == pytest.approx(9.0)


def test_target_curve_matches_brute_force_optimum(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    target = np.array([9.0, 9.0, 2.0])
    weights = np.array([0.2, 0.5, 0.3])

    result = assemble_form(
        model,
        2,
        theta,
        theta_weights=weights,
        target_information=target,
    )

    reference = min(
        (
            float(weights @ np.abs(np.sum(profiles[:, pair], axis=1) - target)),
            pair,
        )
        for pair in combinations(range(profiles.shape[1]), 2)
    )
    assert result.objective_value == pytest.approx(reference[0], abs=1e-10)
    np.testing.assert_array_equal(result.selected_items, np.array(reference[1]))
    assert result.method == "target_information"


def test_scalar_target_is_broadcast(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    result = assemble_form(model, 1, theta, target_information=5.0)

    np.testing.assert_array_equal(result.selected_items, np.array([4]))
    np.testing.assert_allclose(result.information, 5.0)
    assert result.objective_value == pytest.approx(0.0, abs=1e-10)


def test_falls_back_to_itemwise_information(theta, profiles) -> None:
    model = ItemwiseInformationModel(profiles)

    result = assemble_form(model, 2, theta)

    np.testing.assert_array_equal(result.selected_items, np.array([3, 4]))
    assert model.calls == model.n_items + 1


def test_integrates_with_dichotomous_and_polytomous_models() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    dichotomous = TwoParameterLogistic(4)
    dichotomous.set_parameters(
        discrimination=np.array([0.6, 0.9, 1.3, 1.8]),
        difficulty=np.array([-1.2, -0.2, 0.4, 1.1]),
    )
    polytomous = GradedResponseModel(4, n_categories=3)
    polytomous.set_parameters(
        discrimination=np.array([0.6, 0.9, 1.3, 1.8]),
        thresholds=np.array(
            [
                [-1.2, 0.5],
                [-0.9, 0.8],
                [-0.6, 1.1],
                [-0.3, 1.4],
            ]
        ),
    )

    for model in (dichotomous, polytomous):
        information = np.column_stack(
            [
                model.information(theta[:, None], item_idx=item_idx)
                for item_idx in range(model.n_items)
            ]
        )
        expected = np.sort(np.argsort(np.mean(information, axis=0))[-2:])

        result = assemble_form(model, 2, theta)

        np.testing.assert_array_equal(result.selected_items, expected)
        np.testing.assert_allclose(
            result.information,
            np.sum(information[:, expected], axis=1),
        )


def test_summary_contains_objective_content_and_cost(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    blueprint = ContentBlueprint(
        [ContentArea("Core", items=set(range(6)), min_items=2, max_items=2)]
    )
    result = assemble_form(
        model,
        2,
        theta,
        blueprint=blueprint,
        item_costs=np.ones(6),
    )

    summary = result.summary()

    assert isinstance(result, FormAssemblyResult)
    assert "Items: 2" in summary
    assert "weighted information" in summary
    assert "Total cost: 2" in summary
    assert "Core=2" in summary


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"form_size": 0}, "positive"),
        ({"form_size": 7}, "candidate items"),
        ({"theta": np.array([])}, "non-empty"),
        ({"theta": np.array([0.0, np.nan])}, "finite"),
        ({"theta_weights": np.array([1.0, 1.0])}, "shape"),
        ({"theta_weights": np.zeros(3)}, "positive mass"),
        ({"target_information": np.ones(2)}, "scalar or have shape"),
        ({"target_information": -1.0}, "non-negative"),
        ({"candidate_items": [0, 0]}, "duplicate"),
        ({"candidate_items": [8]}, "outside"),
        ({"required_items": {1}, "candidate_items": {0, 2}}, "included"),
        ({"required_items": {1}, "excluded_items": {1}}, "disjoint"),
        ({"required_items": {0, 1}, "form_size": 1}, "smaller"),
        ({"enemy_pairs": {(0, 0)}}, "distinct"),
        ({"item_costs": np.ones(5)}, "shape"),
        ({"item_costs": np.ones(6), "max_cost": -1.0}, "non-negative"),
        ({"max_cost": 2.0}, "item_costs are required"),
    ],
)
def test_rejects_invalid_configuration(theta, profiles, kwargs, message) -> None:
    model = ProfileInformationModel(profiles)
    arguments = {"form_size": 2, "theta": theta}
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        assemble_form(model, **arguments)


def test_rejects_multidimensional_models(theta, profiles) -> None:
    model = ProfileInformationModel(profiles, n_factors=2)

    with pytest.raises(ValueError, match="unidimensional"):
        assemble_form(model, 2, theta)


def test_rejects_blueprint_items_outside_pool(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)
    blueprint = ContentBlueprint([ContentArea("Bad", items={6})])

    with pytest.raises(ValueError, match="outside the model"):
        assemble_form(model, 2, theta, blueprint=blueprint)


def test_rejects_negative_model_information(theta, profiles) -> None:
    invalid = profiles.copy()
    invalid[0, 0] = -0.1
    model = ProfileInformationModel(invalid)

    with pytest.raises(ValueError, match="non-negative"):
        assemble_form(model, 2, theta)


def test_reports_infeasible_constraints(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    with pytest.raises(RuntimeError, match="form assembly failed"):
        assemble_form(
            model,
            2,
            theta,
            required_items={3, 4},
            enemy_pairs={(3, 4)},
        )


def test_solver_options_must_be_mapping(theta, profiles) -> None:
    model = ProfileInformationModel(profiles)

    with pytest.raises(TypeError, match="mapping"):
        assemble_form(model, 2, theta, solver_options=[])  # type: ignore[arg-type]
