"""Regression contracts for CAT content balancing."""

from __future__ import annotations

import math

import pytest

from mirt.cat.content import (
    ContentArea,
    ContentBlueprint,
    NoContentConstraint,
    WeightedContent,
    create_content_constraint,
)
from mirt.cat.engine import CATEngine


class TestContentAreaContracts:
    """Validate configuration at the boundary and own mutable inputs."""

    def test_normalizes_name_and_owns_items(self) -> None:
        items = {0, 1}
        area = ContentArea("  Reading  ", items=items, min_items=1)

        items.clear()

        assert area.name == "Reading"
        assert area.items == {0, 1}

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_empty_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ContentArea(name)

    @pytest.mark.parametrize("value", [True, 1.5, "1"])
    def test_rejects_non_integer_counts(self, value: object) -> None:
        with pytest.raises(TypeError, match="min_items must be an integer"):
            ContentArea("A", items={0, 1}, min_items=value)  # type: ignore[arg-type]

    def test_rejects_impossible_minimum(self) -> None:
        with pytest.raises(ValueError, match="number of area items"):
            ContentArea("A", items={0}, min_items=2)

    def test_rejects_impossible_target(self) -> None:
        with pytest.raises(ValueError, match="number of area items"):
            ContentArea("A", items={0}, target_items=2)


class TestBlueprintContracts:
    """Keep hard constraints enforceable throughout a session."""

    def test_owns_area_configuration(self) -> None:
        original = ContentArea("A", items={0, 1}, min_items=1)
        areas = [original]
        blueprint = ContentBlueprint(areas)

        original.items.clear()
        areas.clear()

        assert blueprint.areas[0].items == {0, 1}

    def test_rejects_duplicate_normalized_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate content area name"):
            ContentBlueprint(
                [
                    ContentArea(" A ", items={0}),
                    ContentArea("A", items={1}),
                ]
            )

    def test_strict_mode_prioritizes_only_unmet_areas(self) -> None:
        blueprint = ContentBlueprint(
            [
                ContentArea("A", items={0, 1, 2}, min_items=1, max_items=2),
                ContentArea("B", items={3, 4, 5}, min_items=2, max_items=3),
            ]
        )

        assert blueprint.filter_items({1, 2, 3, 4, 5}, [0]) == {3, 4, 5}

    def test_strict_mode_never_relaxes_a_maximum(self) -> None:
        blueprint = ContentBlueprint([ContentArea("A", items={0, 1}, max_items=1)])

        with pytest.raises(RuntimeError, match="maximums"):
            blueprint.filter_items({1}, [0])

    def test_strict_mode_reports_infeasible_remaining_minimum(self) -> None:
        blueprint = ContentBlueprint(
            [ContentArea("A", items={0, 1}, min_items=2, max_items=2)]
        )

        assert blueprint.is_feasible({0}, []) is False
        assert blueprint.get_unmet_areas([]) == ["A"]
        with pytest.raises(RuntimeError, match="cannot be met.*A"):
            blueprint.filter_items({0}, [])

    def test_non_strict_mode_prefers_unmet_targets(self) -> None:
        blueprint = ContentBlueprint(
            [
                ContentArea("A", items={0, 1}, max_items=2, target_items=2),
                ContentArea("B", items={2, 3}, max_items=2, target_items=1),
            ],
            strict=False,
        )

        assert blueprint.filter_items({0, 1, 3}, [2]) == {0, 1}

    def test_satisfaction_includes_maximums(self) -> None:
        blueprint = ContentBlueprint([ContentArea("A", items={0, 1}, max_items=1)])

        assert blueprint.is_blueprint_satisfied([0, 1]) is False
        assert blueprint.is_feasible({2}, [0, 1]) is False

    def test_rejects_invalid_candidate_history(self) -> None:
        blueprint = ContentBlueprint([ContentArea("A", items={0, 1})])

        with pytest.raises(ValueError, match="must not contain duplicates"):
            blueprint.filter_items({1}, [0, 0])
        with pytest.raises(ValueError, match="must not include administered"):
            blueprint.filter_items({0, 1}, [0])

    def test_engine_selects_from_an_unmet_area(self, fitted_2pl_model) -> None:
        model = fitted_2pl_model.model
        preferred = {model.n_items - 2, model.n_items - 1}
        blueprint = ContentBlueprint(
            [
                ContentArea("Other", items=set(range(model.n_items - 2))),
                ContentArea("Required", items=preferred, min_items=1, max_items=2),
            ]
        )
        engine = CATEngine(model, content_constraint=blueprint)

        assert engine.select_next_item() in preferred


class TestWeightedContentContracts:
    """Make area weights affect eligibility predictably."""

    def test_normalizes_targets_and_owns_configuration(self) -> None:
        weights = {0: 2.0, 1: 1.0}
        targets = {"A": 2.0, "B": 1.0}
        areas = {0: "A", 1: "B"}
        weighted = WeightedContent(weights, targets, areas)

        weights[0] = 99.0
        targets["A"] = 99.0
        areas[0] = "B"
        returned_weights = weighted.item_weights
        returned_weights[0] = 100.0

        assert weighted.item_weights[0] == 2.0
        assert weighted.item_areas[0] == "A"
        assert weighted.area_targets == pytest.approx({"A": 2 / 3, "B": 1 / 3})

    def test_filter_applies_underrepresentation_priority(self) -> None:
        weighted = WeightedContent(
            item_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            area_targets={"A": 0.5, "B": 0.5},
            item_areas={0: "A", 1: "A", 2: "B", 3: "B"},
        )

        assert weighted.filter_items({1, 2, 3}, [0]) == {2, 3}

    def test_top_k_uses_weight_then_item_index(self) -> None:
        weighted = WeightedContent(
            item_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            area_targets={"A": 0.5, "B": 0.5},
            item_areas={0: "A", 1: "A", 2: "B", 3: "B"},
            top_k=1,
        )

        assert weighted.filter_items({1, 2, 3}, [0]) == {2}

    @pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
    def test_rejects_invalid_weights(self, weight: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            WeightedContent({0: weight}, {"A": 1.0}, {0: "A"})

    def test_rejects_empty_targets(self) -> None:
        with pytest.raises(ValueError, match="positive total"):
            WeightedContent({}, {}, {})

    def test_rejects_missing_area_target(self) -> None:
        with pytest.raises(ValueError, match="Missing target"):
            WeightedContent({0: 1.0}, {"A": 1.0}, {0: "B"})

    def test_rejects_unmapped_weighted_item(self) -> None:
        with pytest.raises(ValueError, match="Every weighted item"):
            WeightedContent({0: 1.0}, {"A": 1.0}, {})


def test_no_constraint_returns_an_owned_set() -> None:
    available = {0, 1}
    filtered = NoContentConstraint().filter_items(available, [])

    filtered.clear()

    assert available == {0, 1}


def test_factory_normalizes_method_whitespace() -> None:
    constraint = create_content_constraint(
        "  WeIgHtEd  ",
        item_weights={0: 1.0},
        area_targets={"A": 1.0},
        item_areas={0: "A"},
    )

    assert isinstance(constraint, WeightedContent)
