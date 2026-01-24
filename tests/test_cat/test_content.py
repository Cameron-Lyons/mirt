"""Tests for content balancing constraints."""

import pytest

from mirt.cat.content import (
    ContentArea,
    ContentBlueprint,
    NoContentConstraint,
    WeightedContent,
    create_content_constraint,
)


class TestContentArea:
    """Tests for ContentArea dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        area = ContentArea(name="Algebra", items={0, 1, 2}, min_items=1, max_items=3)

        assert area.name == "Algebra"
        assert area.items == {0, 1, 2}
        assert area.min_items == 1
        assert area.max_items == 3
        assert area.target_items is None

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        area = ContentArea(name="Test")

        assert area.name == "Test"
        assert area.items == set()
        assert area.min_items == 0
        assert area.max_items == 999
        assert area.target_items is None

    def test_initialization_with_target(self):
        """Test initialization with target_items."""
        area = ContentArea(name="Test", min_items=1, max_items=5, target_items=3)

        assert area.target_items == 3

    def test_validation_negative_min_items(self):
        """Test that negative min_items raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            ContentArea(name="Test", min_items=-1)

    def test_validation_max_less_than_min(self):
        """Test that max_items < min_items raises error."""
        with pytest.raises(ValueError, match="max_items must be >= min_items"):
            ContentArea(name="Test", min_items=5, max_items=3)

    def test_validation_target_less_than_min(self):
        """Test that target_items < min_items raises error."""
        with pytest.raises(ValueError, match="target_items must be >= min_items"):
            ContentArea(name="Test", min_items=5, max_items=10, target_items=3)

    def test_validation_target_greater_than_max(self):
        """Test that target_items > max_items raises error."""
        with pytest.raises(ValueError, match="target_items must be <= max_items"):
            ContentArea(name="Test", min_items=1, max_items=5, target_items=10)


class TestNoContentConstraint:
    """Tests for NoContentConstraint."""

    def test_filter_items_returns_all(self):
        """Test that all items are returned."""
        constraint = NoContentConstraint()
        available = {0, 1, 2, 3, 4}
        administered = [5, 6]

        filtered = constraint.filter_items(available, administered)

        assert filtered == available

    def test_filter_items_empty(self):
        """Test filtering empty set."""
        constraint = NoContentConstraint()
        filtered = constraint.filter_items(set(), [])

        assert filtered == set()

    def test_reset_does_nothing(self):
        """Test that reset does nothing."""
        constraint = NoContentConstraint()
        constraint.reset()


class TestContentBlueprint:
    """Tests for ContentBlueprint."""

    @pytest.fixture
    def blueprint_areas(self):
        """Create sample content areas."""
        return [
            ContentArea(name="Algebra", items={0, 1, 2, 3}, min_items=2, max_items=3),
            ContentArea(name="Geometry", items={4, 5, 6}, min_items=1, max_items=2),
            ContentArea(name="Statistics", items={7, 8, 9}, min_items=1, max_items=2),
        ]

    def test_initialization(self, blueprint_areas):
        """Test blueprint initialization."""
        blueprint = ContentBlueprint(blueprint_areas)

        assert len(blueprint.areas) == 3
        assert blueprint.strict is True

    def test_initialization_non_strict(self, blueprint_areas):
        """Test initialization with strict=False."""
        blueprint = ContentBlueprint(blueprint_areas, strict=False)
        assert blueprint.strict is False

    def test_initialization_overlapping_items_raises_error(self):
        """Test that overlapping items raises error."""
        areas = [
            ContentArea(name="Area1", items={0, 1, 2}),
            ContentArea(name="Area2", items={2, 3, 4}),
        ]

        with pytest.raises(ValueError, match="multiple content areas"):
            ContentBlueprint(areas)

    def test_filter_items_under_max(self, blueprint_areas):
        """Test filtering when under max for all areas."""
        blueprint = ContentBlueprint(blueprint_areas)
        available = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        administered: list[int] = []

        filtered = blueprint.filter_items(available, administered)

        assert filtered == available

    def test_filter_items_at_max(self, blueprint_areas):
        """Test filtering when area is at max."""
        blueprint = ContentBlueprint(blueprint_areas)
        available = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        administered = [0, 1, 2]

        filtered = blueprint.filter_items(available, administered)

        assert 3 not in filtered
        assert filtered.issubset(available)

    def test_filter_items_priority_items(self, blueprint_areas):
        """Test filtering returns priority items when needed."""
        blueprint = ContentBlueprint(blueprint_areas)
        available = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        administered = [0, 1, 2, 4, 5]

        filtered = blueprint.filter_items(available, administered)

        assert len(filtered) > 0

    def test_is_blueprint_satisfied_true(self, blueprint_areas):
        """Test is_blueprint_satisfied when all minimums met."""
        blueprint = ContentBlueprint(blueprint_areas)
        administered = [0, 1, 4, 7]

        assert blueprint.is_blueprint_satisfied(administered) is True

    def test_is_blueprint_satisfied_false(self, blueprint_areas):
        """Test is_blueprint_satisfied when minimums not met."""
        blueprint = ContentBlueprint(blueprint_areas)
        administered = [0, 1, 4]

        assert blueprint.is_blueprint_satisfied(administered) is False

    def test_get_area_counts(self, blueprint_areas):
        """Test getting area counts."""
        blueprint = ContentBlueprint(blueprint_areas)
        administered = [0, 1, 4, 7, 8]

        counts = blueprint.get_area_counts(administered)

        assert counts["Algebra"] == 2
        assert counts["Geometry"] == 1
        assert counts["Statistics"] == 2

    def test_get_remaining_requirements(self, blueprint_areas):
        """Test getting remaining requirements."""
        blueprint = ContentBlueprint(blueprint_areas)
        administered = [0, 4, 7]

        remaining = blueprint.get_remaining_requirements(administered)

        assert remaining["Algebra"] == (1, 2)
        assert remaining["Geometry"] == (0, 1)
        assert remaining["Statistics"] == (0, 1)

    def test_reset_clears_counts(self, blueprint_areas):
        """Test reset clears area counts."""
        blueprint = ContentBlueprint(blueprint_areas)
        blueprint._update_counts([0, 1, 4])

        assert blueprint._area_counts["Algebra"] == 2

        blueprint.reset()

        assert blueprint._area_counts["Algebra"] == 0

    def test_summary(self, blueprint_areas):
        """Test summary method."""
        blueprint = ContentBlueprint(blueprint_areas)
        summary = blueprint.summary()

        assert "Content Blueprint" in summary
        assert "Algebra" in summary
        assert "Geometry" in summary
        assert "Statistics" in summary


class TestWeightedContent:
    """Tests for WeightedContent."""

    def test_initialization(self):
        """Test initialization."""
        item_weights = {0: 1.0, 1: 1.5, 2: 0.5}
        area_targets = {"Math": 0.5, "Reading": 0.5}
        item_areas = {0: "Math", 1: "Math", 2: "Reading"}

        weighted = WeightedContent(item_weights, area_targets, item_areas)

        assert weighted.item_weights == item_weights
        assert weighted.area_targets == area_targets
        assert weighted.item_areas == item_areas

    def test_filter_items_returns_all(self):
        """Test that filter_items returns all available."""
        item_weights = {0: 1.0, 1: 1.5, 2: 0.5}
        area_targets = {"Math": 0.5, "Reading": 0.5}
        item_areas = {0: "Math", 1: "Math", 2: "Reading"}

        weighted = WeightedContent(item_weights, area_targets, item_areas)
        available = {0, 1, 2}

        filtered = weighted.filter_items(available, [])

        assert filtered == available

    def test_get_adjusted_weights_no_administered(self):
        """Test adjusted weights with no administered items."""
        item_weights = {0: 1.0, 1: 1.5, 2: 0.5}
        area_targets = {"Math": 0.5, "Reading": 0.5}
        item_areas = {0: "Math", 1: "Math", 2: "Reading"}

        weighted = WeightedContent(item_weights, area_targets, item_areas)
        available = {0, 1, 2}

        weights = weighted.get_adjusted_weights(available, [])

        assert weights[0] == 1.0
        assert weights[1] == 1.5
        assert weights[2] == 0.5

    def test_get_adjusted_weights_with_administered(self):
        """Test adjusted weights boost underrepresented areas."""
        item_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        area_targets = {"Math": 0.5, "Reading": 0.5}
        item_areas = {0: "Math", 1: "Math", 2: "Reading"}

        weighted = WeightedContent(item_weights, area_targets, item_areas)
        available = {1, 2}
        administered = [0]

        weights = weighted.get_adjusted_weights(available, administered)

        assert weights[2] > weights[1]


class TestCreateContentConstraint:
    """Tests for create_content_constraint factory."""

    @pytest.mark.parametrize(
        "method,expected_class",
        [
            ("blueprint", ContentBlueprint),
            ("weighted", WeightedContent),
            ("none", NoContentConstraint),
            (None, NoContentConstraint),
        ],
    )
    def test_create_valid_methods(self, method, expected_class):
        """Test creating valid constraint methods."""
        if method == "blueprint":
            areas = [ContentArea(name="Test", items={0, 1})]
            constraint = create_content_constraint(method, areas=areas)
        elif method == "weighted":
            constraint = create_content_constraint(
                method,
                item_weights={0: 1.0},
                area_targets={"Test": 1.0},
                item_areas={0: "Test"},
            )
        else:
            constraint = create_content_constraint(method)

        assert isinstance(constraint, expected_class)

    def test_create_invalid_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown content constraint"):
            create_content_constraint("invalid_method")

    def test_case_insensitivity(self):
        """Test case insensitivity."""
        areas = [ContentArea(name="Test", items={0, 1})]

        constraint1 = create_content_constraint("Blueprint", areas=areas)
        constraint2 = create_content_constraint("BLUEPRINT", areas=areas)

        assert type(constraint1) is type(constraint2)


class TestContentConstraintInterface:
    """Tests for ContentConstraint interface."""

    def test_all_constraints_have_filter_items(self):
        """Test that all constraints implement filter_items."""
        areas = [ContentArea(name="Test", items={0, 1, 2})]
        constraints = [
            NoContentConstraint(),
            ContentBlueprint(areas),
            WeightedContent(
                item_weights={0: 1.0},
                area_targets={"Test": 1.0},
                item_areas={0: "Test"},
            ),
        ]

        for constraint in constraints:
            assert hasattr(constraint, "filter_items")
            filtered = constraint.filter_items({0, 1, 2}, [])
            assert isinstance(filtered, set)

    def test_all_constraints_have_reset(self):
        """Test that all constraints implement reset."""
        areas = [ContentArea(name="Test", items={0, 1, 2})]
        constraints = [
            NoContentConstraint(),
            ContentBlueprint(areas),
            WeightedContent(
                item_weights={0: 1.0},
                area_targets={"Test": 1.0},
                item_areas={0: "Test"},
            ),
        ]

        for constraint in constraints:
            assert hasattr(constraint, "reset")
            constraint.reset()
