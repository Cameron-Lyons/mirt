"""Tests for validated and vectorized IRTree models."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.estimation import IRTreeEMEstimator as PublicIRTreeEMEstimator
from mirt.estimation import IRTreeResult
from mirt.estimation.irtree_em import IRTreeEMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models import IRTreeModel as PublicIRTreeModel
from mirt.models import IRTreeSpec as PublicIRTreeSpec
from mirt.models import TreeNode as PublicTreeNode
from mirt.models.irtree import IRTreeModel, IRTreeSpec, TreeNode

BUILT_IN_SPECS = ("bockenholt", "extreme_midpoint", "direction_intensity")


@pytest.mark.parametrize("tree_spec", BUILT_IN_SPECS)
def test_built_in_category_probabilities_are_normalized(tree_spec: str) -> None:
    model = IRTreeModel(n_items=3, tree_spec=tree_spec)
    theta = np.array(
        [
            [-1000.0, 0.0, 1000.0],
            [-1.5, 0.2, 0.7],
            [0.0, 0.0, 0.0],
            [1.5, -0.2, -0.7],
            [1000.0, 0.0, -1000.0],
        ]
    )[:, : model.n_traits]

    probabilities = model.probability(theta)

    assert probabilities.shape == (5, 3, 5)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


@pytest.mark.parametrize(
    ("tree_spec", "expected"),
    [
        ("bockenholt", [0.125, 0.125, 0.5, 0.125, 0.125]),
        ("extreme_midpoint", [0.25, 0.125, 0.25, 0.125, 0.25]),
        ("direction_intensity", [0.125, 0.125, 0.5, 0.125, 0.125]),
    ],
)
def test_built_in_category_probabilities_at_zero(
    tree_spec: str, expected: list[float]
) -> None:
    model = IRTreeModel(n_items=1, tree_spec=tree_spec)

    probabilities = model.probability(np.zeros(model.n_traits), item_idx=0)

    np.testing.assert_allclose(probabilities[0], expected, atol=1e-14)


def test_bockenholt_expansion_uses_stable_node_columns() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")
    responses = np.arange(5, dtype=int).reshape(-1, 1)

    pseudo, traits, valid = model.expand_to_pseudo_items(responses)

    assert model.node_names == (
        "acquiescence",
        "direction",
        "intensity_disagree",
        "intensity_agree",
    )
    np.testing.assert_array_equal(traits, [[0, 1, 2, 2]])
    np.testing.assert_array_equal(
        pseudo[:, 0],
        [
            [0, 0, 1, -1],
            [0, 0, 0, -1],
            [1, -1, -1, -1],
            [0, 1, -1, 0],
            [0, 1, -1, 1],
        ],
    )
    np.testing.assert_array_equal(valid, pseudo >= 0)


@pytest.mark.parametrize("tree_spec", BUILT_IN_SPECS)
def test_expansion_allocates_every_tree_node(tree_spec: str) -> None:
    model = IRTreeModel(n_items=2, tree_spec=tree_spec)

    pseudo, traits, valid = model.expand_to_pseudo_items(
        np.array([[0, 4], [2, 3]], dtype=int)
    )

    assert pseudo.shape == (2, 2, model.n_nodes)
    assert traits.shape == (2, model.n_nodes)
    assert valid.shape == pseudo.shape
    assert set(np.unique(pseudo)).issubset({-1, 0, 1})


def test_expansion_supports_negative_and_nan_missing_values() -> None:
    model = IRTreeModel(n_items=2, tree_spec="bockenholt")

    pseudo, _, valid = model.expand_to_pseudo_items(
        np.array([[-1.0, np.nan], [-5.5, 2.0]])
    )

    np.testing.assert_array_equal(pseudo[0], -1)
    np.testing.assert_array_equal(valid[0], False)
    np.testing.assert_array_equal(pseudo[1, 0], -1)
    assert valid[1, 1, 0]


@pytest.mark.parametrize("bad_response", [0.5, 5.0, np.inf])
def test_invalid_observed_responses_are_rejected(bad_response: float) -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")

    with pytest.raises(MirtDataError):
        model.expand_to_pseudo_items(np.array([[bad_response]]))
    with pytest.raises(MirtDataError):
        model.log_likelihood(np.array([[bad_response]]), np.zeros((1, model.n_traits)))


def test_response_shape_is_validated() -> None:
    model = IRTreeModel(n_items=2, tree_spec="bockenholt")

    with pytest.raises(MirtDataError, match="two-dimensional"):
        model.expand_to_pseudo_items(np.array([0, 1]))
    with pytest.raises(MirtDataError, match="item count"):
        model.expand_to_pseudo_items(np.array([[0]]))


def test_branch_specific_parameters_control_the_correct_node() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")
    theta = np.zeros((1, model.n_traits))
    baseline = model.probability(theta, item_idx=0)[0]
    difficulty = np.zeros((1, model.n_nodes))
    difficulty[0, 3] = 2.0

    model.set_parameters(difficulty=difficulty)
    updated = model.probability(theta, item_idx=0)[0]

    np.testing.assert_allclose(updated[:3], baseline[:3], atol=1e-14)
    assert updated[3] > baseline[3]
    assert updated[4] < baseline[4]
    assert updated[3] + updated[4] == pytest.approx(baseline[3] + baseline[4])


def test_probability_item_slice_matches_full_result() -> None:
    model = IRTreeModel(n_items=3, tree_spec="extreme_midpoint")
    difficulty = (
        np.arange(model.n_items * model.n_nodes, dtype=float).reshape(
            model.n_items, model.n_nodes
        )
        / 10.0
    )
    model.set_parameters(difficulty=difficulty)
    theta = np.array([[0.2, -0.4, 0.7], [-0.1, 0.3, 0.5]])

    full = model.probability(theta)

    assert full.shape == (2, 3, 5)
    np.testing.assert_allclose(model.probability(theta, item_idx=1), full[:, 1])


def test_decision_and_reach_probabilities_expose_tree_processes() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")
    theta = np.zeros((1, model.n_traits))

    decisions = model.decision_probability(theta, item_idx=0)
    reach = model.reach_probability(theta, item_idx=0)

    np.testing.assert_allclose(decisions, 0.5)
    np.testing.assert_allclose(reach, [[1.0, 0.5, 0.25, 0.25]])


def test_expected_score_matches_probability_weighted_categories() -> None:
    model = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    theta = np.array([[0.1, -0.2], [0.5, 0.3]])
    probabilities = model.probability(theta)

    expected = model.expected_score(theta)

    np.testing.assert_allclose(expected, probabilities @ np.arange(5))
    np.testing.assert_allclose(model.expected_score(theta, item_idx=1), expected[:, 1])


def test_simulate_is_reproducible_across_chunk_sizes() -> None:
    model = IRTreeModel(n_items=3, tree_spec="bockenholt")
    theta = np.linspace(-1.5, 1.5, 57)[:, None] * np.array([[1.0, -0.5, 0.25]])

    unchunked = model.simulate(theta, seed=20260829, chunk_size=len(theta))
    chunked = model.simulate(theta, seed=20260829, chunk_size=7)
    automatic = model.simulate(theta, seed=20260829)

    assert unchunked.shape == (57, 3)
    assert unchunked.dtype == np.int32
    np.testing.assert_array_equal(chunked, unchunked)
    np.testing.assert_array_equal(automatic, unchunked)
    assert np.all((unchunked >= 0) & (unchunked < model.n_categories))


def test_simulate_matches_category_probabilities() -> None:
    model = IRTreeModel(n_items=2, tree_spec="extreme_midpoint")
    model.set_parameters(
        discrimination=np.array([[0.8, 1.2, 1.4, 0.7], [1.1, 0.9, 1.3, 1.6]]),
        difficulty=np.array([[0.2, -0.3, 0.5, 0.1], [-0.2, 0.4, 0.0, -0.6]]),
    )
    theta = np.repeat(np.array([[0.1, -0.2, 0.3]]), 50_000, axis=0)

    responses = model.simulate(theta, seed=981, chunk_size=911)
    frequencies = np.stack(
        [
            np.bincount(responses[:, item_idx], minlength=model.n_categories)
            / len(responses)
            for item_idx in range(model.n_items)
        ]
    )

    np.testing.assert_allclose(
        frequencies,
        model.probability(theta[0], item_idx=None)[0],
        atol=0.008,
    )


def test_simulate_accepts_single_person_theta() -> None:
    model = IRTreeModel(n_items=4, tree_spec="direction_intensity")

    responses = model.simulate(np.array([0.1, -0.4]), seed=73)

    assert responses.shape == (1, 4)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_simulate_rejects_invalid_chunk_size(chunk_size: object) -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")

    with pytest.raises(ValueError, match="chunk_size"):
        model.simulate(
            np.zeros((2, model.n_traits)),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


def test_information_matches_category_score_definition() -> None:
    model = IRTreeModel(n_items=1, tree_spec="extreme_midpoint")
    model.set_parameters(
        discrimination=np.array([[1.2, 0.7, 1.5, 0.8]]),
        difficulty=np.array([[0.1, -0.3, 0.5, -0.2]]),
    )
    theta = np.array([[0.2, -0.4, 0.7]])
    probabilities = model.probability(theta, item_idx=0)[0]
    epsilon = 1e-5
    score_columns = []
    for trait_idx in range(model.n_traits):
        theta_high = theta.copy()
        theta_low = theta.copy()
        theta_high[0, trait_idx] += epsilon
        theta_low[0, trait_idx] -= epsilon
        score_columns.append(
            (
                np.log(model.probability(theta_high, item_idx=0)[0])
                - np.log(model.probability(theta_low, item_idx=0)[0])
            )
            / (2.0 * epsilon)
        )
    scores = np.column_stack(score_columns)
    expected = np.einsum("c,ci,cj->ij", probabilities, scores, scores)

    information = model.information(theta, item_idx=0)

    assert information.shape == (1, model.n_traits, model.n_traits)
    np.testing.assert_allclose(information[0], expected, atol=1e-9)


def test_information_full_and_item_shapes_are_consistent() -> None:
    model = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    theta = np.zeros((3, model.n_traits))

    information = model.information(theta)

    assert information.shape == (3, 2, 2, 2)
    np.testing.assert_allclose(model.information(theta, 1), information[:, 1])


def test_log_likelihood_is_vectorized_and_ignores_missing() -> None:
    model = IRTreeModel(n_items=2, tree_spec="extreme_midpoint")
    responses = np.array([[0.0, 4.0], [np.nan, 2.0], [3.0, -1.0]])
    theta = np.array([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2], [-0.5, 0.1, 0.2]])
    probabilities = model.probability(theta)
    expected = np.array(
        [
            np.log(probabilities[0, 0, 0]) + np.log(probabilities[0, 1, 4]),
            np.log(probabilities[1, 1, 2]),
            np.log(probabilities[2, 0, 3]),
        ]
    )

    np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)


def test_expanded_and_category_log_likelihoods_match() -> None:
    model = IRTreeModel(n_items=2, tree_spec="bockenholt")
    model.set_parameters(
        discrimination=np.array([[0.8, 1.2, 1.4, 0.7], [1.1, 0.9, 1.3, 1.6]]),
        difficulty=np.array([[0.2, -0.3, 0.5, 0.1], [-0.2, 0.4, 0.0, -0.6]]),
    )
    responses = np.array([[0, 4], [2, 3], [1, -1]], dtype=int)
    theta = np.array([[0.1, -0.2, 0.3], [0.5, 0.2, -0.4], [-0.3, 0.7, 0.1]])
    pseudo, assignments, valid = model.expand_to_pseudo_items(responses)
    estimator = IRTreeEMEstimator()

    expanded = np.array(
        [
            estimator._compute_log_likelihood_at_theta(
                model, pseudo, assignments, valid, person_theta
            )[person_idx]
            for person_idx, person_theta in enumerate(theta)
        ]
    )

    np.testing.assert_allclose(expanded, model.log_likelihood(responses, theta))


@pytest.mark.parametrize(
    "theta",
    [
        np.zeros((2, 2)),
        np.zeros((2, 3, 1)),
        np.array([[0.0, np.nan, 0.0]]),
        np.array([[0.0, np.inf, 0.0]]),
    ],
)
def test_invalid_theta_is_rejected(theta: np.ndarray) -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")

    with pytest.raises(MirtValidationError):
        model.probability(theta)


@pytest.mark.parametrize("item_idx", [-1, 2, True])
def test_invalid_item_indices_are_rejected(item_idx: int) -> None:
    model = IRTreeModel(n_items=2, tree_spec="bockenholt")

    with pytest.raises(IndexError):
        model.probability(np.zeros((1, model.n_traits)), item_idx=item_idx)
    with pytest.raises(IndexError):
        model.get_item_parameters(item_idx)


def test_parameter_updates_are_validated_and_atomic() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")
    before = model.parameters

    with pytest.raises(MirtValidationError, match="Shape mismatch"):
        model.set_parameters(
            difficulty=np.ones((1, model.n_nodes)),
            discrimination=np.ones((1, model.n_nodes - 1)),
        )

    for name in before:
        np.testing.assert_array_equal(model.parameters[name], before[name])

    with pytest.raises(MirtValidationError, match="positive"):
        model.set_parameters(discrimination=np.zeros((1, model.n_nodes)))
    with pytest.raises(MirtValidationError, match="finite"):
        model.set_parameters(difficulty=np.full((1, model.n_nodes), np.nan))
    with pytest.raises(MirtValidationError, match="Unknown"):
        model.set_parameters(guessing=np.ones((1, model.n_nodes)))


def test_parameter_accessors_return_copies_and_validate_nodes() -> None:
    model = IRTreeModel(n_items=2, tree_spec="bockenholt")
    item_parameters = model.get_item_parameters(0)
    item_parameters["difficulty"][0] = 99.0
    assert model.parameters["difficulty"][0, 0] == 0.0

    model.set_item_parameter(1, "difficulty", np.arange(model.n_nodes, dtype=float))

    assert model.get_node_parameters(1, 3) == {
        "discrimination": 1.0,
        "difficulty": 3.0,
    }
    with pytest.raises(IndexError):
        model.get_node_parameters(0, model.n_nodes)


def test_trait_correlations_are_validated_and_copied() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt")
    correlations = np.array([[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]])

    model.set_parameters(trait_correlations=correlations)
    returned = model.trait_correlations
    assert returned is not None
    returned[0, 1] = 0.9
    np.testing.assert_array_equal(model.trait_correlations, correlations)

    with pytest.raises(MirtValidationError, match="symmetric"):
        model.set_parameters(trait_correlations=np.triu(np.ones((3, 3))))
    with pytest.raises(MirtValidationError, match="unit diagonal"):
        model.set_parameters(trait_correlations=np.eye(3) * 2.0)
    invalid_psd = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(MirtValidationError, match="positive semidefinite"):
        model.set_parameters(trait_correlations=invalid_psd)


def test_uncorrelated_model_rejects_trait_correlations() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt", correlated_traits=False)

    assert model.trait_correlations is None
    with pytest.raises(MirtValidationError, match="disabled"):
        model.set_parameters(trait_correlations=np.eye(model.n_traits))


def test_copy_is_independent() -> None:
    model = IRTreeModel(n_items=1, tree_spec="bockenholt", item_names=["Prompt"])
    clone = model.copy()

    clone.set_item_parameter(0, "difficulty", np.ones(model.n_nodes))

    np.testing.assert_array_equal(model.parameters["difficulty"], 0.0)
    np.testing.assert_array_equal(clone.parameters["difficulty"], 1.0)
    assert clone.item_names == ["Prompt"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_items": 0},
        {"n_items": True},
        {"n_items": 1, "n_categories": 1},
        {"n_items": 1, "item_names": []},
        {"n_items": 1, "item_names": [""]},
        {"n_items": 1, "correlated_traits": 1},
        {"n_items": 1, "tree_spec": "unknown"},
    ],
)
def test_model_structure_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(MirtValidationError):
        IRTreeModel(**kwargs)  # type: ignore[arg-type]


def test_custom_spec_category_count_must_match_model() -> None:
    spec = IRTreeSpec(
        name="Binary",
        n_categories=2,
        n_traits=1,
        trait_names=["Trait"],
        root=TreeNode("root", 0, {0: 0, 1: 1}),
    )

    with pytest.raises(MirtValidationError, match="n_categories must match"):
        IRTreeModel(n_items=1, tree_spec=spec, n_categories=5)


def test_custom_binary_tree_is_supported() -> None:
    spec = IRTreeSpec(
        name="Binary",
        n_categories=2,
        n_traits=1,
        trait_names=["Trait"],
        root=TreeNode("root", 0, {0: 0, 1: 1}),
    )
    model = IRTreeModel(n_items=2, tree_spec=spec, n_categories=2)

    probabilities = model.probability(np.array([[0.0], [1.0]]))

    assert model.n_nodes == 1
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)


def test_custom_spec_is_snapshotted_at_construction() -> None:
    spec = IRTreeSpec(
        name="Binary",
        n_categories=2,
        n_traits=1,
        trait_names=["Trait"],
        root=TreeNode("root", 0, {0: 0, 1: 1}),
    )
    model = IRTreeModel(n_items=1, tree_spec=spec, n_categories=2)

    spec.name = "Mutated"
    spec.root.children = {0: 1, 1: 0}

    assert model.tree_spec.name == "Binary"
    assert model.tree_spec.root.children == {0: 0, 1: 1}


@pytest.mark.parametrize(
    "root",
    [
        TreeNode("ternary", 0, {0: 0, 1: 1, 2: 2}),
        TreeNode("duplicate", 0, {0: 0, 1: 0}),
        TreeNode("bad_trait", 1, {0: 0, 1: 1}),
    ],
)
def test_invalid_custom_trees_are_rejected(root: TreeNode) -> None:
    spec = IRTreeSpec(
        name="Invalid",
        n_categories=2,
        n_traits=1,
        trait_names=["Trait"],
        root=root,
    )

    with pytest.raises(MirtValidationError):
        IRTreeModel(n_items=1, tree_spec=spec, n_categories=2)


def test_cyclic_custom_tree_is_rejected() -> None:
    root = TreeNode("root", 0)
    root.children = {0: root, 1: 1}
    spec = IRTreeSpec("Cycle", 2, 1, ["Trait"], root)

    with pytest.raises(MirtValidationError, match="cycle"):
        IRTreeModel(n_items=1, tree_spec=spec, n_categories=2)


def test_shared_custom_node_is_rejected() -> None:
    shared = TreeNode("shared", 0, {0: 0, 1: 1})
    root = TreeNode("root", 0, {0: shared, 1: shared})
    spec = IRTreeSpec("Shared", 2, 1, ["Trait"], root)

    with pytest.raises(MirtValidationError, match="shared"):
        IRTreeModel(n_items=1, tree_spec=spec, n_categories=2)


def test_category_paths_validate_bounds_and_match_terminals() -> None:
    spec = IRTreeSpec.bockenholt_adi()

    path = spec.get_path_to_category(4)

    assert [decision for _, decision in path] == [0, 1, 1]
    assert path[-1][0].get_child(path[-1][1]) == 4
    for category in (-1, 5, True):
        with pytest.raises(MirtValidationError):
            spec.get_path_to_category(category)


def test_public_model_exports_are_available() -> None:
    assert PublicIRTreeModel is IRTreeModel
    assert PublicIRTreeSpec is IRTreeSpec
    assert PublicTreeNode is TreeNode
    assert PublicIRTreeEMEstimator is IRTreeEMEstimator
    assert mirt.IRTreeModel is IRTreeModel
    assert mirt.IRTreeSpec is IRTreeSpec
    assert mirt.TreeNode is TreeNode
    assert mirt.IRTreeEMEstimator is IRTreeEMEstimator
    assert mirt.IRTreeResult is IRTreeResult


def test_summary_and_representation_include_structure() -> None:
    model = IRTreeModel(n_items=2, tree_spec="extreme_midpoint")

    summary = model.summary()

    assert "Extreme-Midpoint" in summary
    assert "Decision Nodes:     4" in summary
    assert "not fitted" in repr(model)
