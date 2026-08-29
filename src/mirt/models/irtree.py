from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError

_IRTREE_MAX_PROBABILITY_VALUES = 1_000_000


@dataclass
class TreeNode:
    """Node in the response tree structure.

    Each node represents a decision point in the response process.
    Terminal nodes map to response categories.

    Attributes
    ----------
    name : str
        Descriptive name for the decision (e.g., "direction", "intensity")
    latent_trait : int
        Index of the latent trait governing this decision
    children : dict
        Maps response values to child nodes or terminal category indices
    """

    name: str
    latent_trait: int
    children: dict[int, TreeNode | int] = field(default_factory=dict)

    def is_terminal(self, response: int) -> bool:
        """Check if response leads to a terminal node."""
        child = self.children.get(response)
        return child is None or isinstance(child, int)

    def get_child(self, response: int) -> TreeNode | int | None:
        """Get child node or terminal category for a response."""
        return self.children.get(response)

    def depth(self) -> int:
        """Compute maximum depth of subtree."""
        max_child_depth = 0
        for child in self.children.values():
            if isinstance(child, TreeNode):
                max_child_depth = max(max_child_depth, child.depth())
        return 1 + max_child_depth

    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in self.children.values():
            if isinstance(child, TreeNode):
                count += child.count_nodes()
        return count


@dataclass
class IRTreeSpec:
    """Specification for an IRTree model structure.

    Defines the tree structure that decomposes ordinal responses into
    a sequence of binary decisions.

    Attributes
    ----------
    name : str
        Model name
    n_categories : int
        Number of response categories
    n_traits : int
        Number of latent traits
    trait_names : list[str]
        Names for each latent trait
    root : TreeNode
        Root node of the decision tree
    """

    name: str
    n_categories: int
    n_traits: int
    trait_names: list[str]
    root: TreeNode

    @classmethod
    def bockenholt_adi(cls, n_categories: int = 5) -> IRTreeSpec:
        """Bockenholt's Acquiescence-Direction-Intensity model.

        For Likert scales, decomposes responses into:
        1. Acquiescence (0 = non-acquiescence, 1 = acquiescence)
        2. Direction (0 = disagree, 1 = agree) - only if non-acquiescence
        3. Intensity (0 = mild, 1 = extreme) - only if non-acquiescence

        Standard 5-point mapping:
        - 1 (Strongly Disagree): A=0, D=0, I=1
        - 2 (Disagree): A=0, D=0, I=0
        - 3 (Neutral): A=1
        - 4 (Agree): A=0, D=1, I=0
        - 5 (Strongly Agree): A=0, D=1, I=1
        """
        if n_categories != 5:
            raise ValueError("Bockenholt ADI model requires 5 categories")

        intensity_disagree = TreeNode(
            name="intensity_disagree",
            latent_trait=2,
            children={0: 1, 1: 0},
        )

        intensity_agree = TreeNode(
            name="intensity_agree",
            latent_trait=2,
            children={0: 3, 1: 4},
        )

        direction = TreeNode(
            name="direction",
            latent_trait=1,
            children={0: intensity_disagree, 1: intensity_agree},
        )

        root = TreeNode(
            name="acquiescence",
            latent_trait=0,
            children={0: direction, 1: 2},
        )

        return cls(
            name="Bockenholt ADI",
            n_categories=5,
            n_traits=3,
            trait_names=["Acquiescence", "Direction", "Intensity"],
            root=root,
        )

    @classmethod
    def extreme_midpoint(cls, n_categories: int = 5) -> IRTreeSpec:
        """Extreme response style and midpoint model.

        Decomposes responses into:
        1. Extreme (0 = non-extreme, 1 = extreme endpoint)
        2. Midpoint (0 = not midpoint, 1 = midpoint) - if non-extreme
        3. Direction (0 = low side, 1 = high side) - when needed

        5-point mapping:
        - 1: Extreme=1, Direction=0
        - 2: Extreme=0, Mid=0, Direction=0
        - 3: Extreme=0, Mid=1
        - 4: Extreme=0, Mid=0, Direction=1
        - 5: Extreme=1, Direction=1
        """
        if n_categories != 5:
            raise ValueError("Extreme-midpoint model requires 5 categories")

        direction_nonextreme = TreeNode(
            name="direction_nonextreme",
            latent_trait=1,
            children={0: 1, 1: 3},
        )

        direction_extreme = TreeNode(
            name="direction_extreme",
            latent_trait=1,
            children={0: 0, 1: 4},
        )

        midpoint = TreeNode(
            name="midpoint",
            latent_trait=2,
            children={0: direction_nonextreme, 1: 2},
        )

        root = TreeNode(
            name="extreme",
            latent_trait=0,
            children={0: midpoint, 1: direction_extreme},
        )

        return cls(
            name="Extreme-Midpoint",
            n_categories=5,
            n_traits=3,
            trait_names=["Extreme", "Direction", "Midpoint"],
            root=root,
        )

    @classmethod
    def simple_direction_intensity(cls, n_categories: int = 5) -> IRTreeSpec:
        """Simple two-process direction-intensity model.

        Decomposes responses into:
        1. Intensity presence (0 = neutral, 1 = directional response)
        2. Direction (0 = disagree, 1 = agree)
        3. Extremity (0 = mild, 1 = extreme)

        5-point mapping:
        - 1: I=1, D=0, E=1 (strong disagree)
        - 2: I=1, D=0, E=0 (disagree)
        - 3: I=0 (neutral)
        - 4: I=1, D=1, E=0 (agree)
        - 5: I=1, D=1, E=1 (strong agree)

        The intensity-presence and extremity decisions share the intensity
        latent trait while retaining separate item parameters.
        """
        if n_categories != 5:
            raise ValueError("Direction-intensity model requires 5 categories")

        intensity_low = TreeNode(
            name="intensity_low",
            latent_trait=1,
            children={0: 1, 1: 0},
        )

        intensity_high = TreeNode(
            name="intensity_high",
            latent_trait=1,
            children={0: 3, 1: 4},
        )

        direction = TreeNode(
            name="direction",
            latent_trait=0,
            children={0: intensity_low, 1: intensity_high},
        )

        root = TreeNode(
            name="intensity_presence",
            latent_trait=1,
            children={0: 2, 1: direction},
        )

        return cls(
            name="Direction-Intensity",
            n_categories=5,
            n_traits=2,
            trait_names=["Direction", "Intensity"],
            root=root,
        )

    def get_path_to_category(self, category: int) -> list[tuple[TreeNode, int]]:
        """Get the path of decisions leading to a category.

        Returns list of (node, decision) pairs from root to terminal.
        """
        if (
            isinstance(category, bool)
            or not isinstance(category, Integral)
            or category < 0
            or category >= self.n_categories
        ):
            raise MirtValidationError(
                f"category must be in [0, {self.n_categories})",
                parameter="category",
                value=category,
                expected=f"integer in [0, {self.n_categories})",
            )
        path: list[tuple[TreeNode, int]] = []
        if not self._find_path(self.root, int(category), path):
            raise MirtValidationError(
                f"category {category} is not reachable in the tree",
                parameter="category",
                value=category,
            )
        return path

    def _find_path(
        self, node: TreeNode, category: int, path: list[tuple[TreeNode, int]]
    ) -> bool:
        """Recursively find path to category."""
        for decision, child in node.children.items():
            if isinstance(child, int):
                if child == category:
                    path.append((node, decision))
                    return True
            else:
                path.append((node, decision))
                if self._find_path(child, category, path):
                    return True
                path.pop()
        return False

    def validate(self) -> None:
        """Validate that the specification is a complete binary tree.

        IRTree pseudo-items are binary decisions, and deterministic expansion
        requires every response category to have exactly one terminal path.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise MirtValidationError(
                "name must be a non-empty string", parameter="name", value=self.name
            )
        if (
            isinstance(self.n_categories, bool)
            or not isinstance(self.n_categories, Integral)
            or self.n_categories < 2
        ):
            raise MirtValidationError(
                "n_categories must be an integer of at least 2",
                parameter="n_categories",
                value=self.n_categories,
                expected=">= 2",
            )
        if (
            isinstance(self.n_traits, bool)
            or not isinstance(self.n_traits, Integral)
            or self.n_traits < 1
        ):
            raise MirtValidationError(
                "n_traits must be a positive integer",
                parameter="n_traits",
                value=self.n_traits,
                expected=">= 1",
            )
        if len(self.trait_names) != self.n_traits:
            raise MirtValidationError(
                "trait_names length must match n_traits",
                parameter="trait_names",
                value=len(self.trait_names),
                expected=str(self.n_traits),
            )
        if any(
            not isinstance(name, str) or not name.strip() for name in self.trait_names
        ):
            raise MirtValidationError(
                "trait_names must contain non-empty strings",
                parameter="trait_names",
            )
        if len(set(self.trait_names)) != len(self.trait_names):
            raise MirtValidationError(
                "trait_names must be unique", parameter="trait_names"
            )
        if not isinstance(self.root, TreeNode):
            raise MirtValidationError(
                "root must be a TreeNode", parameter="root", value=type(self.root)
            )

        seen: set[int] = set()
        active: set[int] = set()
        terminals: list[int] = []

        def visit(node: TreeNode) -> None:
            node_id = id(node)
            if node_id in active:
                raise MirtValidationError("tree contains a cycle", parameter="root")
            if node_id in seen:
                raise MirtValidationError(
                    "tree nodes cannot be shared across branches", parameter="root"
                )
            if not isinstance(node.name, str) or not node.name.strip():
                raise MirtValidationError(
                    "node names must be non-empty strings",
                    parameter="node.name",
                    value=node.name,
                )
            if (
                isinstance(node.latent_trait, bool)
                or not isinstance(node.latent_trait, Integral)
                or node.latent_trait < 0
                or node.latent_trait >= self.n_traits
            ):
                raise MirtValidationError(
                    f"node {node.name!r} has an invalid latent trait",
                    parameter="latent_trait",
                    value=node.latent_trait,
                    expected=f"integer in [0, {self.n_traits})",
                )
            if set(node.children) != {0, 1}:
                raise MirtValidationError(
                    f"node {node.name!r} must have decisions 0 and 1",
                    parameter="children",
                    value=tuple(node.children),
                    expected="{0, 1}",
                )

            active.add(node_id)
            seen.add(node_id)
            for child in node.children.values():
                if isinstance(child, TreeNode):
                    visit(child)
                elif isinstance(child, bool) or not isinstance(child, Integral):
                    raise MirtValidationError(
                        "terminal values must be integer response categories",
                        parameter="children",
                        value=child,
                    )
                else:
                    terminals.append(int(child))
            active.remove(node_id)

        visit(self.root)

        expected = list(range(self.n_categories))
        if sorted(terminals) != expected:
            raise MirtValidationError(
                "terminal categories must contain each response category exactly once",
                parameter="children",
                value=sorted(terminals),
                expected=str(expected),
            )


class IRTreeModel:
    """IRTree model for ordinal responses with response styles.

    IRTree models decompose ordinal responses into a sequence of binary
    decisions represented as a tree structure. This enables modeling of
    response styles (acquiescence, extreme responding, midpoint endorsement)
    separately from substantive content.

    Parameters
    ----------
    n_items : int
        Number of items
    tree_spec : IRTreeSpec or str
        Tree structure specification. Can be "bockenholt", "extreme_midpoint",
        "direction_intensity", or a custom IRTreeSpec.
    n_categories : int
        Number of response categories (default 5 for Likert)
    item_names : list[str], optional
        Names for each item
    correlated_traits : bool
        Whether to estimate trait correlations (default True)
    """

    model_name = "IRTree"
    n_items: int
    n_categories: int
    correlated_traits: bool
    item_names: list[str]
    tree_spec: IRTreeSpec
    n_traits: int
    trait_names: list[str]
    _nodes: tuple[TreeNode, ...]
    _node_traits: NDArray[np.intp]
    _path_decisions: NDArray[np.int8]
    _reach_decisions: NDArray[np.int8]
    _is_fitted: bool

    def __init__(
        self,
        n_items: int,
        tree_spec: IRTreeSpec
        | Literal[
            "bockenholt", "extreme_midpoint", "direction_intensity"
        ] = "bockenholt",
        n_categories: int = 5,
        item_names: list[str] | None = None,
        correlated_traits: bool = True,
    ) -> None:
        if (
            isinstance(n_items, bool)
            or not isinstance(n_items, Integral)
            or n_items < 1
        ):
            raise MirtValidationError(
                "n_items must be a positive integer",
                parameter="n_items",
                value=n_items,
                expected=">= 1",
            )
        if (
            isinstance(n_categories, bool)
            or not isinstance(n_categories, Integral)
            or n_categories < 2
        ):
            raise MirtValidationError(
                "n_categories must be an integer of at least 2",
                parameter="n_categories",
                value=n_categories,
                expected=">= 2",
            )
        if not isinstance(correlated_traits, bool):
            raise MirtValidationError(
                "correlated_traits must be boolean",
                parameter="correlated_traits",
                value=correlated_traits,
            )

        self.n_items = int(n_items)
        self.n_categories = int(n_categories)
        self.correlated_traits = correlated_traits

        if item_names is None:
            self.item_names = [f"Item_{i}" for i in range(self.n_items)]
        else:
            if len(item_names) != self.n_items:
                raise MirtValidationError(
                    "item_names length must match n_items",
                    parameter="item_names",
                    value=len(item_names),
                    expected=str(self.n_items),
                )
            if any(
                not isinstance(name, str) or not name.strip() for name in item_names
            ):
                raise MirtValidationError(
                    "item_names must contain non-empty strings",
                    parameter="item_names",
                )
            self.item_names = list(item_names)

        if isinstance(tree_spec, str):
            if tree_spec == "bockenholt":
                self.tree_spec = IRTreeSpec.bockenholt_adi(n_categories)
            elif tree_spec == "extreme_midpoint":
                self.tree_spec = IRTreeSpec.extreme_midpoint(n_categories)
            elif tree_spec == "direction_intensity":
                self.tree_spec = IRTreeSpec.simple_direction_intensity(n_categories)
            else:
                raise MirtValidationError(
                    f"Unknown tree spec: {tree_spec}",
                    parameter="tree_spec",
                    value=tree_spec,
                )
        elif isinstance(tree_spec, IRTreeSpec):
            self.tree_spec = deepcopy(tree_spec)
        else:
            raise MirtValidationError(
                "tree_spec must be a built-in name or IRTreeSpec",
                parameter="tree_spec",
                value=type(tree_spec),
            )

        if self.tree_spec.n_categories != self.n_categories:
            raise MirtValidationError(
                "tree_spec n_categories must match the model",
                parameter="n_categories",
                value=self.n_categories,
                expected=str(self.tree_spec.n_categories),
            )
        self.tree_spec.validate()

        self.n_traits = self.tree_spec.n_traits
        self.trait_names = list(self.tree_spec.trait_names)
        self._compile_tree()

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._trait_correlations: NDArray[np.float64] | None = None
        self._is_fitted = False

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize item parameters for each pseudo-item."""
        shape = (self.n_items, self.n_nodes)
        self._parameters["discrimination"] = np.ones(shape, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(shape, dtype=np.float64)

        if self.correlated_traits:
            self._trait_correlations = np.eye(self.n_traits, dtype=np.float64)

    def _compile_tree(self) -> None:
        """Compile stable node indices and category decision paths."""
        nodes: list[TreeNode] = []
        category_paths: list[list[tuple[int, int]] | None] = [
            None for _ in range(self.n_categories)
        ]
        reach_paths: list[list[tuple[int, int]]] = []

        def visit(node: TreeNode, prefix: list[tuple[int, int]]) -> None:
            node_idx = len(nodes)
            nodes.append(node)
            reach_paths.append(prefix.copy())

            for decision in (0, 1):
                child = node.children[decision]
                child_path = [*prefix, (node_idx, decision)]
                if isinstance(child, TreeNode):
                    visit(child, child_path)
                else:
                    category_paths[int(child)] = child_path

        visit(self.tree_spec.root, [])

        self._nodes = tuple(nodes)
        self._node_traits = np.asarray(
            [node.latent_trait for node in nodes], dtype=np.intp
        )
        self._path_decisions = np.full(
            (self.n_categories, len(nodes)), -1, dtype=np.int8
        )
        self._reach_decisions = np.full((len(nodes), len(nodes)), -1, dtype=np.int8)

        for category, path in enumerate(category_paths):
            assert path is not None
            for node_idx, decision in path:
                self._path_decisions[category, node_idx] = decision
        for target_node, path in enumerate(reach_paths):
            for node_idx, decision in path:
                self._reach_decisions[target_node, node_idx] = decision

    @property
    def n_nodes(self) -> int:
        """Number of binary decision nodes per item."""
        return len(self._nodes)

    @property
    def node_names(self) -> tuple[str, ...]:
        """Decision-node names in parameter-column order."""
        return tuple(node.name for node in self._nodes)

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._parameters.items()}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        n_parameters = sum(value.size for value in self._parameters.values())
        if self._trait_correlations is not None:
            n_parameters += self.n_traits * (self.n_traits - 1) // 2
        return n_parameters

    @property
    def trait_correlations(self) -> NDArray[np.float64] | None:
        if self._trait_correlations is None:
            return None
        return self._trait_correlations.copy()

    def expand_to_pseudo_items(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.bool_]]:
        """Expand ordinal responses to binary pseudo-items.

        Parameters
        ----------
        responses : NDArray
            Ordinal response matrix (n_persons, n_items) with values 0 to
            n_categories - 1. Negative values and NaN are treated as missing.

        Returns
        -------
        tuple
            - pseudo_responses: Binary responses (n_persons, n_items, n_nodes)
            - trait_assignments: Trait index for each pseudo-item (n_items, n_nodes)
            - valid_mask: Which pseudo-items are valid per person
        """
        response_codes, observed = self._validate_responses(responses)
        decisions = self._path_decisions[response_codes]
        valid_mask = observed[..., None] & (decisions >= 0)
        pseudo_responses = np.where(valid_mask, decisions, -1).astype(np.int32)
        trait_assignments = np.broadcast_to(
            self._node_traits, (self.n_items, self.n_nodes)
        ).astype(np.int32, copy=True)
        return pseudo_responses, trait_assignments, valid_mask

    def _validate_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            theta_array = np.asarray(theta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "theta must contain numeric values", parameter="theta"
            ) from exc
        if theta_array.ndim == 1:
            theta_array = theta_array.reshape(1, -1)
        if theta_array.ndim != 2:
            raise MirtValidationError(
                "theta must be one- or two-dimensional",
                parameter="theta",
                value=theta_array.ndim,
                expected="1 or 2 dimensions",
            )
        if theta_array.shape[1] != self.n_traits:
            raise MirtValidationError(
                "theta trait dimension does not match the model",
                parameter="theta",
                value=theta_array.shape,
                expected=f"(*, {self.n_traits})",
            )
        if not np.all(np.isfinite(theta_array)):
            raise MirtValidationError(
                "theta must contain only finite values", parameter="theta"
            )
        return theta_array

    def _validate_item_idx(self, item_idx: int) -> int:
        if (
            isinstance(item_idx, bool)
            or not isinstance(item_idx, Integral)
            or item_idx < 0
            or item_idx >= self.n_items
        ):
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return int(item_idx)

    def _validate_node_idx(self, node_idx: int) -> int:
        if (
            isinstance(node_idx, bool)
            or not isinstance(node_idx, Integral)
            or node_idx < 0
            or node_idx >= self.n_nodes
        ):
            raise IndexError(f"Node index {node_idx} out of range [0, {self.n_nodes})")
        return int(node_idx)

    def _validate_responses(
        self, responses: NDArray[np.int_]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        try:
            response_array = np.asarray(responses, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtDataError("responses must contain numeric values") from exc
        if response_array.ndim != 2:
            raise MirtDataError(
                "responses must be two-dimensional", ndim=response_array.ndim
            )
        if response_array.shape[1] != self.n_items:
            raise MirtDataError(
                "response item count does not match the model",
                n_items=response_array.shape[1],
                expected=self.n_items,
            )

        missing = np.isnan(response_array) | (response_array < 0)
        observed = ~missing
        observed_values = response_array[observed]
        if not np.all(np.isfinite(observed_values)):
            raise MirtDataError("observed responses must be finite")
        if np.any(observed_values != np.floor(observed_values)):
            raise MirtDataError("observed responses must be integer category codes")
        if np.any(observed_values >= self.n_categories):
            raise MirtDataError(f"observed responses must be below {self.n_categories}")

        codes = np.zeros(response_array.shape, dtype=np.intp)
        codes[observed] = observed_values.astype(np.intp)
        return codes, observed

    def _node_probabilities(
        self, theta: NDArray[np.float64], item_idx: int | None
    ) -> NDArray[np.float64]:
        if item_idx is None:
            discrimination = self._parameters["discrimination"]
            difficulty = self._parameters["difficulty"]
        else:
            index = self._validate_item_idx(item_idx)
            discrimination = self._parameters["discrimination"][index : index + 1]
            difficulty = self._parameters["difficulty"][index : index + 1]

        trait_values = theta[:, self._node_traits]
        logits = discrimination[None, :, :] * (
            trait_values[:, None, :] - difficulty[None, :, :]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            probabilities = np.asarray(sigmoid(logits), dtype=np.float64)
        return np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)

    @staticmethod
    def _combine_decision_probabilities(
        node_probabilities: NDArray[np.float64],
        decisions: NDArray[np.int8],
    ) -> NDArray[np.float64]:
        log_success = np.log(node_probabilities)
        log_failure = np.log1p(-node_probabilities)
        success_mask = (decisions == 1).astype(np.float64)
        failure_mask = (decisions == 0).astype(np.float64)
        log_probability = np.einsum(
            "pin,cn->pic", log_success, success_mask, optimize=True
        )
        log_probability += np.einsum(
            "pin,cn->pic", log_failure, failure_mask, optimize=True
        )
        return np.exp(log_probability)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities by traversing tree.

        Parameters
        ----------
        theta : NDArray
            Latent trait values (n_persons, n_traits) or (n_traits,)
        item_idx : int, optional
            If provided, compute for single item

        Returns
        -------
        NDArray
            Category probabilities (n_persons, n_categories) or
            (n_persons, n_items, n_categories)
        """
        theta_array = self._validate_theta(theta)
        node_probabilities = self._node_probabilities(theta_array, item_idx)
        probabilities = self._combine_decision_probabilities(
            node_probabilities, self._path_decisions
        )
        if item_idx is not None:
            return probabilities[:, 0, :]
        return probabilities

    def decision_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return conditional probabilities of decision 1 at every tree node."""
        theta_array = self._validate_theta(theta)
        probabilities = self._node_probabilities(theta_array, item_idx)
        if item_idx is not None:
            return probabilities[:, 0, :]
        return probabilities

    def reach_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return probabilities of reaching every tree node."""
        theta_array = self._validate_theta(theta)
        node_probabilities = self._node_probabilities(theta_array, item_idx)
        probabilities = self._combine_decision_probabilities(
            node_probabilities, self._reach_decisions
        )
        if item_idx is not None:
            return probabilities[:, 0, :]
        return probabilities

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected response-category scores."""
        probabilities = self.probability(theta, item_idx)
        return probabilities @ np.arange(self.n_categories, dtype=np.float64)

    def simulate(
        self,
        theta: NDArray[np.float64],
        seed: int | None = None,
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.int_]:
        """Simulate ordinal responses conditional on latent trait values.

        Parameters
        ----------
        theta : NDArray
            Latent trait values (n_persons, n_traits) or (n_traits,).
        seed : int, optional
            Random seed for reproducible response draws.
        chunk_size : int, optional
            Maximum number of persons evaluated at once. By default, a
            memory-bounded chunk size is selected from the model dimensions.

        Returns
        -------
        NDArray
            Integer category codes with shape (n_persons, n_items).

        Notes
        -----
        A fixed seed produces identical responses for every valid chunk size.
        """
        theta_array = self._validate_theta(theta)
        n_persons = theta_array.shape[0]
        if chunk_size is None:
            probability_values_per_person = self.n_items * self.n_categories
            chunk_size = max(
                1,
                min(
                    n_persons,
                    _IRTREE_MAX_PROBABILITY_VALUES // probability_values_per_person,
                ),
            )
        elif isinstance(chunk_size, bool) or not isinstance(
            chunk_size, (int, np.integer)
        ):
            raise ValueError("chunk_size must be a positive integer")
        elif chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        rng = np.random.default_rng(seed)
        responses = np.empty((n_persons, self.n_items), dtype=np.int32)
        for start in range(0, n_persons, int(chunk_size)):
            stop = min(start + int(chunk_size), n_persons)
            probabilities = self.probability(theta_array[start:stop])
            cumulative = np.cumsum(probabilities, axis=-1)
            cumulative[..., -1] = 1.0
            uniforms = rng.random(probabilities.shape[:-1])
            responses[start:stop] = np.sum(
                uniforms[..., None] > cumulative,
                axis=-1,
                dtype=np.int32,
            )
        return responses

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected Fisher information matrices for the traits.

        Returns one diagonal trait-information matrix per person and item.
        For a single item the item dimension is omitted.
        """
        theta_array = self._validate_theta(theta)
        node_probabilities = self._node_probabilities(theta_array, item_idx)
        reach = self._combine_decision_probabilities(
            node_probabilities, self._reach_decisions
        )

        if item_idx is None:
            discrimination = self._parameters["discrimination"]
        else:
            index = self._validate_item_idx(item_idx)
            discrimination = self._parameters["discrimination"][index : index + 1]

        node_information = (
            reach
            * node_probabilities
            * (1.0 - node_probabilities)
            * discrimination[None, :, :] ** 2
        )
        trait_selector = np.eye(self.n_traits, dtype=np.float64)[self._node_traits]
        diagonal = np.einsum(
            "pin,nt->pit", node_information, trait_selector, optimize=True
        )
        matrices = np.zeros((*diagonal.shape, self.n_traits), dtype=np.float64)
        trait_indices = np.arange(self.n_traits)
        matrices[..., trait_indices, trait_indices] = diagonal
        if item_idx is not None:
            return matrices[:, 0, :, :]
        return matrices

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood of responses given theta.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        theta : NDArray
            Latent trait values (n_persons, n_traits)

        Returns
        -------
        NDArray
            Log-likelihoods per person (n_persons,)
        """
        response_codes, observed = self._validate_responses(responses)
        theta_array = self._validate_theta(theta)
        if theta_array.shape[0] != response_codes.shape[0]:
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=theta_array.shape[0],
                response_persons=response_codes.shape[0],
            )

        probabilities = self.probability(theta_array)
        selected = np.take_along_axis(probabilities, response_codes[..., None], axis=2)[
            ..., 0
        ]
        contributions = np.where(
            observed,
            np.log(np.clip(selected, PROB_EPSILON, 1.0)),
            0.0,
        )
        return np.sum(contributions, axis=1)

    def _validate_trait_correlations(
        self, value: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self.correlated_traits:
            raise MirtValidationError(
                "trait correlations are disabled for this model",
                parameter="trait_correlations",
            )
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "trait_correlations must contain numeric values",
                parameter="trait_correlations",
            ) from exc
        expected_shape = (self.n_traits, self.n_traits)
        if array.shape != expected_shape:
            raise MirtValidationError(
                "trait_correlations has the wrong shape",
                parameter="trait_correlations",
                value=array.shape,
                expected=str(expected_shape),
            )
        if not np.all(np.isfinite(array)):
            raise MirtValidationError(
                "trait_correlations must be finite",
                parameter="trait_correlations",
            )
        if not np.allclose(array, array.T, atol=1e-10, rtol=0.0):
            raise MirtValidationError(
                "trait_correlations must be symmetric",
                parameter="trait_correlations",
            )
        if not np.allclose(np.diag(array), 1.0, atol=1e-10, rtol=0.0):
            raise MirtValidationError(
                "trait_correlations must have a unit diagonal",
                parameter="trait_correlations",
            )
        if np.min(np.linalg.eigvalsh(array)) < -1e-10:
            raise MirtValidationError(
                "trait_correlations must be positive semidefinite",
                parameter="trait_correlations",
            )
        return array.copy()

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Atomically set validated model parameters."""
        valid_names = {*self._parameters, "trait_correlations"}
        unknown = set(params) - valid_names
        if unknown:
            name = sorted(unknown)[0]
            raise MirtValidationError(
                f"Unknown parameter: {name}",
                parameter=name,
                expected=", ".join(sorted(valid_names)),
            )

        updated = {name: value.copy() for name, value in self._parameters.items()}
        correlations = (
            None
            if self._trait_correlations is None
            else self._trait_correlations.copy()
        )

        for name, value in params.items():
            if name == "trait_correlations":
                correlations = self._validate_trait_correlations(value)
                continue
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must contain numeric values", parameter=name
                ) from exc
            if array.shape != updated[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}",
                    parameter=name,
                    value=array.shape,
                    expected=str(updated[name].shape),
                )
            if not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"{name} must contain only finite values", parameter=name
                )
            if name == "discrimination" and np.any(array <= 0):
                raise MirtValidationError(
                    "discrimination values must be positive",
                    parameter=name,
                    expected="> 0",
                )
            updated[name] = array.copy()

        self._parameters = updated
        self._trait_correlations = correlations
        return self

    def get_item_parameters(self, item_idx: int) -> dict[str, NDArray[np.float64]]:
        """Get parameters for a specific item."""
        index = self._validate_item_idx(item_idx)
        return {name: value[index].copy() for name, value in self._parameters.items()}

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: Literal["discrimination", "difficulty"],
        value: NDArray[np.float64],
    ) -> None:
        """Set all node values for one item."""
        index = self._validate_item_idx(item_idx)
        if param_name not in self._parameters:
            raise MirtValidationError(
                f"Unknown item parameter: {param_name}", parameter=param_name
            )
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (self.n_nodes,):
            raise MirtValidationError(
                f"Shape mismatch for {param_name}",
                parameter=param_name,
                value=array.shape,
                expected=str((self.n_nodes,)),
            )
        updated = self._parameters[param_name].copy()
        updated[index] = array
        if param_name == "discrimination":
            self.set_parameters(discrimination=updated)
        else:
            self.set_parameters(difficulty=updated)

    def get_node_parameters(self, item_idx: int, node_idx: int) -> dict[str, float]:
        """Get discrimination and difficulty for one item decision node."""
        item = self._validate_item_idx(item_idx)
        node = self._validate_node_idx(node_idx)
        return {
            name: float(value[item, node]) for name, value in self._parameters.items()
        }

    def copy(self) -> Self:
        """Return an independent model copy with the same specification."""
        clone = self.__class__(
            n_items=self.n_items,
            tree_spec=self.tree_spec,
            n_categories=self.n_categories,
            item_names=self.item_names.copy(),
            correlated_traits=self.correlated_traits,
        )
        clone._parameters = {
            name: value.copy() for name, value in self._parameters.items()
        }
        clone._trait_correlations = (
            None
            if self._trait_correlations is None
            else self._trait_correlations.copy()
        )
        clone._is_fitted = self._is_fitted
        return clone

    def summary(self) -> str:
        """Generate model summary."""
        lines = []
        width = 70

        lines.append("=" * width)
        lines.append(f"{'IRTree Model Summary':^{width}}")
        lines.append("=" * width)

        lines.append(f"Tree Structure:     {self.tree_spec.name}")
        lines.append(f"Number of Items:    {self.n_items}")
        lines.append(f"Number of Categories: {self.n_categories}")
        lines.append(f"Number of Traits:   {self.n_traits}")
        lines.append(f"Decision Nodes:     {self.n_nodes}")
        lines.append(f"Trait Names:        {', '.join(self.trait_names)}")
        lines.append(f"Correlated Traits:  {self.correlated_traits}")
        lines.append(f"Fitted:             {self._is_fitted}")

        if self._is_fitted and self._trait_correlations is not None:
            lines.append("-" * width)
            lines.append("\nTrait Correlations:")
            header = "".ljust(15)
            for name in self.trait_names:
                header += f"{name:>12}"
            lines.append(header)
            for i, name in enumerate(self.trait_names):
                row = f"{name:<15}"
                for j in range(self.n_traits):
                    row += f"{self._trait_correlations[i, j]:>12.3f}"
                lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"IRTreeModel(n_items={self.n_items}, "
            f"tree_spec={self.tree_spec.name!r}, {status})"
        )
