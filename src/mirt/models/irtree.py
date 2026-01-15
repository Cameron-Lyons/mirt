from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    pass


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
        2. Direction (0 = low side, 1 = high side)
        3. Midpoint (0 = not midpoint, 1 = midpoint) - only if non-extreme

        5-point mapping:
        - 1: Extreme=1, Direction=0
        - 2: Extreme=0, Mid=0, Direction=0
        - 3: Extreme=0, Mid=1
        - 4: Extreme=0, Mid=0, Direction=1
        - 5: Extreme=1, Direction=1
        """
        if n_categories != 5:
            raise ValueError("Extreme-midpoint model requires 5 categories")

        midpoint_low = TreeNode(
            name="midpoint_low",
            latent_trait=2,
            children={0: 1, 1: 2},
        )

        midpoint_high = TreeNode(
            name="midpoint_high",
            latent_trait=2,
            children={0: 3, 1: 2},
        )

        direction_nonextreme = TreeNode(
            name="direction_nonextreme",
            latent_trait=1,
            children={0: midpoint_low, 1: midpoint_high},
        )

        direction_extreme = TreeNode(
            name="direction_extreme",
            latent_trait=1,
            children={0: 0, 1: 4},
        )

        root = TreeNode(
            name="extreme",
            latent_trait=0,
            children={0: direction_nonextreme, 1: direction_extreme},
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
        1. Direction (0 = disagree, 1 = agree)
        2. Intensity (0, 1, 2 for mild/moderate/extreme)

        5-point mapping:
        - 1: D=0, I=2 (strong disagree)
        - 2: D=0, I=1 (disagree)
        - 3: neutral (both directions possible with I=0)
        - 4: D=1, I=1 (agree)
        - 5: D=1, I=2 (strong agree)
        """
        if n_categories != 5:
            raise ValueError("Direction-intensity model requires 5 categories")

        intensity_low = TreeNode(
            name="intensity_low",
            latent_trait=1,
            children={0: 2, 1: 1, 2: 0},
        )

        intensity_high = TreeNode(
            name="intensity_high",
            latent_trait=1,
            children={0: 2, 1: 3, 2: 4},
        )

        root = TreeNode(
            name="direction",
            latent_trait=0,
            children={0: intensity_low, 1: intensity_high},
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
        path: list[tuple[TreeNode, int]] = []
        self._find_path(self.root, category, path)
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
        self.n_items = n_items
        self.n_categories = n_categories
        self.correlated_traits = correlated_traits
        self.item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if isinstance(tree_spec, str):
            if tree_spec == "bockenholt":
                self.tree_spec = IRTreeSpec.bockenholt_adi(n_categories)
            elif tree_spec == "extreme_midpoint":
                self.tree_spec = IRTreeSpec.extreme_midpoint(n_categories)
            elif tree_spec == "direction_intensity":
                self.tree_spec = IRTreeSpec.simple_direction_intensity(n_categories)
            else:
                raise ValueError(f"Unknown tree spec: {tree_spec}")
        else:
            self.tree_spec = tree_spec

        self.n_traits = self.tree_spec.n_traits
        self.trait_names = self.tree_spec.trait_names

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._trait_correlations: NDArray[np.float64] | None = None
        self._is_fitted = False

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize item parameters for each pseudo-item."""
        n_nodes = self.tree_spec.root.count_nodes()

        self._parameters["discrimination"] = np.ones((self.n_items, n_nodes))
        self._parameters["difficulty"] = np.zeros((self.n_items, n_nodes))

        if self.correlated_traits:
            self._trait_correlations = np.eye(self.n_traits)

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._parameters.items()}

    @property
    def trait_correlations(self) -> NDArray[np.float64] | None:
        if self._trait_correlations is None:
            return None
        return self._trait_correlations.copy()

    def expand_to_pseudo_items(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        """Expand ordinal responses to binary pseudo-items.

        Parameters
        ----------
        responses : NDArray
            Ordinal response matrix (n_persons, n_items) with values 0 to n_categories-1

        Returns
        -------
        tuple
            - pseudo_responses: Binary responses (n_persons, n_items, max_nodes)
            - trait_assignments: Trait index for each pseudo-item (n_items, max_nodes)
            - valid_mask: Which pseudo-items are valid per person (n_persons, n_items, max_nodes)
        """

        n_persons, n_items = responses.shape
        max_depth = self.tree_spec.root.depth()

        pseudo_responses = np.full((n_persons, n_items, max_depth), -1, dtype=np.int32)
        trait_assignments = np.zeros((n_items, max_depth), dtype=np.int32)
        valid_mask = np.zeros((n_persons, n_items, max_depth), dtype=np.bool_)

        for j in range(n_items):
            node_idx = 0

            def assign_node_traits(node: TreeNode, depth: int) -> None:
                nonlocal node_idx
                if depth < max_depth:
                    trait_assignments[j, node_idx] = node.latent_trait
                    node_idx += 1
                    for child in node.children.values():
                        if isinstance(child, TreeNode):
                            assign_node_traits(child, depth + 1)

            assign_node_traits(self.tree_spec.root, 0)

        for i in range(n_persons):
            for j in range(n_items):
                resp = responses[i, j]
                if resp < 0:
                    continue

                path = self.tree_spec.get_path_to_category(int(resp))

                for node_idx, (node, decision) in enumerate(path):
                    if node_idx < max_depth:
                        pseudo_responses[i, j, node_idx] = decision
                        valid_mask[i, j, node_idx] = True

        return pseudo_responses, trait_assignments, valid_mask

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
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)

        n_persons = theta.shape[0]

        if item_idx is not None:
            probs = np.zeros((n_persons, self.n_categories))
            self._compute_category_probs(theta, item_idx, probs)
            return probs

        all_probs = np.zeros((n_persons, self.n_items, self.n_categories))
        for j in range(self.n_items):
            self._compute_category_probs(theta, j, all_probs[:, j, :])
        return all_probs

    def _compute_category_probs(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        probs: NDArray[np.float64],
    ) -> None:
        """Compute category probabilities for a single item."""
        n_persons = theta.shape[0]

        for cat in range(self.n_categories):
            path = self.tree_spec.get_path_to_category(cat)
            cat_prob = np.ones(n_persons)

            node_idx = 0
            for node, decision in path:
                trait_idx = node.latent_trait
                a = self._parameters["discrimination"][item_idx, node_idx]
                b = self._parameters["difficulty"][item_idx, node_idx]

                z = a * (theta[:, trait_idx] - b)
                p_node = sigmoid(z)
                p_node = np.clip(p_node, PROB_EPSILON, 1 - PROB_EPSILON)

                if decision == 1:
                    cat_prob *= p_node
                else:
                    cat_prob *= 1 - p_node

                node_idx += 1

            probs[:, cat] = cat_prob

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
        n_persons = responses.shape[0]
        probs = self.probability(theta)

        ll = np.zeros(n_persons)
        for i in range(n_persons):
            for j in range(self.n_items):
                resp = responses[i, j]
                if resp >= 0:
                    p = probs[i, j, resp]
                    ll[i] += np.log(np.clip(p, PROB_EPSILON, 1.0))

        return ll

    def set_parameters(self, **params: NDArray[np.float64]) -> IRTreeModel:
        """Set model parameters."""
        for name, value in params.items():
            if name in self._parameters:
                self._parameters[name] = np.asarray(value)
            elif name == "trait_correlations":
                self._trait_correlations = np.asarray(value)
            else:
                raise ValueError(f"Unknown parameter: {name}")
        return self

    def get_item_parameters(self, item_idx: int) -> dict[str, NDArray[np.float64]]:
        """Get parameters for a specific item."""
        return {name: value[item_idx] for name, value in self._parameters.items()}

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
