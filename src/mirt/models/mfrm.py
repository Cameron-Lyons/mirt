"""Many-Facet Rasch Model (MFRM) for rating data.

This module provides:
- ManyFacetRaschModel for binary responses with multiple facets
- PolytomousMFRM for polytomous responses with rating scale or partial credit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class MFRMResult:
    """Results from MFRM estimation.

    Attributes
    ----------
    model : ManyFacetRaschModel
        Fitted model.
    facet_parameters : dict
        Estimated facet parameters.
    facet_se : dict
        Standard errors for facet parameters.
    infit : dict
        Infit statistics per facet level.
    outfit : dict
        Outfit statistics per facet level.
    log_likelihood : float
        Log-likelihood at convergence.
    n_iterations : int
        Number of iterations.
    converged : bool
        Whether estimation converged.
    """

    model: ManyFacetRaschModel
    facet_parameters: dict[str, NDArray[np.float64]]
    facet_se: dict[str, NDArray[np.float64]]
    infit: dict[str, NDArray[np.float64]]
    outfit: dict[str, NDArray[np.float64]]
    log_likelihood: float
    n_iterations: int
    converged: bool


@dataclass
class Facet:
    """Definition of a facet in MFRM.

    Attributes
    ----------
    name : str
        Name of the facet (e.g., 'rater', 'task', 'criterion').
    n_levels : int
        Number of levels in this facet.
    labels : list of str
        Labels for each level.
    is_anchored : bool
        Whether this facet is anchored (e.g., sum constrained to 0).
    anchor_value : float
        Value to anchor (typically 0 for centering).
    """

    name: str
    n_levels: int
    labels: list[str] | None = None
    is_anchored: bool = True
    anchor_value: float = 0.0

    def __post_init__(self):
        if self.labels is None:
            self.labels = [f"{self.name}_{i}" for i in range(self.n_levels)]
        if len(self.labels) != self.n_levels:
            raise ValueError(
                f"labels length ({len(self.labels)}) must match n_levels ({self.n_levels})"
            )


class ManyFacetRaschModel:
    """Many-Facet Rasch Model for binary responses.

    The MFRM extends the Rasch model to include multiple facets beyond
    persons and items. Common facets include raters, tasks, and criteria.

    Parameters
    ----------
    n_items : int
        Number of items.
    facets : list of Facet
        List of facets (excluding persons and items which are implicit).
    item_names : list of str, optional
        Names for items.

    Attributes
    ----------
    item_difficulty : ndarray of shape (n_items,)
        Item difficulty parameters.
    facet_parameters : dict
        Parameters for each facet (severity/leniency).

    Notes
    -----
    The model is:

        logit(P(X=1)) = θ - b_i - Σ_f d_f

    where θ is person ability, b_i is item difficulty, and d_f are
    facet parameters (e.g., rater severity).

    References
    ----------
    Linacre, J. M. (1994). Many-Facet Rasch Measurement.
        Chicago: MESA Press.
    """

    model_name = "MFRM"

    def __init__(
        self,
        n_items: int,
        facets: list[Facet],
        item_names: list[str] | None = None,
    ) -> None:
        if n_items < 1:
            raise ValueError("n_items must be at least 1")

        self._n_items = n_items
        self._facets = list(facets)
        self._item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if len(self._item_names) != n_items:
            raise ValueError(
                f"item_names length ({len(self._item_names)}) must match n_items ({n_items})"
            )

        self._item_difficulty = np.zeros(n_items)
        self._facet_parameters: dict[str, NDArray[np.float64]] = {}
        for facet in self._facets:
            self._facet_parameters[facet.name] = np.zeros(facet.n_levels)

        self._is_fitted = False

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def n_facets(self) -> int:
        return len(self._facets)

    @property
    def facet_names(self) -> list[str]:
        return [f.name for f in self._facets]

    @property
    def facets(self) -> list[Facet]:
        return list(self._facets)

    @property
    def item_names(self) -> list[str]:
        return self._item_names.copy()

    @property
    def item_difficulty(self) -> NDArray[np.float64]:
        return self._item_difficulty.copy()

    @property
    def facet_parameters(self) -> dict[str, NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._facet_parameters.items()}

    def get_facet(self, name: str) -> Facet:
        """Get facet by name."""
        for facet in self._facets:
            if facet.name == name:
                return facet
        raise ValueError(f"Unknown facet: {name}")

    def set_item_difficulty(self, difficulty: NDArray[np.float64]) -> Self:
        difficulty = np.asarray(difficulty, dtype=np.float64)
        if difficulty.shape != (self._n_items,):
            raise ValueError(
                f"difficulty shape {difficulty.shape} != ({self._n_items},)"
            )
        self._item_difficulty = difficulty
        return self

    def set_facet_parameters(
        self, facet_name: str, parameters: NDArray[np.float64]
    ) -> Self:
        if facet_name not in self._facet_parameters:
            raise ValueError(f"Unknown facet: {facet_name}")

        parameters = np.asarray(parameters, dtype=np.float64)
        facet = self.get_facet(facet_name)
        if parameters.shape != (facet.n_levels,):
            raise ValueError(
                f"parameters shape {parameters.shape} != ({facet.n_levels},)"
            )

        if facet.is_anchored:
            parameters = parameters - np.mean(parameters) + facet.anchor_value

        self._facet_parameters[facet_name] = parameters
        return self

    def log_odds(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute log-odds for a response.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int
            Item index.
        facet_indices : dict
            Index for each facet (e.g., {'rater': 2, 'task': 0}).

        Returns
        -------
        ndarray
            Log-odds of correct response.
        """
        theta = np.asarray(theta).ravel()
        log_odds = theta - self._item_difficulty[item_idx]

        for facet_name, facet_idx in facet_indices.items():
            if facet_name in self._facet_parameters:
                log_odds = log_odds - self._facet_parameters[facet_name][facet_idx]

        return log_odds

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute response probability.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int
            Item index.
        facet_indices : dict
            Index for each facet.

        Returns
        -------
        ndarray
            Probability of correct response.
        """
        z = self.log_odds(theta, item_idx, facet_indices)
        return 1.0 / (1.0 + np.exp(-z))

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        p = self.probability(theta, item_idx, facet_indices)
        return p * (1.0 - p)

    def copy(self) -> Self:
        new_model = ManyFacetRaschModel(
            n_items=self._n_items,
            facets=[
                Facet(
                    name=f.name,
                    n_levels=f.n_levels,
                    labels=f.labels.copy() if f.labels else None,
                    is_anchored=f.is_anchored,
                    anchor_value=f.anchor_value,
                )
                for f in self._facets
            ],
            item_names=self._item_names.copy(),
        )
        new_model._item_difficulty = self._item_difficulty.copy()
        new_model._facet_parameters = {
            k: v.copy() for k, v in self._facet_parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class PolytomousMFRM(ManyFacetRaschModel):
    """Many-Facet Rasch Model for polytomous responses.

    Extends MFRM to handle ordinal responses using either rating scale
    or partial credit structure.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int
        Number of response categories (same for all items in rating scale).
    facets : list of Facet
        List of facets.
    category_structure : str
        Either 'rating_scale' (shared thresholds) or 'partial_credit'
        (item-specific thresholds).
    item_names : list of str, optional
        Names for items.

    Attributes
    ----------
    thresholds : ndarray
        Category thresholds. For rating_scale, shape is (n_categories-1,).
        For partial_credit, shape is (n_items, n_categories-1).
    """

    model_name = "PolytomousMFRM"

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        facets: list[Facet],
        category_structure: Literal["rating_scale", "partial_credit"] = "rating_scale",
        item_names: list[str] | None = None,
    ) -> None:
        super().__init__(n_items=n_items, facets=facets, item_names=item_names)

        if n_categories < 2:
            raise ValueError("n_categories must be at least 2")

        self._n_categories = n_categories
        self._category_structure = category_structure

        if category_structure == "rating_scale":
            self._thresholds = np.linspace(-1, 1, n_categories - 1)
        else:
            self._thresholds = np.zeros((n_items, n_categories - 1))

    @property
    def n_categories(self) -> int:
        return self._n_categories

    @property
    def category_structure(self) -> str:
        return self._category_structure

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._thresholds.copy()

    def set_thresholds(self, thresholds: NDArray[np.float64]) -> Self:
        thresholds = np.asarray(thresholds, dtype=np.float64)

        if self._category_structure == "rating_scale":
            expected_shape = (self._n_categories - 1,)
        else:
            expected_shape = (self._n_items, self._n_categories - 1)

        if thresholds.shape != expected_shape:
            raise ValueError(f"thresholds shape {thresholds.shape} != {expected_shape}")

        self._thresholds = thresholds
        return self

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute probability of response in a specific category.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int
            Item index.
        category : int
            Response category (0 to n_categories-1).
        facet_indices : dict
            Index for each facet.

        Returns
        -------
        ndarray
            Category probability.
        """
        theta = np.asarray(theta).ravel()
        n_persons = len(theta)

        base_measure = theta - self._item_difficulty[item_idx]
        for facet_name, facet_idx in facet_indices.items():
            if facet_name in self._facet_parameters:
                base_measure = (
                    base_measure - self._facet_parameters[facet_name][facet_idx]
                )

        if self._category_structure == "rating_scale":
            tau = self._thresholds
        else:
            tau = self._thresholds[item_idx]

        exp_terms = np.zeros((n_persons, self._n_categories))
        for k in range(self._n_categories):
            if k == 0:
                exp_terms[:, k] = 1.0
            else:
                cumsum_tau = np.sum(tau[:k])
                exp_terms[:, k] = np.exp(k * base_measure - cumsum_tau)

        denom = exp_terms.sum(axis=1)
        probs = exp_terms[:, category] / denom

        return probs

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute all category probabilities.

        Returns
        -------
        ndarray of shape (n_persons, n_categories)
            Category probabilities.
        """
        theta = np.asarray(theta).ravel()
        n_persons = len(theta)

        probs = np.zeros((n_persons, self._n_categories))
        for k in range(self._n_categories):
            probs[:, k] = self.category_probability(theta, item_idx, k, facet_indices)

        return probs

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        facet_indices: dict[str, int],
    ) -> NDArray[np.float64]:
        """Compute expected response score."""
        probs = self.probability(theta, item_idx, facet_indices)
        categories = np.arange(self._n_categories)
        return np.sum(probs * categories[None, :], axis=1)

    def copy(self) -> Self:
        new_model = PolytomousMFRM(
            n_items=self._n_items,
            n_categories=self._n_categories,
            facets=[
                Facet(
                    name=f.name,
                    n_levels=f.n_levels,
                    labels=f.labels.copy() if f.labels else None,
                    is_anchored=f.is_anchored,
                    anchor_value=f.anchor_value,
                )
                for f in self._facets
            ],
            category_structure=self._category_structure,
            item_names=self._item_names.copy(),
        )
        new_model._item_difficulty = self._item_difficulty.copy()
        new_model._facet_parameters = {
            k: v.copy() for k, v in self._facet_parameters.items()
        }
        new_model._thresholds = self._thresholds.copy()
        new_model._is_fitted = self._is_fitted
        return new_model
