"""Many-Facet Rasch Model (MFRM) for rating data.

This module provides:
- ManyFacetRaschModel for binary responses with multiple facets
- PolytomousMFRM for polytomous responses with rating scale or partial credit
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._core import sigmoid


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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("facet name must not be empty")
        if (
            isinstance(self.n_levels, (bool, np.bool_))
            or not isinstance(self.n_levels, (int, np.integer))
            or self.n_levels < 1
        ):
            raise ValueError("n_levels must be a positive integer")
        self.n_levels = int(self.n_levels)
        if isinstance(self.anchor_value, (bool, np.bool_)):
            raise ValueError("anchor_value must be finite")
        try:
            self.anchor_value = float(self.anchor_value)
        except (TypeError, ValueError) as error:
            raise ValueError("anchor_value must be finite") from error
        if not np.isfinite(self.anchor_value):
            raise ValueError("anchor_value must be finite")

        if self.labels is None:
            self.labels = [f"{self.name}_{i}" for i in range(self.n_levels)]
        else:
            self.labels = list(self.labels)
        if len(self.labels) != self.n_levels:
            raise ValueError(
                f"labels length ({len(self.labels)}) must match n_levels ({self.n_levels})"
            )
        if any(not isinstance(label, str) or not label for label in self.labels):
            raise ValueError("facet labels must be non-empty strings")
        if len(set(self.labels)) != self.n_levels:
            raise ValueError("facet labels must be unique")


def _copy_facet(facet: Facet) -> Facet:
    """Return an independent copy of a validated facet definition."""
    return Facet(
        name=facet.name,
        n_levels=facet.n_levels,
        labels=facet.labels.copy() if facet.labels is not None else None,
        is_anchored=facet.is_anchored,
        anchor_value=facet.anchor_value,
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
        if (
            isinstance(n_items, (bool, np.bool_))
            or not isinstance(n_items, (int, np.integer))
            or n_items < 1
        ):
            raise ValueError("n_items must be at least 1 and an integer")

        self._n_items = int(n_items)
        facet_definitions = list(facets)
        if any(not isinstance(facet, Facet) for facet in facet_definitions):
            raise TypeError("facets must contain Facet instances")
        self._facets = [_copy_facet(facet) for facet in facet_definitions]
        facet_names = [facet.name for facet in self._facets]
        if len(set(facet_names)) != len(facet_names):
            raise ValueError("facet names must be unique")

        self._item_names = (
            list(item_names)
            if item_names is not None
            else [f"Item_{i}" for i in range(self._n_items)]
        )

        if len(self._item_names) != self._n_items:
            raise ValueError(
                f"item_names length ({len(self._item_names)}) must match "
                f"n_items ({self._n_items})"
            )

        self._item_difficulty = np.zeros(self._n_items)
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
        return [_copy_facet(facet) for facet in self._facets]

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
                return _copy_facet(facet)
        raise ValueError(f"Unknown facet: {name}")

    def set_item_difficulty(self, difficulty: NDArray[np.float64]) -> Self:
        difficulty = np.asarray(difficulty, dtype=np.float64)
        if difficulty.shape != (self._n_items,):
            raise ValueError(
                f"difficulty shape {difficulty.shape} != ({self._n_items},)"
            )
        if not np.all(np.isfinite(difficulty)):
            raise ValueError("difficulty values must be finite")
        self._item_difficulty = difficulty.copy()
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
        if not np.all(np.isfinite(parameters)):
            raise ValueError("facet parameters must be finite")

        if facet.is_anchored:
            parameters = parameters - np.mean(parameters) + facet.anchor_value

        self._facet_parameters[facet_name] = parameters.copy()
        return self

    @staticmethod
    def _prepare_theta(theta: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(theta, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise ValueError("theta must be a scalar or one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("theta values must be finite")
        return values

    def _validate_item_idx(self, item_idx: int) -> int:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
            item_idx, (int, np.integer)
        ):
            raise TypeError("item_idx must be an integer")
        index = int(item_idx)
        if index < 0 or index >= self._n_items:
            raise IndexError(f"item_idx {index} out of range [0, {self._n_items})")
        return index

    def _facet_effect(
        self,
        n_persons: int,
        item_idx: int | None,
        facet_indices: Mapping[str, ArrayLike] | None,
    ) -> NDArray[np.float64]:
        """Validate facet assignments and return their combined severity."""
        provided = set(facet_indices or {})
        expected = set(self.facet_names)
        unknown = sorted(provided - expected)
        missing = sorted(expected - provided)
        if unknown:
            raise ValueError(f"Unknown facet assignments: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"Missing facet assignments: {', '.join(missing)}")

        target_shape = (n_persons, self._n_items) if item_idx is None else (n_persons,)
        effect = np.zeros(target_shape, dtype=np.float64)
        if not expected:
            return effect

        assert facet_indices is not None
        for facet in self._facets:
            indices = np.asarray(facet_indices[facet.name])
            if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
                indices.dtype, np.bool_
            ):
                raise TypeError(f"facet '{facet.name}' indices must be integers")

            if indices.ndim == 0:
                prepared_indices = indices
            elif item_idx is None and indices.shape == (n_persons,):
                prepared_indices = indices[:, None]
            elif item_idx is None and indices.shape == (
                n_persons,
                self._n_items,
            ):
                prepared_indices = indices
            elif item_idx is not None and indices.shape == (n_persons,):
                prepared_indices = indices
            elif item_idx is not None and indices.shape == (
                n_persons,
                self._n_items,
            ):
                prepared_indices = indices[:, item_idx]
            else:
                expected_shapes = (
                    f"scalar, ({n_persons},), or ({n_persons}, {self._n_items})"
                )
                raise ValueError(
                    f"facet '{facet.name}' indices must have shape {expected_shapes}; "
                    f"got {indices.shape}"
                )

            if np.any((prepared_indices < 0) | (prepared_indices >= facet.n_levels)):
                raise IndexError(
                    f"facet '{facet.name}' index out of range [0, {facet.n_levels})"
                )
            effect += self._facet_parameters[facet.name][prepared_indices]

        return effect

    def log_odds(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute log-odds for a response.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int, optional
            Item index. If omitted, return log-odds for every item.
        facet_indices : mapping
            Assignment for every facet. Each value may be a scalar, a
            person-level vector, or a person-by-item matrix.

        Returns
        -------
        ndarray
            Log-odds of correct response, with shape ``(n_persons,)`` for a
            single item or ``(n_persons, n_items)`` for all items.
        """
        theta_values = self._prepare_theta(theta)
        validated_item_idx = (
            self._validate_item_idx(item_idx) if item_idx is not None else None
        )
        facet_effect = self._facet_effect(
            len(theta_values),
            validated_item_idx,
            facet_indices,
        )

        if validated_item_idx is not None:
            return (
                theta_values - self._item_difficulty[validated_item_idx] - facet_effect
            )
        return theta_values[:, None] - self._item_difficulty[None, :] - facet_effect

    def probability(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int, optional
            Item index. If omitted, return probabilities for every item.
        facet_indices : mapping
            Scalar or batched assignment for every facet.

        Returns
        -------
        ndarray
            Probability of correct response, with shape ``(n_persons,)`` for
            one item or ``(n_persons, n_items)`` for all items.
        """
        z = self.log_odds(theta, item_idx, facet_indices)
        return sigmoid(z)

    def information(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        p = self.probability(theta, item_idx, facet_indices)
        return p * (1.0 - p)

    def test_information(
        self,
        theta: ArrayLike,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Return test information summed across all items."""
        return np.sum(self.information(theta, None, facet_indices), axis=1)

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
        if (
            isinstance(n_categories, (bool, np.bool_))
            or not isinstance(n_categories, (int, np.integer))
            or n_categories < 2
        ):
            raise ValueError("n_categories must be at least 2")
        if category_structure not in {"rating_scale", "partial_credit"}:
            raise ValueError(
                "category_structure must be 'rating_scale' or 'partial_credit'"
            )

        super().__init__(n_items=n_items, facets=facets, item_names=item_names)

        self._n_categories = int(n_categories)
        self._category_structure = category_structure

        if category_structure == "rating_scale":
            self._thresholds = np.linspace(-1, 1, self._n_categories - 1)
        else:
            self._thresholds = np.zeros((self._n_items, self._n_categories - 1))

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
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("threshold values must be finite")

        self._thresholds = thresholds.copy()
        return self

    def _validate_category(self, category: int) -> int:
        if isinstance(category, (bool, np.bool_)) or not isinstance(
            category, (int, np.integer)
        ):
            raise TypeError("category must be an integer")
        value = int(category)
        if value < 0 or value >= self._n_categories:
            raise IndexError(f"category {value} out of range [0, {self._n_categories})")
        return value

    def _category_logits(
        self,
        theta: ArrayLike,
        item_idx: int | None,
        facet_indices: Mapping[str, ArrayLike] | None,
    ) -> NDArray[np.float64]:
        base_measure = self.log_odds(theta, item_idx, facet_indices)
        categories = np.arange(self._n_categories, dtype=np.float64)

        if item_idx is not None:
            validated_item_idx = self._validate_item_idx(item_idx)
            thresholds = (
                self._thresholds
                if self._category_structure == "rating_scale"
                else self._thresholds[validated_item_idx]
            )
            cumulative_thresholds = np.concatenate(([0.0], np.cumsum(thresholds)))
            return (
                base_measure[:, None] * categories[None, :]
                - cumulative_thresholds[None, :]
            )

        if self._category_structure == "rating_scale":
            cumulative_thresholds = np.concatenate(([0.0], np.cumsum(self._thresholds)))
            return (
                base_measure[:, :, None] * categories[None, None, :]
                - cumulative_thresholds[None, None, :]
            )

        cumulative_thresholds = np.concatenate(
            (
                np.zeros((self._n_items, 1), dtype=np.float64),
                np.cumsum(self._thresholds, axis=1),
            ),
            axis=1,
        )
        return (
            base_measure[:, :, None] * categories[None, None, :]
            - cumulative_thresholds[None, :, :]
        )

    @staticmethod
    def _normalized_exponentials(
        logits: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute a stable softmax along the response-category axis."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        numerators = np.exp(shifted)
        return numerators / np.sum(numerators, axis=-1, keepdims=True)

    def category_probability(
        self,
        theta: ArrayLike,
        item_idx: int | None,
        category: int,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute probability of response in a specific category.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int, optional
            Item index. If omitted, return the category probability for every
            item.
        category : int
            Response category (0 to n_categories-1).
        facet_indices : dict
            Index for each facet.

        Returns
        -------
        ndarray
            Category probability.
        """
        validated_category = self._validate_category(category)
        return self.probability(theta, item_idx, facet_indices)[..., validated_category]

    def probability(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute all category probabilities.

        Returns
        -------
        ndarray
            Category probabilities, with shape ``(n_persons, n_categories)``
            for one item or ``(n_persons, n_items, n_categories)`` for all
            items.
        """
        logits = self._category_logits(theta, item_idx, facet_indices)
        return self._normalized_exponentials(logits)

    def expected_score(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected response score."""
        probs = self.probability(theta, item_idx, facet_indices)
        categories = np.arange(self._n_categories, dtype=np.float64)
        return np.sum(probs * categories, axis=-1)

    def information(
        self,
        theta: ArrayLike,
        item_idx: int | None = None,
        facet_indices: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]:
        """Return Fisher information for the latent measure.

        For the rating-scale and partial-credit forms this is the variance of
        the response-category score under the model.
        """
        probabilities = self.probability(theta, item_idx, facet_indices)
        categories = np.arange(self._n_categories, dtype=np.float64)
        expected = np.sum(probabilities * categories, axis=-1, keepdims=True)
        return np.sum(
            probabilities * (categories - expected) ** 2,
            axis=-1,
        )

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
