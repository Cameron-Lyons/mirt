"""Abstract base classes for IRT item models."""

from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray


class BaseItemModel(ABC):
    """Abstract base class for all IRT item models.

    This class defines the interface that all IRT models must implement.
    It provides the foundation for computing item response probabilities,
    log-likelihoods, and information functions.

    Parameters
    ----------
    n_items : int
        Number of items in the test/scale.
    n_factors : int, default=1
        Number of latent factors (dimensions).
    item_names : list of str, optional
        Names for each item. If not provided, items are named "Item_0", "Item_1", etc.

    Attributes
    ----------
    n_items : int
        Number of items.
    n_factors : int
        Number of latent factors.
    item_names : list of str
        Names of items.
    is_fitted : bool
        Whether the model has been fitted.
    """

    # Class attributes to be overridden by subclasses
    model_name: str = "BaseModel"
    n_params_per_item: int = 0
    supports_multidimensional: bool = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        if n_items <= 0:
            raise ValueError("n_items must be positive")
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        if n_factors > 1 and not self.supports_multidimensional:
            raise ValueError(
                f"{self.model_name} does not support multidimensional models"
            )

        self.n_items = n_items
        self.n_factors = n_factors
        self.item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if len(self.item_names) != n_items:
            raise ValueError(
                f"Length of item_names ({len(self.item_names)}) must match n_items ({n_items})"
            )

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._is_fitted: bool = False
        self._initialize_parameters()

    @abstractmethod
    def _initialize_parameters(self) -> None:
        """Initialize model parameters with reasonable starting values.

        This method should populate self._parameters with initial values
        for all model parameters.
        """
        ...

    @abstractmethod
    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        For dichotomous models, returns P(X=1|theta).
        For polytomous models, returns P(X=k|theta) for all categories.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int, optional
            If specified, compute probability for this item only.
            If None, compute for all items.

        Returns
        -------
        ndarray
            Response probabilities. Shape depends on model type and item_idx.
        """
        ...

    @abstractmethod
    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for given responses and theta.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix. Missing values should be coded as -1 or NaN.
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.

        Returns
        -------
        ndarray of shape (n_persons,)
            Log-likelihood for each person.
        """
        ...

    @abstractmethod
    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information at given theta values.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int, optional
            If specified, compute information for this item only.
            If None, compute total test information.

        Returns
        -------
        ndarray
            Fisher information values.
        """
        ...

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        """Return a copy of current model parameters.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their values.
        """
        return {k: v.copy() for k, v in self._parameters.items()}

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters in the model."""
        return sum(p.size for p in self._parameters.values())

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set model parameters.

        Parameters
        ----------
        **params : ndarray
            Parameter arrays to set. Names must match existing parameters.

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If an unknown parameter name is provided.
        """
        for name, value in params.items():
            if name not in self._parameters:
                valid_params = ", ".join(self._parameters.keys())
                raise ValueError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}"
                )
            value_arr = np.asarray(value, dtype=np.float64)
            if value_arr.shape != self._parameters[name].shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {self._parameters[name].shape}, "
                    f"got {value_arr.shape}"
                )
            self._parameters[name] = value_arr
        return self

    def get_item_parameters(self, item_idx: int) -> dict[str, float | NDArray[np.float64]]:
        """Get parameters for a specific item.

        Parameters
        ----------
        item_idx : int
            Index of the item.

        Returns
        -------
        dict
            Dictionary of parameter values for the item.
        """
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")

        result: dict[str, float | NDArray[np.float64]] = {}
        for name, values in self._parameters.items():
            if values.ndim == 1 and len(values) == self.n_items:
                result[name] = float(values[item_idx])
            elif values.ndim == 2 and values.shape[0] == self.n_items:
                result[name] = values[item_idx].copy()
            else:
                result[name] = values.copy()
        return result

    def _ensure_theta_2d(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ensure theta is 2D array of shape (n_persons, n_factors).

        Parameters
        ----------
        theta : ndarray
            Theta values, can be 1D or 2D.

        Returns
        -------
        ndarray of shape (n_persons, n_factors)
            2D theta array.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2:
            raise ValueError(f"theta must be 1D or 2D, got {theta.ndim}D")
        if theta.shape[1] != self.n_factors:
            raise ValueError(
                f"theta has {theta.shape[1]} factors, expected {self.n_factors}"
            )
        return theta

    def copy(self) -> Self:
        """Create a copy of the model with the same parameters.

        Returns
        -------
        BaseItemModel
            A new model instance with copied parameters.
        """
        new_model = self.__class__(
            n_items=self.n_items,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._is_fitted = self._is_fitted
        return new_model

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"n_items={self.n_items}, "
            f"n_factors={self.n_factors}, "
            f"{status})"
        )


class DichotomousItemModel(BaseItemModel):
    """Base class for dichotomous (0/1) response models.

    Dichotomous models are used for binary response data (e.g., correct/incorrect,
    yes/no). Subclasses include 1PL, 2PL, 3PL, and 4PL models.
    """

    def icc(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute Item Characteristic Curve: P(X=1|theta).

        This is an alias for probability() with a specific item index.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int
            Index of the item.

        Returns
        -------
        ndarray of shape (n_persons,)
            Probability of correct response at each theta.
        """
        return self.probability(theta, item_idx)

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute expected item or test score.

        For dichotomous items, E[X|theta] = P(X=1|theta).

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int, optional
            If specified, return expected score for one item.
            If None, return sum of expected scores (expected test score).

        Returns
        -------
        ndarray
            Expected scores.
        """
        probs = self.probability(theta, item_idx)
        if item_idx is None:
            # Sum across items for expected test score
            return np.sum(probs, axis=1)
        return probs


class PolytomousItemModel(BaseItemModel):
    """Base class for polytomous (0, 1, 2, ..., K) response models.

    Polytomous models are used for ordered categorical responses
    (e.g., Likert scales). Subclasses include GRM, GPCM, PCM, and NRM.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list of int
        Number of response categories per item. If int, same for all items.
        Categories are numbered 0, 1, ..., n_categories-1.
    n_factors : int, default=1
        Number of latent factors.
    item_names : list of str, optional
        Names for each item.
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        # Store n_categories before calling super().__init__
        # because _initialize_parameters needs it
        if isinstance(n_categories, int):
            self._n_categories = [n_categories] * n_items
        else:
            if len(n_categories) != n_items:
                raise ValueError(
                    f"Length of n_categories ({len(n_categories)}) must match n_items ({n_items})"
                )
            self._n_categories = list(n_categories)

        for i, n_cat in enumerate(self._n_categories):
            if n_cat < 2:
                raise ValueError(
                    f"Item {i} has {n_cat} categories; minimum is 2"
                )

        super().__init__(n_items, n_factors, item_names)

    @property
    def n_categories(self) -> list[int]:
        """Number of response categories for each item."""
        return self._n_categories.copy()

    @property
    def max_categories(self) -> int:
        """Maximum number of categories across all items."""
        return max(self._n_categories)

    @abstractmethod
    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute P(X=k|theta) for a specific category.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int
            Index of the item.
        category : int
            Category index (0 to n_categories-1).

        Returns
        -------
        ndarray of shape (n_persons,)
            Probability of response in the specified category.
        """
        ...

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute expected item or test score: E[X|theta].

        For polytomous items, E[X|theta] = sum(k * P(X=k|theta)).

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int, optional
            If specified, return expected score for one item.
            If None, return sum of expected scores across all items.

        Returns
        -------
        ndarray
            Expected scores.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            expected = np.zeros(n_persons)
            for k in range(n_cat):
                expected += k * self.category_probability(theta, item_idx, k)
            return expected

        # Sum across all items
        total_expected = np.zeros(n_persons)
        for i in range(self.n_items):
            total_expected += self.expected_score(theta, i)
        return total_expected

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute category response curves for an item.

        Returns probabilities for all categories at each theta value.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int
            Index of the item.

        Returns
        -------
        ndarray of shape (n_persons, n_categories)
            Probability of each category at each theta.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        curves = np.zeros((n_persons, n_cat))
        for k in range(n_cat):
            curves[:, k] = self.category_probability(theta, item_idx, k)

        return curves
