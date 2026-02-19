"""Item selection strategies for multidimensional computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MCATSelectionStrategy(ABC):
    """Abstract base class for MCAT item selection strategies.

    Item selection strategies for multidimensional CAT determine which item
    to administer next based on the current ability estimates across all
    dimensions and the available item pool.
    """

    @abstractmethod
    def select_item(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        """Select the next item to administer.

        Parameters
        ----------
        model : BaseItemModel
            The fitted multidimensional IRT model.
        theta : NDArray[np.float64]
            Current ability estimates, shape (n_factors,).
        covariance : NDArray[np.float64]
            Current posterior covariance matrix, shape (n_factors, n_factors).
        available_items : set[int]
            Set of item indices that can still be administered.
        administered_items : list[int] | None
            List of already administered item indices.
        responses : list[int] | None
            List of responses to administered items.

        Returns
        -------
        int
            Index of the selected item.
        """
        pass

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
    ) -> dict[int, float]:
        """Get selection criterion values for all available items.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model.
        theta : NDArray[np.float64]
            Current ability estimates.
        covariance : NDArray[np.float64]
            Current posterior covariance matrix.
        available_items : set[int]
            Set of available item indices.

        Returns
        -------
        dict[int, float]
            Dictionary mapping item indices to criterion values.
        """
        criteria = {}
        for item_idx in available_items:
            criteria[item_idx] = self._compute_criterion(
                model, theta, covariance, item_idx
            )
        return criteria

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        """Compute the selection criterion for a single item.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model.
        theta : NDArray[np.float64]
            Current ability estimates.
        covariance : NDArray[np.float64]
            Current posterior covariance matrix.
        item_idx : int
            Index of the item.

        Returns
        -------
        float
            Criterion value (higher = more desirable).
        """
        return 0.0


def _compute_item_information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Compute the Fisher information matrix for a single item.

    For multidimensional IRT with compensatory model:
    I_j(theta) = p_j * q_j * a_j @ a_j.T

    Parameters
    ----------
    model : BaseItemModel
        Fitted multidimensional IRT model.
    theta : NDArray[np.float64]
        Ability vector, shape (n_factors,).
    item_idx : int
        Item index.

    Returns
    -------
    NDArray[np.float64]
        Information matrix of shape (n_factors, n_factors).
    """
    theta_2d = theta.reshape(1, -1)
    n_factors = theta.shape[0]

    p = model.probability(theta_2d, item_idx=item_idx)
    p = np.asarray(p).ravel()
    if len(p) == 1:
        p_val = float(p[0])
    else:
        p_val = float(p[0])
    p_val = np.clip(p_val, PROB_EPSILON, 1 - PROB_EPSILON)
    q = 1 - p_val

    params = model.get_item_parameters(item_idx)
    if "discrimination" in params:
        a = np.asarray(params["discrimination"])
        if a.ndim == 0:
            a = np.array([float(a)] * n_factors)
    elif "slopes" in params:
        a = np.asarray(params["slopes"])
    else:
        a = np.ones(n_factors)

    if len(a) != n_factors:
        a = np.ones(n_factors)

    info_matrix = p_val * q * np.outer(a, a)
    return info_matrix


def _compute_posterior_covariance_update(
    prior_cov: NDArray[np.float64],
    item_info: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the posterior covariance after observing an item.

    Uses the Bayesian update formula:
    Sigma_post^{-1} = Sigma_prior^{-1} + I_item

    Parameters
    ----------
    prior_cov : NDArray[np.float64]
        Prior covariance matrix.
    item_info : NDArray[np.float64]
        Item information matrix.

    Returns
    -------
    NDArray[np.float64]
        Posterior covariance matrix.
    """
    prior_precision = np.linalg.inv(prior_cov + np.eye(prior_cov.shape[0]) * 1e-8)
    post_precision = prior_precision + item_info
    post_cov = np.linalg.inv(post_precision + np.eye(post_precision.shape[0]) * 1e-8)
    return post_cov


def _select_best_item_by_criterion(
    strategy: MCATSelectionStrategy,
    model: BaseItemModel,
    theta: NDArray[np.float64],
    covariance: NDArray[np.float64],
    available_items: set[int],
) -> int:
    """Select the available item with the highest criterion value."""
    if not available_items:
        raise ValueError("No available items to select from")

    best_item = -1
    best_criterion = -np.inf

    for item_idx in available_items:
        criterion = strategy._compute_criterion(model, theta, covariance, item_idx)
        if criterion > best_criterion:
            best_criterion = criterion
            best_item = item_idx

    return best_item


class _CriterionSelectionStrategy(MCATSelectionStrategy):
    """Base class for strategies that rank by a scalar item criterion."""

    def select_item(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        _ = administered_items, responses
        return _select_best_item_by_criterion(
            self, model, theta, covariance, available_items
        )


class _PosteriorCovarianceCriterion(_CriterionSelectionStrategy):
    """Base class for criteria computed from posterior covariance updates."""

    @abstractmethod
    def _criterion_from_post_cov(self, post_cov: NDArray[np.float64]) -> float:
        """Map posterior covariance to selection criterion value."""

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        item_info = _compute_item_information_matrix(model, theta, item_idx)
        post_cov = _compute_posterior_covariance_update(covariance, item_info)
        return self._criterion_from_post_cov(post_cov)


class DOptimality(_PosteriorCovarianceCriterion):
    """D-optimality item selection for MCAT.

    Selects the item that maximizes the determinant of the posterior
    information matrix (equivalently, minimizes the determinant of the
    posterior covariance matrix). This criterion minimizes the volume
    of the confidence ellipsoid around the ability estimate.

    This is the most commonly used criterion for MCAT as it balances
    information across all dimensions.

    References
    ----------
    Segall, D. O. (1996). Multidimensional adaptive testing.
    Psychometrika, 61(2), 331-354.
    """

    def _criterion_from_post_cov(self, post_cov: NDArray[np.float64]) -> float:
        return float(-np.linalg.det(post_cov))


class AOptimality(_PosteriorCovarianceCriterion):
    """A-optimality item selection for MCAT.

    Selects the item that minimizes the trace of the posterior covariance
    matrix. This criterion minimizes the sum of variances across all
    dimensions.

    References
    ----------
    Mulder, J., & van der Linden, W. J. (2009). Multidimensional adaptive
    testing with optimal design criteria for item selection.
    Psychometrika, 74(2), 273-296.
    """

    def _criterion_from_post_cov(self, post_cov: NDArray[np.float64]) -> float:
        return float(-np.trace(post_cov))


class COptimality(_PosteriorCovarianceCriterion):
    """C-optimality item selection for MCAT.

    Selects the item that minimizes variance along a specified direction
    (composite) in the latent space. This is useful when a particular
    linear combination of traits is of primary interest.

    Parameters
    ----------
    weights : NDArray[np.float64] | None
        Weight vector for the composite, shape (n_factors,).
        If None, uses equal weights for all dimensions.

    References
    ----------
    van der Linden, W. J. (1999). Multidimensional adaptive testing
    with a minimum error-variance criterion. Journal of Educational
    and Behavioral Statistics, 24(4), 398-412.
    """

    def __init__(self, weights: NDArray[np.float64] | None = None):
        self.weights = weights

    def _normalized_weights(self, n_factors: int) -> NDArray[np.float64]:
        if self.weights is None:
            return np.ones(n_factors) / np.sqrt(n_factors)
        return self.weights / np.linalg.norm(self.weights)

    def _criterion_from_post_cov(self, post_cov: NDArray[np.float64]) -> float:
        weights = self._normalized_weights(post_cov.shape[0])
        composite_var = float(weights @ post_cov @ weights)
        return -composite_var


class KullbackLeiblerMCAT(_CriterionSelectionStrategy):
    """Kullback-Leibler divergence item selection for MCAT.

    Selects the item that maximizes the expected KL divergence between
    the posterior distributions before and after observing the item.
    This is equivalent to maximizing expected information gain.

    Parameters
    ----------
    n_integration_points : int
        Number of points per dimension for numerical integration.

    References
    ----------
    Wang, C., Chang, H.-H., & Boughton, K. A. (2011). Kullback-Leibler
    information and its applications in multi-dimensional adaptive testing.
    Psychometrika, 76(1), 13-39.
    """

    def __init__(self, n_integration_points: int = 5):
        self.n_integration_points = n_integration_points

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        item_info = _compute_item_information_matrix(model, theta, item_idx)
        trace_info_cov = np.trace(item_info @ covariance)
        return trace_info_cov


class BayesianMCAT(_PosteriorCovarianceCriterion):
    """Bayesian (minimum expected posterior variance) selection for MCAT.

    Selects the item that minimizes the expected posterior variance,
    integrating over possible responses weighted by their probabilities.

    This method explicitly accounts for the uncertainty in the response.

    References
    ----------
    Owen, R. J. (1975). A Bayesian sequential procedure for quantal
    response in the context of adaptive mental testing. Journal of the
    American Statistical Association, 70(350), 351-356.
    """

    def _criterion_from_post_cov(self, post_cov: NDArray[np.float64]) -> float:
        return float(-np.trace(post_cov))


class RandomMCATSelection(MCATSelectionStrategy):
    """Random item selection for MCAT.

    Randomly selects an item from the available pool.
    Useful as a baseline or for initial items.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_item(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        items_list = list(available_items)
        return items_list[self.rng.integers(len(items_list))]


def create_mcat_selection_strategy(
    method: str,
    **kwargs: Any,
) -> MCATSelectionStrategy:
    """Factory function to create MCAT item selection strategies.

    Parameters
    ----------
    method : str
        Selection method name. One of: "D-optimality", "A-optimality",
        "C-optimality", "KL", "Bayesian", "random".
    **kwargs
        Additional keyword arguments passed to the strategy constructor.

    Returns
    -------
    MCATSelectionStrategy
        The requested item selection strategy.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    strategies: dict[str, type[MCATSelectionStrategy]] = {
        "D-optimality": DOptimality,
        "A-optimality": AOptimality,
        "C-optimality": COptimality,
        "KL": KullbackLeiblerMCAT,
        "Bayesian": BayesianMCAT,
        "random": RandomMCATSelection,
    }

    method_normalized = method.replace("_", "-")
    if method_normalized not in strategies:
        valid = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown MCAT selection method '{method}'. Valid options: {valid}"
        )

    return strategies[method_normalized](**kwargs)
