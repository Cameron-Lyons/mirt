"""Item selection strategies for multidimensional computerized adaptive testing."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt._logistic import _sigmoid_derivative
from mirt._model_defaults import uses_builtin_model_hooks, uses_original_model_hook
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtModelError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_SELECTION_WORKING_BYTES = 8 * 1024 * 1024


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


def _validate_ability_vector(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``theta`` as a finite vector with one value per factor."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (model.n_factors,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"theta must contain {model.n_factors} finite factor values")
    return theta


def _slope_rows(model: BaseItemModel) -> NDArray[np.float64]:
    """Return one slope vector per item for the compensatory fallback.

    ``discrimination`` takes precedence over ``slopes``. Scalar item slopes
    apply to every factor, and parameters without a recognizable per-factor
    shape fall back to unit slopes.
    """
    n_items, n_factors = model.n_items, model.n_factors
    parameters = model.parameters
    name = next(
        (key for key in ("discrimination", "slopes") if key in parameters), None
    )
    if name is not None:
        values = np.asarray(parameters[name], dtype=np.float64)
        if values.ndim == 1 and values.shape[0] == n_items:
            return np.repeat(values[:, None], n_factors, axis=1)
        if values.shape == (n_items, n_factors):
            return values
    return np.ones((n_items, n_factors))


def _affine_logistic_matrices(
    model: BaseItemModel,
    theta_2d: NDArray[np.float64],
    indices: NDArray[np.intp],
) -> NDArray[np.float64] | None:
    """Batch an unmodified ``MultidimensionalModel``'s native item matrices.

    The model's ``item_information_matrix`` is ``sigma'(z_j) a_j a_j^T``, so
    one logit evaluation for the whole bank serves every candidate. Items
    whose variance or slope products need the model's log-space recovery are
    evaluated by its own method. Returns None for any other model.
    """
    from mirt.models.multidimensional import MultidimensionalModel

    if (
        type(model) is not MultidimensionalModel
        or not uses_builtin_model_hooks(model)
        or not uses_original_model_hook(model, "item_information_matrix")
    ):
        return None

    logits = np.asarray(model._logits(theta_2d), dtype=np.float64)
    logits = logits.reshape(-1)[indices]
    slopes = np.asarray(model.slopes, dtype=np.float64)[indices]
    variance = _sigmoid_derivative(logits)
    tiny = np.finfo(np.float64).tiny
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        coefficients = slopes[:, :, None] * slopes[:, None, :]
        matrices = variance[:, None, None] * coefficients
        active = (slopes[:, :, None] != 0.0) & (slopes[:, None, :] != 0.0)
        unsafe = active & (~np.isfinite(coefficients) | (np.abs(coefficients) < tiny))
    recover = np.any(unsafe, axis=(1, 2)) | ((variance < tiny) & np.isfinite(logits))
    for position in np.flatnonzero(recover).tolist():
        matrices[position] = model.item_information_matrix(
            theta_2d, int(indices[position])
        )[0]
    return matrices


def _item_information_matrices(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    items: list[int] | NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute Fisher information matrices for several items at one ability.

    A model's native ``item_information_matrix`` is used whenever it exists,
    so noncompensatory and polytomous response functions keep their true
    probability gradients. The built-in compensatory ``MultidimensionalModel``
    evaluates it for all candidates at once. Dichotomous models without that
    method use the compensatory logistic matrix ``p_j * q_j * a_j @ a_j.T``
    from one probability call for the whole candidate set. Polytomous models
    must define the native method, because no single response probability
    determines their information.

    Parameters
    ----------
    model : BaseItemModel
        Fitted multidimensional IRT model.
    theta : NDArray[np.float64]
        Ability vector, shape (n_factors,).
    items : list[int] | NDArray[np.int_]
        Item indices.

    Returns
    -------
    NDArray[np.float64]
        Information matrices of shape (n_items_requested, n_factors, n_factors).

    Raises
    ------
    MirtModelError
        If a polytomous model does not define ``item_information_matrix``.
    """
    theta = _validate_ability_vector(model, theta)
    indices = np.asarray(items)
    if indices.size and not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("item indices must be integers")
    indices = indices.astype(np.intp).reshape(-1)
    invalid = (indices < 0) | (indices >= model.n_items)
    if np.any(invalid):
        item_idx = int(indices[np.argmax(invalid)])
        raise IndexError(f"item_idx {item_idx} out of range [0, {model.n_items})")

    theta_2d = theta.reshape(1, -1)
    n_factors = model.n_factors
    native_information = getattr(model, "item_information_matrix", None)
    if callable(native_information):
        batched = _affine_logistic_matrices(model, theta_2d, indices)
        if batched is not None:
            if not np.all(np.isfinite(batched)):
                raise ValueError("item_information_matrix must return finite values")
            return batched
        matrices = np.empty((indices.size, n_factors, n_factors))
        for position, item_idx in enumerate(indices.tolist()):
            information = np.asarray(
                native_information(theta_2d, item_idx), dtype=np.float64
            )
            if information.shape == (1, n_factors, n_factors):
                information = information[0]
            elif information.shape != (n_factors, n_factors):
                raise ValueError(
                    "item_information_matrix must return shape "
                    f"({n_factors}, {n_factors}) or "
                    f"(1, {n_factors}, {n_factors}), got {information.shape}"
                )
            matrices[position] = information
        if not np.all(np.isfinite(matrices)):
            raise ValueError("item_information_matrix must return finite values")
        return matrices

    if model.is_polytomous:
        model_type = type(model).__name__
        raise MirtModelError(
            f"{model_type} does not define item_information_matrix; MCAT "
            "selection requires exact item information matrices for "
            "polytomous models",
            model_type=model_type,
        )

    probabilities = np.asarray(model.probability(theta_2d), dtype=np.float64)
    if probabilities.size == model.n_items:
        p = probabilities.reshape(-1)[indices]
    else:
        p = np.array(
            [
                np.asarray(model.probability(theta_2d, item_idx=item_idx)).ravel()[0]
                for item_idx in indices.tolist()
            ],
            dtype=np.float64,
        )
    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
    slopes = _slope_rows(model)[indices]
    return (p * (1 - p))[:, None, None] * (slopes[:, :, None] * slopes[:, None, :])


def _compute_item_information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Compute the Fisher information matrix for a single item.

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
    return _item_information_matrices(model, theta, [item_idx])[0]


def _candidate_batches(model: BaseItemModel, items: list[int]) -> list[list[int]]:
    """Split candidates to bound candidate x factor x factor working arrays."""
    batch_size = max(1, _SELECTION_WORKING_BYTES // (32 * model.n_factors**2))
    return [
        items[start : start + batch_size] for start in range(0, len(items), batch_size)
    ]


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

    criteria = strategy.get_item_criteria(model, theta, covariance, available_items)
    if not all(np.isfinite(value) for value in criteria.values()):
        raise ValueError("Item selection criteria must be finite")
    # Stable ties make item selection reproducible regardless of set ordering.
    return max(sorted(criteria), key=criteria.__getitem__)


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

    def _criteria_from_post_covs(
        self, post_covs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map stacked posterior covariances to criterion values.

        Built-in criteria evaluate the whole stack at once unless a subclass
        overrides their per-matrix ``_criterion_from_post_cov``.
        """
        return np.array(
            [self._criterion_from_post_cov(post_cov) for post_cov in post_covs],
            dtype=np.float64,
        )

    def _uses_criterion_of(self, cls: type[_PosteriorCovarianceCriterion]) -> bool:
        """Return whether ``cls`` authored this strategy's per-matrix criterion."""
        return type(self)._criterion_from_post_cov is cls._criterion_from_post_cov

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
    ) -> dict[int, float]:
        """Rank candidates while sharing the current posterior precision.

        All candidates start from the same covariance. Invert it once, then
        update candidate matrices in bounded batches while retaining native
        model information and subclass-specific criteria.
        """
        if not available_items:
            return {}
        items = sorted(available_items)
        regularization = np.eye(model.n_factors) * 1e-8
        prior_precision = np.linalg.inv(covariance + regularization)
        criteria: dict[int, float] = {}
        for batch in _candidate_batches(model, items):
            information = _item_information_matrices(model, theta, batch)
            information += prior_precision
            information += regularization
            values = self._criteria_from_post_covs(np.linalg.inv(information))
            criteria.update(zip(batch, values.tolist(), strict=True))
        return criteria

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

    def _criteria_from_post_covs(
        self, post_covs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self._uses_criterion_of(DOptimality):
            return super()._criteria_from_post_covs(post_covs)
        return -np.linalg.det(post_covs)


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

    def _criteria_from_post_covs(
        self, post_covs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self._uses_criterion_of(AOptimality):
            return super()._criteria_from_post_covs(post_covs)
        return -np.trace(post_covs, axis1=1, axis2=2)


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

    def _criteria_from_post_covs(
        self, post_covs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self._uses_criterion_of(COptimality):
            return super()._criteria_from_post_covs(post_covs)
        weights = self._normalized_weights(post_covs.shape[-1])
        return -((weights @ post_covs) @ weights)


class KullbackLeiblerMCAT(_CriterionSelectionStrategy):
    """Kullback-Leibler information item selection for MCAT.

    Ranks items by the posterior-weighted Kullback-Leibler information
    around the current estimate. Under the local quadratic approximation used
    here, the KL information of item ``j`` averaged over the posterior is
    ``trace(I_j(theta) @ Sigma) / 2``, so items are ranked by
    ``trace(I_j(theta) @ Sigma)``, where ``Sigma`` is the current posterior
    covariance. Information matrices are evaluated in bounded candidate
    batches.

    Parameters
    ----------
    n_integration_points : int | None
        Deprecated and ignored. The quadratic approximation needs no
        numerical integration.

    References
    ----------
    Wang, C., Chang, H.-H., & Boughton, K. A. (2011). Kullback-Leibler
    information and its applications in multi-dimensional adaptive testing.
    Psychometrika, 76(1), 13-39.
    """

    def __init__(self, n_integration_points: int | None = None):
        if n_integration_points is not None:
            warnings.warn(
                "n_integration_points is deprecated and ignored; "
                "KullbackLeiblerMCAT uses a closed-form quadratic approximation",
                DeprecationWarning,
                stacklevel=2,
            )
        self.n_integration_points = n_integration_points

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
    ) -> dict[int, float]:
        """Evaluate ``trace(I_j(theta) @ Sigma)`` for all candidates in batches.

        Subclasses that override the per-item ``_compute_criterion`` are
        evaluated one item at a time through it instead.
        """
        if type(self)._compute_criterion is not KullbackLeiblerMCAT._compute_criterion:
            return super().get_item_criteria(model, theta, covariance, available_items)
        if not available_items:
            return {}
        items = sorted(available_items)
        criteria: dict[int, float] = {}
        for batch in _candidate_batches(model, items):
            values = self._trace_criteria(model, theta, covariance, batch)
            criteria.update(zip(batch, values.tolist(), strict=True))
        return criteria

    @staticmethod
    def _trace_criteria(
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        items: list[int],
    ) -> NDArray[np.float64]:
        """Return ``trace(I_j(theta) @ Sigma)`` for each item."""
        information = _item_information_matrices(model, theta, items)
        sigma = np.asarray(covariance, dtype=np.float64)
        return np.einsum("jab,ba->j", information, sigma)

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        return float(self._trace_criteria(model, theta, covariance, [item_idx])[0])


class BayesianMCAT(AOptimality):
    """Bayesian (minimum expected posterior variance) selection for MCAT.

    Selects the item that minimizes the expected total posterior variance
    after the response. With the Fisher (Laplace) posterior update used by
    the D-, A-, and C-optimality strategies, the updated covariance
    ``(Sigma^-1 + I_j(theta))^-1`` does not depend on the observed response
    for canonical-link models such as the compensatory logistic and
    partial-credit families. The expectation over responses is therefore the
    A-optimality criterion, which this class shares.

    References
    ----------
    Owen, R. J. (1975). A Bayesian sequential procedure for quantal
    response in the context of adaptive mental testing. Journal of the
    American Statistical Association, 70(350), 351-356.
    """


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

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        covariance: NDArray[np.float64],
        available_items: set[int],
    ) -> dict[int, float]:
        """Return independent uniform scores from this strategy's generator.

        Ranking these scores, as randomesque exposure control does, yields a
        seeded uniformly random choice instead of a fixed item order.
        """
        items = sorted(available_items)
        return dict(zip(items, self.rng.random(len(items)).tolist(), strict=True))


def create_mcat_selection_strategy(
    method: str,
    **kwargs: Any,
) -> MCATSelectionStrategy:
    """Factory function to create MCAT item selection strategies.

    Parameters
    ----------
    method : str
        Selection method name. One of: "D-optimality", "A-optimality",
        "C-optimality", "KL", "Bayesian", "random". Names match regardless
        of case and surrounding whitespace, and ``"_"`` is accepted for
        ``"-"``.
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

    normalized = method.strip().lower().replace("_", "-")
    for name, strategy_class in strategies.items():
        if name.lower() == normalized:
            return strategy_class(**kwargs)
    valid = ", ".join(strategies.keys())
    raise ValueError(
        f"Unknown MCAT selection method '{method}'. Valid options: {valid}"
    )
