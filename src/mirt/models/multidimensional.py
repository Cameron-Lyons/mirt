from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._logistic import (
    _affine_logits,
    _affine_probability,
    _information,
    _item_information,
    _test_information,
)
from mirt._model_defaults import register_builtin_model as _register_builtin_model
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError
from mirt.models.base import DichotomousItemModel

# Logistic slopes are divided by this scaling constant D to approximate
# normal-ogive slopes before they are standardized, as in R mirt's summary().
NORMAL_OGIVE_SCALING = 1.702


def _validated_latent_covariance(
    latent_covariance: ArrayLike, n_factors: int
) -> NDArray[np.float64]:
    """Return ``latent_covariance`` as a positive-definite float matrix."""
    try:
        matrix = np.array(latent_covariance, dtype=np.float64)
    except (TypeError, ValueError):
        raise MirtValidationError(
            "latent_covariance must be a numeric matrix",
            parameter="latent_covariance",
        ) from None
    if matrix.shape != (n_factors, n_factors):
        raise MirtValidationError(
            f"latent_covariance must have shape ({n_factors}, {n_factors})",
            parameter="latent_covariance",
            value=matrix.shape,
        )
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=1e-10, atol=1e-12
    ):
        raise MirtValidationError(
            "latent_covariance must be finite and symmetric",
            parameter="latent_covariance",
        )
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise MirtValidationError(
            "latent_covariance must be positive definite",
            parameter="latent_covariance",
        ) from None
    return matrix


def standardized_loadings(
    slopes: ArrayLike,
    latent_covariance: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert logistic slopes to standardized factor loadings.

    Parameters
    ----------
    slopes : array_like of shape (n_items, n_factors)
        Slopes of the logistic slope-intercept model.
    latent_covariance : array_like of shape (n_factors, n_factors), optional
        Covariance matrix of the factors, such as
        ``FitResult.latent_covariance``. Defaults to the identity.

    Returns
    -------
    loadings : ndarray of shape (n_items, n_factors)
        Standardized loadings.
    communalities : ndarray of shape (n_items,)
        Share of each item's latent response variance that the factors
        explain.

    Notes
    -----
    As in R mirt's ``summary()``, the slopes ``a`` are first divided by
    ``D = 1.702`` to the normal-ogive metric, ``a* = a / D``. With factor
    covariance ``Sigma``, standard deviations ``sigma`` and correlation matrix
    ``Phi``, item ``j`` has the latent response variance
    ``1 + a*_j' Sigma a*_j``, its loading on factor ``k`` is
    ``a*_jk sigma_k / sqrt(1 + a*_j' Sigma a*_j)`` and its communality is
    ``lambda_j' Phi lambda_j``. For uncorrelated factors this is R's
    conversion; with correlated factors R divides by the factor variances
    only, which leaves its loadings unchanged by the correlations.
    """
    a = np.asarray(slopes, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    a = a / NORMAL_OGIVE_SCALING
    correlation: NDArray[np.float64] | None = None
    if latent_covariance is not None:
        covariance = _validated_latent_covariance(latent_covariance, a.shape[1])
        deviation = np.sqrt(np.diag(covariance))
        a = a * deviation
        correlation = covariance / np.outer(deviation, deviation)
        if np.array_equal(correlation, np.eye(a.shape[1])):
            correlation = None

    # Rows are scaled by their largest slope, so huge slopes do not overflow.
    scale = np.maximum(1.0, np.max(np.abs(a), axis=1, keepdims=True))
    with np.errstate(under="ignore"):
        scaled = a / scale
        if correlation is None:
            explained = np.sum(scaled**2, axis=1, keepdims=True)
        else:
            quadratic = np.einsum("ik,kl,il->i", scaled, correlation, scaled)
            explained = quadratic[:, None]
        total = (1.0 / scale) ** 2 + explained
        return scaled / np.sqrt(total), (explained / total)[:, 0]


@_register_builtin_model
class MultidimensionalModel(DichotomousItemModel):
    """Compensatory slope-intercept 2PL, ``a' theta + d``, as in R mirt.

    An exploratory model has at least two factors with every slope free. A
    confirmatory model frees the slopes of its ``loading_pattern``; it may
    have a single factor, the slope-intercept 2PL that model syntax fits when
    it names the intercepts ``d``.
    """

    model_name = "MIRT"
    supports_multidimensional = True
    _rotatable_slopes = "slopes"

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
        model_type: Literal["exploratory", "confirmatory"] = "exploratory",
        loading_pattern: NDArray[np.float64] | None = None,
    ) -> None:
        if n_factors < 2 and not (n_factors == 1 and model_type == "confirmatory"):
            raise ValueError(
                "MultidimensionalModel requires n_factors >= 2, or one factor "
                "with model_type='confirmatory'"
            )

        self.model_type = model_type

        if model_type == "confirmatory":
            if loading_pattern is None:
                raise ValueError("loading_pattern required for confirmatory model")
            loading_pattern = np.asarray(loading_pattern)
            if loading_pattern.shape != (n_items, n_factors):
                raise ValueError(
                    f"loading_pattern shape {loading_pattern.shape} doesn't match "
                    f"(n_items={n_items}, n_factors={n_factors})"
                )
            self._loading_pattern = loading_pattern.copy()
        else:
            self._loading_pattern = np.ones((n_items, n_factors))

        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        slopes = np.ones((self.n_items, self.n_factors)) * 0.8
        slopes = slopes * self._loading_pattern

        self._parameters["slopes"] = slopes
        self._parameters["intercepts"] = np.zeros(self.n_items)

    @property
    def slopes(self) -> NDArray[np.float64]:
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    @property
    def loading_pattern(self) -> NDArray[np.float64]:
        return self._loading_pattern.copy()

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["slopes"] &= self._loading_pattern != 0.0
        return self._apply_free_parameter_restrictions(masks)

    def _curve_parameters(
        self, item_idx: int | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        slopes, intercepts = self.slopes, self.intercepts
        if item_idx is not None:
            item_idx = self._validate_item_index(item_idx)
            return slopes[item_idx], intercepts[item_idx]
        return slopes, intercepts

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts = self._curve_parameters(item_idx)
        return _affine_probability(theta, slopes, intercepts)

    def _logits(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        slopes, intercepts = self._curve_parameters(item_idx)
        return _affine_logits(theta, slopes, intercepts)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return _affine_probability(
            theta, self.slopes, self.intercepts, item_indices=indices
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts = self._curve_parameters(item_idx)
        return _information(
            theta, slopes, lambda points: _affine_logits(points, slopes, intercepts)
        )

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return item Fisher matrices across multidimensional theta points."""
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        return _item_information(
            self._logits(theta, item_idx), self._parameters["slopes"][item_idx]
        )

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return summed Fisher matrices across all items and theta points."""
        theta = self._ensure_theta_2d(theta)
        return _test_information(theta, self._parameters["slopes"], self._logits)

    def to_irt_parameterization(self) -> dict[str, NDArray[np.float64]]:
        a = self._parameters["slopes"]
        d = self._parameters["intercepts"]

        a_sum = np.sum(a, axis=1)
        b = -d / (a_sum + PROB_EPSILON)

        return {
            "discrimination": a.copy(),
            "difficulty": b,
        }

    def get_factor_loadings(
        self,
        standardized: bool = True,
        latent_covariance: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return the slopes or the standardized factor loadings.

        Parameters
        ----------
        standardized : bool, default=True
            Convert the slopes to standardized loadings as R mirt's
            ``summary()`` does; otherwise return a copy of the slopes.
        latent_covariance : array_like of shape (n_factors, n_factors), optional
            Factor covariance used for standardizing, such as
            ``FitResult.latent_covariance``. Defaults to uncorrelated factors
            with unit variance.

        Returns
        -------
        ndarray of shape (n_items, n_factors)
            Loadings, or slopes when ``standardized`` is false.

        See Also
        --------
        mirt.models.multidimensional.standardized_loadings : The conversion.
        """
        a = self._parameters["slopes"]

        if not standardized:
            return a.copy()
        return standardized_loadings(a, latent_covariance)[0]

    def communalities(
        self, latent_covariance: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Return the share of each item's latent response variance explained.

        Parameters
        ----------
        latent_covariance : array_like of shape (n_factors, n_factors), optional
            Factor covariance, such as ``FitResult.latent_covariance``.
            Defaults to uncorrelated factors with unit variance.

        Returns
        -------
        ndarray of shape (n_items,)
            Communalities of the standardized loadings, R mirt's ``h2`` for
            uncorrelated factors.
        """
        return standardized_loadings(self._parameters["slopes"], latent_covariance)[1]

    def set_parameters(self, **params: NDArray[np.float64]) -> "MultidimensionalModel":
        if "slopes" in params:
            slopes = np.asarray(params["slopes"])
            slopes = slopes * self._loading_pattern
            params["slopes"] = slopes

        return super().set_parameters(**params)

    def copy(self) -> "MultidimensionalModel":
        """Copy parameters and retain the confirmatory loading constraints."""
        model = self.__class__(
            n_items=self.n_items,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
            model_type=self.model_type,
            loading_pattern=self._loading_pattern.copy(),
        )
        model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        self._copy_parameter_restrictions_to(model)
        model._is_fitted = self._is_fitted
        return model
