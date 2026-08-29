"""Explanatory item and person covariate IRT models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import DichotomousItemModel
from mirt.utils.numeric import standard_normal_quadrature


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _numeric_matrix(
    values: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _parameter_vector(
    values: NDArray[np.float64],
    size: int,
    name: str,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} shape {vector.shape} doesn't match ({size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _names_or_default(
    names: list[str] | None,
    size: int,
    prefix: str,
    name: str,
) -> list[str]:
    if names is None:
        return [f"{prefix}_{index}" for index in range(size)]
    if len(names) != size:
        raise ValueError(
            f"Length of {name} ({len(names)}) must match expected size ({size})"
        )
    if not all(isinstance(value, str) for value in names):
        raise ValueError(f"{name} must contain strings")
    return list(names)


def _item_index(item_idx: int | None, n_items: int) -> int | None:
    if item_idx is None:
        return None
    if (
        isinstance(item_idx, (bool, np.bool_))
        or not isinstance(item_idx, (int, np.integer))
        or item_idx < 0
        or item_idx >= n_items
    ):
        raise IndexError(f"item_idx must be in [0, {n_items})")
    return int(item_idx)


def _person_vector(
    values: NDArray[np.float64] | float,
    n_persons: int,
    name: str,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim == 0:
        vector = np.full(n_persons, float(vector), dtype=np.float64)
    elif vector.ndim == 2 and vector.shape[1:] == (1,):
        vector = vector[:, 0]
    elif vector.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    if vector.shape not in {(1,), (n_persons,)}:
        raise ValueError(f"{name} must contain one value or {n_persons} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if vector.shape == (1,) and n_persons != 1:
        return np.full(n_persons, vector[0], dtype=np.float64)
    return vector.copy()


@dataclass
class LLTMResult:
    """Results from LLTM estimation."""

    feature_weights: NDArray[np.float64]
    feature_se: NDArray[np.float64]
    item_difficulties: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    n_iterations: int
    converged: bool


@dataclass
class LatentRegressionResult:
    """Results from latent regression estimation."""

    regression_weights: NDArray[np.float64]
    regression_se: NDArray[np.float64]
    residual_variance: float
    r_squared: float
    log_likelihood: float
    aic: float
    bic: float
    n_iterations: int
    converged: bool


@dataclass
class ExplanatoryIRTResult:
    """Results from combined explanatory IRT estimation."""

    feature_weights: NDArray[np.float64]
    feature_se: NDArray[np.float64]
    regression_weights: NDArray[np.float64]
    regression_se: NDArray[np.float64]
    residual_variance: float
    discrimination: NDArray[np.float64]
    item_difficulties: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    n_iterations: int
    converged: bool


class LLTM(DichotomousItemModel):
    """Linear Logistic Test Model.

    The LLTM constrains item difficulties to be linear combinations of item
    features (cognitive operations, content categories, etc.). This allows
    modeling what makes items difficult.

    Parameters
    ----------
    n_items : int
        Number of items.
    item_features : ndarray of shape (n_items, n_features)
        Design matrix specifying item feature values. Each row is an item,
        each column is a feature (e.g., number of operations, content type).
    feature_names : list of str, optional
        Names for the features.
    item_names : list of str, optional
        Names for items.

    Attributes
    ----------
    feature_weights : ndarray of shape (n_features,)
        Weights (eta parameters) for each feature. Item difficulty is
        computed as item_features @ feature_weights.
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters. Can be constrained to equality
        for a Rasch-like LLTM.

    Notes
    -----
    The LLTM model is:

        P(X=1|θ) = 1 / (1 + exp(-a * (θ - Σ q_jk * η_k)))

    where q_jk is the feature value for item j on feature k, and η_k is
    the weight for feature k.

    References
    ----------
    Fischer, G. H. (1973). The linear logistic test model as an instrument
        in educational research. Acta Psychologica, 37, 359-374.
    """

    model_name = "LLTM"
    n_params_per_item = 1
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        item_features: NDArray[np.float64],
        feature_names: list[str] | None = None,
        item_names: list[str] | None = None,
        constrain_discrimination: bool = True,
    ) -> None:
        n_items = _positive_integer(n_items, "n_items")
        item_features = _numeric_matrix(item_features, "item_features")
        if item_features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows, expected {n_items}"
            )
        if item_features.shape[1] == 0:
            raise ValueError("item_features must contain at least one feature")

        self._item_features = item_features
        self._n_features = item_features.shape[1]
        self._feature_names = _names_or_default(
            feature_names,
            self._n_features,
            "Feature",
            "feature_names",
        )
        self._constrain_discrimination = _boolean(
            constrain_discrimination,
            "constrain_discrimination",
        )

        super().__init__(
            n_items,
            n_factors=1,
            item_names=None if item_names is None else list(item_names),
        )

    @property
    def item_features(self) -> NDArray[np.float64]:
        return self._item_features.copy()

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()

    @property
    def constrain_discrimination(self) -> bool:
        """Whether item discriminations must share one common value."""
        return self._constrain_discrimination

    @property
    def feature_weights(self) -> NDArray[np.float64]:
        return self._parameters["feature_weights"].copy()

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"].copy()

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._item_features @ self._parameters["feature_weights"]

    def _initialize_parameters(self) -> None:
        self._parameters["feature_weights"] = np.zeros(self._n_features)
        self._parameters["discrimination"] = np.ones(self.n_items)

    def set_feature_weights(self, weights: NDArray[np.float64]) -> Self:
        """Set finite item-feature effects without retaining caller storage."""
        return self.set_parameters(feature_weights=weights)

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set finite, identified item parameters."""
        normalized = dict(params)
        if "feature_weights" in normalized:
            normalized["feature_weights"] = _parameter_vector(
                normalized["feature_weights"],
                self._n_features,
                "feature_weights",
            )
        if "discrimination" in normalized:
            discrimination = _parameter_vector(
                normalized["discrimination"],
                self.n_items,
                "discrimination",
            )
            if np.any(discrimination <= 0.0):
                raise ValueError("discrimination must contain positive values")
            if self._constrain_discrimination and not np.allclose(
                discrimination,
                discrimination[0],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "constrained discrimination must use one common item value"
                )
            normalized["discrimination"] = discrimination
        return super().set_parameters(**normalized)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        index = _item_index(item_idx, self.n_items)
        assert index is not None
        if param_name == "discrimination":
            if self._constrain_discrimination:
                raise ValueError(
                    "Cannot set one discrimination when discrimination is constrained"
                )
            scalar = np.asarray(value, dtype=np.float64)
            if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError("discrimination must be a finite positive scalar")
        super().set_item_parameter(index, param_name, value)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values")
        theta_1d = theta.ravel()
        index = _item_index(item_idx, self.n_items)

        a = self._parameters["discrimination"]
        b = self.difficulty

        if index is not None:
            z = a[index] * (theta_1d - b[index])
            return sigmoid(z)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return sigmoid(z)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        index = _item_index(item_idx, self.n_items)
        p = self.probability(theta, index)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if index is not None:
            return (a[index] ** 2) * p * q

        return (a[None, :] ** 2) * p * q

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            item_features=self._item_features.copy(),
            feature_names=self._feature_names.copy(),
            item_names=self.item_names.copy(),
            constrain_discrimination=self._constrain_discrimination,
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._is_fitted = self._is_fitted
        return new_model


class LatentRegressionModel:
    """Latent regression model for person ability.

    Models person ability as a function of observed covariates with
    residual variance.

    Parameters
    ----------
    n_covariates : int
        Number of person covariates.
    covariate_names : list of str, optional
        Names for covariates.
    include_intercept : bool, default=True
        Whether to include an intercept term.

    Attributes
    ----------
    regression_weights : ndarray of shape (n_covariates,) or (n_covariates+1,)
        Regression coefficients. If include_intercept is True, the first
        element is the intercept.
    residual_variance : float
        Variance of residual ability not explained by covariates.

    Notes
    -----
    The model is:

        θ_i = X_i @ β + ε_i, where ε_i ~ N(0, σ²)

    where X_i is the covariate vector for person i, β are the regression
    weights, and σ² is the residual variance.
    """

    def __init__(
        self,
        n_covariates: int,
        covariate_names: list[str] | None = None,
        include_intercept: bool = True,
    ) -> None:
        self._n_covariates = _positive_integer(n_covariates, "n_covariates")
        self._include_intercept = _boolean(include_intercept, "include_intercept")
        self._n_weights = self._n_covariates + int(self._include_intercept)
        self._covariate_names = _names_or_default(
            covariate_names,
            self._n_covariates,
            "X",
            "covariate_names",
        )

        self._regression_weights = np.zeros(self._n_weights)
        self._residual_variance = 1.0

    @property
    def n_covariates(self) -> int:
        return self._n_covariates

    @property
    def covariate_names(self) -> list[str]:
        return self._covariate_names.copy()

    @property
    def include_intercept(self) -> bool:
        return self._include_intercept

    @property
    def regression_weights(self) -> NDArray[np.float64]:
        return self._regression_weights.copy()

    @property
    def residual_variance(self) -> float:
        return self._residual_variance

    def set_regression_weights(self, weights: NDArray[np.float64]) -> Self:
        """Set finite regression weights without retaining caller storage."""
        self._regression_weights = _parameter_vector(
            weights,
            self._n_weights,
            "regression_weights",
        )
        return self

    def set_residual_variance(self, variance: float) -> Self:
        raw = np.asarray(variance)
        if (
            isinstance(variance, (bool, np.bool_))
            or raw.ndim != 0
            or raw.dtype.kind not in "iuf"
        ):
            raise ValueError("residual_variance must be finite and positive")
        numeric_variance = float(raw)
        if not np.isfinite(numeric_variance) or numeric_variance <= 0.0:
            raise ValueError("residual_variance must be finite and positive")
        self._residual_variance = numeric_variance
        return self

    def _prepare_design_matrix(
        self, covariates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raw = np.asarray(covariates)
        if raw.dtype.kind not in "biuf":
            raise ValueError("covariates must contain numeric values")
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim == 1:
            if self._n_covariates == 1:
                values = values.reshape(-1, 1)
            else:
                values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("covariates must be one- or two-dimensional")
        if values.shape[1] != self._n_covariates:
            raise ValueError(
                f"covariates has {values.shape[1]} columns, "
                f"expected {self._n_covariates}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("covariates must contain only finite values")
        if self._include_intercept:
            intercept = np.ones((values.shape[0], 1))
            return np.hstack([intercept, values])
        return values

    def predict_mean(self, covariates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict expected ability given covariates."""
        X = self._prepare_design_matrix(covariates)
        return X @ self._regression_weights

    def prior_mean(self, covariates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Alias for predict_mean for use in estimation."""
        return self.predict_mean(covariates)

    def prior_variance(self) -> float:
        """Return residual variance as prior variance."""
        return self._residual_variance

    def log_prior_density(
        self,
        theta: NDArray[np.float64] | float,
        covariates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log prior density of theta given covariates."""
        mu = self.predict_mean(covariates)
        theta_values = _person_vector(theta, mu.size, "theta")
        sigma2 = self._residual_variance
        return (
            -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (theta_values - mu) ** 2 / sigma2
        )

    def copy(self) -> Self:
        """Create an independent copy of the latent regression."""
        new_model = self.__class__(
            n_covariates=self._n_covariates,
            covariate_names=self._covariate_names.copy(),
            include_intercept=self._include_intercept,
        )
        new_model._regression_weights = self._regression_weights.copy()
        new_model._residual_variance = self._residual_variance
        return new_model


class ExplanatoryIRT(DichotomousItemModel):
    """Combined Explanatory IRT model with LLTM and latent regression.

    Combines item-side explanation (LLTM) with person-side explanation
    (latent regression) for a fully explanatory IRT model.

    Parameters
    ----------
    n_items : int
        Number of items.
    item_features : ndarray of shape (n_items, n_item_features)
        Design matrix for item features.
    n_person_covariates : int
        Number of person covariates.
    feature_names : list of str, optional
        Names for item features.
    covariate_names : list of str, optional
        Names for person covariates.
    item_names : list of str, optional
        Names for items.
    constrain_discrimination : bool, default=True
        Whether to constrain all discriminations to equality.
    include_intercept : bool, default=True
        Whether to include intercept in latent regression.

    Notes
    -----
    The full model is:

        P(X_ij=1|X_i) = 1 / (1 + exp(-a_j * (X_i @ β + ε_i - Q_j @ η)))

    where:
    - X_i are person i's covariates
    - β are regression weights
    - ε_i ~ N(0, σ²) is residual ability
    - Q_j are item j's features
    - η are feature weights
    """

    model_name = "ExplanatoryIRT"
    n_params_per_item = 1
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        item_features: NDArray[np.float64],
        n_person_covariates: int,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
        item_names: list[str] | None = None,
        constrain_discrimination: bool = True,
        include_intercept: bool = True,
    ) -> None:
        n_items = _positive_integer(n_items, "n_items")
        item_features = _numeric_matrix(item_features, "item_features")
        if item_features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows, expected {n_items}"
            )
        if item_features.shape[1] == 0:
            raise ValueError("item_features must contain at least one feature")

        self._item_features = item_features
        self._n_item_features = item_features.shape[1]
        self._n_person_covariates = _positive_integer(
            n_person_covariates,
            "n_person_covariates",
        )
        self._constrain_discrimination = _boolean(
            constrain_discrimination,
            "constrain_discrimination",
        )
        self._include_intercept = _boolean(include_intercept, "include_intercept")

        self._feature_names = _names_or_default(
            feature_names,
            self._n_item_features,
            "ItemFeature",
            "feature_names",
        )
        self._covariate_names = _names_or_default(
            covariate_names,
            self._n_person_covariates,
            "PersonCov",
            "covariate_names",
        )

        self._latent_regression = LatentRegressionModel(
            n_covariates=self._n_person_covariates,
            covariate_names=self._covariate_names,
            include_intercept=self._include_intercept,
        )

        super().__init__(
            n_items,
            n_factors=1,
            item_names=None if item_names is None else list(item_names),
        )

    @property
    def item_features(self) -> NDArray[np.float64]:
        return self._item_features.copy()

    @property
    def n_item_features(self) -> int:
        return self._n_item_features

    @property
    def n_person_covariates(self) -> int:
        return self._n_person_covariates

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()

    @property
    def covariate_names(self) -> list[str]:
        return self._covariate_names.copy()

    @property
    def constrain_discrimination(self) -> bool:
        """Whether item discriminations must share one common value."""
        return self._constrain_discrimination

    @property
    def include_intercept(self) -> bool:
        """Whether latent regression includes an intercept."""
        return self._include_intercept

    @property
    def feature_weights(self) -> NDArray[np.float64]:
        return self._parameters["feature_weights"].copy()

    @property
    def regression_weights(self) -> NDArray[np.float64]:
        return self._latent_regression.regression_weights

    @property
    def residual_variance(self) -> float:
        return self._latent_regression.residual_variance

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"].copy()

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._item_features @ self._parameters["feature_weights"]

    @property
    def latent_regression(self) -> LatentRegressionModel:
        return self._latent_regression

    def _initialize_parameters(self) -> None:
        self._parameters["feature_weights"] = np.zeros(self._n_item_features)
        self._parameters["discrimination"] = np.ones(self.n_items)

    def set_feature_weights(self, weights: NDArray[np.float64]) -> Self:
        """Set finite item-feature effects without retaining caller storage."""
        return self.set_parameters(feature_weights=weights)

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set finite, identified item parameters."""
        normalized = dict(params)
        if "feature_weights" in normalized:
            normalized["feature_weights"] = _parameter_vector(
                normalized["feature_weights"],
                self._n_item_features,
                "feature_weights",
            )
        if "discrimination" in normalized:
            discrimination = _parameter_vector(
                normalized["discrimination"],
                self.n_items,
                "discrimination",
            )
            if np.any(discrimination <= 0.0):
                raise ValueError("discrimination must contain positive values")
            if self._constrain_discrimination and not np.allclose(
                discrimination,
                discrimination[0],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "constrained discrimination must use one common item value"
                )
            normalized["discrimination"] = discrimination
        return super().set_parameters(**normalized)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        index = _item_index(item_idx, self.n_items)
        assert index is not None
        if param_name == "discrimination":
            if self._constrain_discrimination:
                raise ValueError(
                    "Cannot set one discrimination when discrimination is constrained"
                )
            scalar = np.asarray(value, dtype=np.float64)
            if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError("discrimination must be a finite positive scalar")
        super().set_item_parameter(index, param_name, value)

    def set_regression_weights(self, weights: NDArray[np.float64]) -> Self:
        self._latent_regression.set_regression_weights(weights)
        return self

    def set_residual_variance(self, variance: float) -> Self:
        self._latent_regression.set_residual_variance(variance)
        return self

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values")
        theta_1d = theta.ravel()
        index = _item_index(item_idx, self.n_items)

        a = self._parameters["discrimination"]
        b = self.difficulty

        if index is not None:
            z = a[index] * (theta_1d - b[index])
            return sigmoid(z)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return sigmoid(z)

    def probability_given_covariates(
        self,
        covariates: NDArray[np.float64],
        residual_theta: NDArray[np.float64] | float | None = None,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute probability given person covariates.

        Parameters
        ----------
        covariates : ndarray of shape (n_persons, n_covariates)
            Person covariates.
        residual_theta : float or ndarray of shape (n_persons,), optional
            Residual ability not explained by covariates. A scalar is shared
            across respondents. If None, uses 0.
        item_idx : int, optional
            If given, compute probability for single item.

        Returns
        -------
        ndarray
            Response probabilities.
        """
        mu = self._latent_regression.predict_mean(covariates)
        if residual_theta is None:
            theta = mu
        else:
            theta = mu + _person_vector(
                residual_theta,
                mu.size,
                "residual_theta",
            )
        return self.probability(theta.reshape(-1, 1), item_idx)

    def marginal_probability_given_covariates(
        self,
        covariates: NDArray[np.float64],
        item_idx: int | None = None,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Integrate response probabilities over residual ability.

        Quadrature evaluations are vectorized across respondents and items,
        while peak working memory remains proportional to the returned array.

        Parameters
        ----------
        covariates : ndarray of shape (n_persons, n_person_covariates)
            Person covariates.
        item_idx : int, optional
            If given, return probabilities for one item.
        n_quadpts : int, default=21
            Number of Gauss-Hermite quadrature points.

        Returns
        -------
        ndarray
            Marginal probabilities with shape ``(n_persons, n_items)`` or
            ``(n_persons,)`` when ``item_idx`` is supplied.
        """
        n_quadpts = _positive_integer(n_quadpts, "n_quadpts")
        index = _item_index(item_idx, self.n_items)
        mean = self._latent_regression.predict_mean(covariates)
        nodes, weights = standard_normal_quadrature(n_quadpts)
        scale = np.sqrt(self._latent_regression.residual_variance)
        shape = (mean.size,) if index is not None else (mean.size, self.n_items)
        marginal = np.zeros(shape, dtype=np.float64)

        for node, weight in zip(nodes, weights, strict=True):
            theta = (mean + scale * node).reshape(-1, 1)
            marginal += weight * self.probability(theta, index)

        return np.clip(marginal, 0.0, 1.0)

    def marginal_log_likelihood_given_covariates(
        self,
        responses: NDArray[np.int_],
        covariates: NDArray[np.float64],
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Return joint response-pattern log likelihoods given covariates.

        Each respondent's likelihood is integrated over the shared residual
        ability distribution, preserving dependence among their item responses.
        Negative response values are treated as missing.
        """
        n_quadpts = _positive_integer(n_quadpts, "n_quadpts")
        raw_responses = np.asarray(responses)
        if raw_responses.ndim != 2:
            raise ValueError("responses must be two-dimensional")
        if raw_responses.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {raw_responses.shape[1]} items, expected {self.n_items}"
            )
        if raw_responses.dtype.kind not in "biuf":
            raise ValueError("responses must contain numeric values")
        if not np.all(np.isfinite(raw_responses)):
            raise ValueError("responses must contain only finite values")
        observed = raw_responses >= 0
        if np.any(observed & (raw_responses != 0) & (raw_responses != 1)):
            raise ValueError("observed responses must contain only 0 and 1")

        mean = self._latent_regression.predict_mean(covariates)
        if raw_responses.shape[0] != mean.size:
            raise ValueError(
                "responses and covariates must contain the same number of persons"
            )
        nodes, weights = standard_normal_quadrature(n_quadpts)
        from mirt.backends.rust.explanatory import (
            compute_explanatory_marginal_log_likelihood,
        )

        return compute_explanatory_marginal_log_likelihood(
            raw_responses,
            mean,
            np.sqrt(self._latent_regression.residual_variance),
            nodes,
            weights,
            self._parameters["discrimination"],
            self.difficulty,
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        index = _item_index(item_idx, self.n_items)
        p = self.probability(theta, index)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if index is not None:
            return (a[index] ** 2) * p * q

        return (a[None, :] ** 2) * p * q

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            item_features=self._item_features.copy(),
            n_person_covariates=self._n_person_covariates,
            feature_names=self._feature_names.copy(),
            covariate_names=self._covariate_names.copy(),
            item_names=self.item_names.copy(),
            constrain_discrimination=self._constrain_discrimination,
            include_intercept=self._include_intercept,
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._latent_regression = self._latent_regression.copy()
        new_model._is_fitted = self._is_fitted
        return new_model


class RaschLLTM(LLTM):
    """Rasch-constrained LLTM with all discriminations fixed to 1.

    This is the original LLTM as proposed by Fischer (1973).
    """

    model_name = "RaschLLTM"

    def __init__(
        self,
        n_items: int,
        item_features: NDArray[np.float64],
        feature_names: list[str] | None = None,
        item_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            n_items=n_items,
            item_features=item_features,
            feature_names=feature_names,
            item_names=item_names,
            constrain_discrimination=True,
        )

    def _initialize_parameters(self) -> None:
        self._parameters["feature_weights"] = np.zeros(self._n_features)
        self._parameters["discrimination"] = np.ones(self.n_items)

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in RaschLLTM (fixed to 1)")
        return super().set_parameters(**params)

    def copy(self) -> Self:
        """Create an independent Rasch-constrained copy."""
        new_model = self.__class__(
            n_items=self.n_items,
            item_features=self._item_features.copy(),
            feature_names=self._feature_names.copy(),
            item_names=self.item_names.copy(),
        )
        new_model._parameters["feature_weights"] = self._parameters[
            "feature_weights"
        ].copy()
        new_model._is_fitted = self._is_fitted
        return new_model
