from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import DichotomousItemModel


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
        item_features = np.asarray(item_features, dtype=np.float64)
        if item_features.ndim != 2:
            raise ValueError("item_features must be 2D")
        if item_features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows, expected {n_items}"
            )

        self._item_features = item_features
        self._n_features = item_features.shape[1]
        self._feature_names = feature_names or [
            f"Feature_{i}" for i in range(self._n_features)
        ]
        self._constrain_discrimination = constrain_discrimination

        if len(self._feature_names) != self._n_features:
            raise ValueError(
                f"Length of feature_names ({len(self._feature_names)}) must match "
                f"number of features ({self._n_features})"
            )

        super().__init__(n_items, n_factors=1, item_names=item_names)

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
    def feature_weights(self) -> NDArray[np.float64]:
        return self._parameters["feature_weights"]

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._item_features @ self._parameters["feature_weights"]

    def _initialize_parameters(self) -> None:
        self._parameters["feature_weights"] = np.zeros(self._n_features)
        if self._constrain_discrimination:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones(self.n_items)

    def set_feature_weights(self, weights: NDArray[np.float64]) -> Self:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self._n_features,):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match ({self._n_features},)"
            )
        self._parameters["feature_weights"] = weights
        return self

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self.difficulty

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            return sigmoid(z)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return sigmoid(z)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            return (a[item_idx] ** 2) * p * q

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
        if n_covariates < 1:
            raise ValueError("n_covariates must be at least 1")

        self._n_covariates = n_covariates
        self._include_intercept = include_intercept
        self._n_weights = n_covariates + 1 if include_intercept else n_covariates

        if covariate_names is None:
            self._covariate_names = [f"X_{i}" for i in range(n_covariates)]
        else:
            if len(covariate_names) != n_covariates:
                raise ValueError(
                    f"Length of covariate_names ({len(covariate_names)}) must match "
                    f"n_covariates ({n_covariates})"
                )
            self._covariate_names = list(covariate_names)

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
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self._n_weights,):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match ({self._n_weights},)"
            )
        self._regression_weights = weights
        return self

    def set_residual_variance(self, variance: float) -> Self:
        if variance <= 0:
            raise ValueError("residual_variance must be positive")
        self._residual_variance = variance
        return self

    def _prepare_design_matrix(
        self, covariates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[1] != self._n_covariates:
            raise ValueError(
                f"covariates has {covariates.shape[1]} columns, "
                f"expected {self._n_covariates}"
            )
        if self._include_intercept:
            intercept = np.ones((covariates.shape[0], 1))
            return np.hstack([intercept, covariates])
        return covariates

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
        self, theta: NDArray[np.float64], covariates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute log prior density of theta given covariates."""
        mu = self.predict_mean(covariates)
        sigma2 = self._residual_variance
        return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (theta - mu) ** 2 / sigma2


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
        item_features = np.asarray(item_features, dtype=np.float64)
        if item_features.ndim != 2:
            raise ValueError("item_features must be 2D")
        if item_features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {item_features.shape[0]} rows, expected {n_items}"
            )

        self._item_features = item_features
        self._n_item_features = item_features.shape[1]
        self._n_person_covariates = n_person_covariates
        self._constrain_discrimination = constrain_discrimination
        self._include_intercept = include_intercept

        self._feature_names = feature_names or [
            f"ItemFeature_{i}" for i in range(self._n_item_features)
        ]
        self._covariate_names = covariate_names or [
            f"PersonCov_{i}" for i in range(n_person_covariates)
        ]

        self._latent_regression = LatentRegressionModel(
            n_covariates=n_person_covariates,
            covariate_names=self._covariate_names,
            include_intercept=include_intercept,
        )

        super().__init__(n_items, n_factors=1, item_names=item_names)

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
    def feature_weights(self) -> NDArray[np.float64]:
        return self._parameters["feature_weights"]

    @property
    def regression_weights(self) -> NDArray[np.float64]:
        return self._latent_regression.regression_weights

    @property
    def residual_variance(self) -> float:
        return self._latent_regression.residual_variance

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

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
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self._n_item_features,):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match "
                f"({self._n_item_features},)"
            )
        self._parameters["feature_weights"] = weights
        return self

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
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self.difficulty

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            return sigmoid(z)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return sigmoid(z)

    def probability_given_covariates(
        self,
        covariates: NDArray[np.float64],
        residual_theta: NDArray[np.float64] | None = None,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute probability given person covariates.

        Parameters
        ----------
        covariates : ndarray of shape (n_persons, n_covariates)
            Person covariates.
        residual_theta : ndarray of shape (n_persons,), optional
            Residual ability not explained by covariates. If None, uses 0.
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
            theta = mu + np.asarray(residual_theta).ravel()
        return self.probability(theta.reshape(-1, 1), item_idx)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            return (a[item_idx] ** 2) * p * q

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
        new_model._latent_regression = LatentRegressionModel(
            n_covariates=self._n_person_covariates,
            covariate_names=self._covariate_names,
            include_intercept=self._include_intercept,
        )
        new_model._latent_regression.set_regression_weights(
            self._latent_regression.regression_weights
        )
        new_model._latent_regression.set_residual_variance(
            self._latent_regression.residual_variance
        )
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
