"""Mixture IRT Models.

This module implements mixture IRT models that combine latent class
analysis with IRT to model heterogeneous populations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import BaseItemModel

if TYPE_CHECKING:
    pass


class MixtureIRT(BaseItemModel):
    """Mixture IRT model with latent classes.

    Mixture IRT models assume that the population consists of K latent
    classes, each with its own set of item parameters. The response
    probability is a mixture over classes:

    P(X_ij = 1 | theta_i) = sum_k P(C_i = k) * P_k(X_ij = 1 | theta_i)

    where:
    - C_i: Latent class membership for person i
    - P(C_i = k): Class membership probability (mixing proportion)
    - P_k: Class-specific IRT model

    This is useful for:
    - Detecting qualitatively different response strategies
    - Modeling item parameter drift across subpopulations
    - Identifying outlier groups (e.g., rapid guessers)
    """

    model_name = "MixtureIRT"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_classes: int = 2,
        base_model: Literal["1PL", "2PL", "3PL"] = "2PL",
        item_names: list[str] | None = None,
    ) -> None:
        """Initialize Mixture IRT model.

        Parameters
        ----------
        n_items : int
            Number of items
        n_classes : int
            Number of latent classes (default: 2)
        base_model : str
            Base IRT model for each class ('1PL', '2PL', or '3PL')
        item_names : list, optional
            Item names
        """
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")

        self._n_classes = n_classes
        self._base_model = base_model

        super().__init__(n_items=n_items, n_factors=1, item_names=item_names)

    @property
    def n_classes(self) -> int:
        """Number of latent classes."""
        return self._n_classes

    @property
    def base_model(self) -> str:
        """Base IRT model type."""
        return self._base_model

    def _initialize_parameters(self) -> None:
        """Initialize parameters for all classes."""
        self._parameters["class_proportions"] = (
            np.ones(self._n_classes) / self._n_classes
        )

        for k in range(self._n_classes):
            if self._base_model != "1PL":
                self._parameters[f"discrimination_class{k}"] = np.ones(
                    self.n_items, dtype=np.float64
                )

            offset = (k - (self._n_classes - 1) / 2) * 0.5
            self._parameters[f"difficulty_class{k}"] = (
                np.zeros(self.n_items, dtype=np.float64) + offset
            )

            if self._base_model == "3PL":
                self._parameters[f"guessing_class{k}"] = np.full(
                    self.n_items, 0.2, dtype=np.float64
                )

    @property
    def class_proportions(self) -> NDArray[np.float64]:
        """Class mixing proportions."""
        return self._parameters["class_proportions"]

    def get_class_parameters(self, class_idx: int) -> dict[str, NDArray[np.float64]]:
        """Get parameters for a specific class.

        Parameters
        ----------
        class_idx : int
            Class index

        Returns
        -------
        dict
            Dictionary with discrimination, difficulty, and optionally guessing
        """
        params = {}

        if self._base_model != "1PL":
            params["discrimination"] = self._parameters[
                f"discrimination_class{class_idx}"
            ]
        else:
            params["discrimination"] = np.ones(self.n_items)

        params["difficulty"] = self._parameters[f"difficulty_class{class_idx}"]

        if self._base_model == "3PL":
            params["guessing"] = self._parameters[f"guessing_class{class_idx}"]
        else:
            params["guessing"] = np.zeros(self.n_items)

        return params

    def class_probability(
        self,
        theta: NDArray[np.float64],
        class_idx: int,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability for a specific class.

        Parameters
        ----------
        theta : NDArray
            Ability values
        class_idx : int
            Class index
        item_idx : int, optional
            Item index

        Returns
        -------
        NDArray
            Response probabilities
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        params = self.get_class_parameters(class_idx)
        a = params["discrimination"]
        b = params["difficulty"]
        c = params["guessing"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_2pl = 1.0 / (1.0 + np.exp(-z))
            return c[item_idx] + (1 - c[item_idx]) * p_2pl

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_2pl = 1.0 / (1.0 + np.exp(-z))
        return c[None, :] + (1 - c[None, :]) * p_2pl

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal response probability (averaged over classes).

        P(X = 1 | theta) = sum_k pi_k * P_k(X = 1 | theta)

        Parameters
        ----------
        theta : NDArray
            Ability values
        item_idx : int, optional
            Item index

        Returns
        -------
        NDArray
            Marginal response probabilities
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        pi = self._parameters["class_proportions"]

        if item_idx is not None:
            probs = np.zeros(n_persons)
            for k in range(self._n_classes):
                probs += pi[k] * self.class_probability(theta, k, item_idx)
            return probs

        probs = np.zeros((n_persons, self.n_items))
        for k in range(self._n_classes):
            probs += pi[k] * self.class_probability(theta, k, None)
        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for each person."""
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        responses.shape[0]

        probs = self.probability(theta)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(probs) + (1 - responses) * np.log(1 - probs),
            0.0,
        )

        return ll.sum(axis=1)

    def class_posterior(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute posterior probability of class membership.

        P(C = k | X, theta) proportional to pi_k * prod_j P_k(X_j | theta)

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        theta : NDArray
            Ability estimates

        Returns
        -------
        NDArray
            Posterior probabilities (n_persons, n_classes)
        """
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = responses.shape[0]

        pi = self._parameters["class_proportions"]

        log_likes = np.zeros((n_persons, self._n_classes))

        for k in range(self._n_classes):
            probs_k = self.class_probability(theta, k, None)
            probs_k = np.clip(probs_k, 1e-10, 1 - 1e-10)

            valid = responses >= 0
            ll_k = np.where(
                valid,
                responses * np.log(probs_k) + (1 - responses) * np.log(1 - probs_k),
                0.0,
            )
            log_likes[:, k] = ll_k.sum(axis=1) + np.log(pi[k] + 1e-10)

        log_sum = np.logaddexp.reduce(log_likes, axis=1, keepdims=True)
        posterior = np.exp(log_likes - log_sum)

        return posterior

    def classify_persons(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Classify persons into latent classes.

        Parameters
        ----------
        responses : NDArray
            Response matrix
        theta : NDArray
            Ability estimates

        Returns
        -------
        NDArray
            Class assignments (n_persons,)
        """
        posterior = self.class_posterior(responses, theta)
        return np.argmax(posterior, axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal Fisher information."""
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        avg_a_sq = np.zeros(self.n_items)
        pi = self._parameters["class_proportions"]

        for k in range(self._n_classes):
            params = self.get_class_parameters(k)
            avg_a_sq += pi[k] * params["discrimination"] ** 2

        if item_idx is not None:
            return avg_a_sq[item_idx] * p * q
        else:
            return avg_a_sq[None, :] * p * q

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = MixtureIRT(
            n_items=self.n_items,
            n_classes=self._n_classes,
            base_model=self._base_model,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


def fit_mixture_irt(
    responses: NDArray[np.int_],
    n_classes: int = 2,
    base_model: Literal["1PL", "2PL", "3PL"] = "2PL",
    max_iter: int = 100,
    tol: float = 1e-4,
    n_quadpts: int = 21,
    verbose: bool = False,
) -> tuple[MixtureIRT, NDArray[np.float64]]:
    """Fit mixture IRT model using EM algorithm.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    n_classes : int
        Number of latent classes
    base_model : str
        Base IRT model type
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    n_quadpts : int
        Number of quadrature points
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (fitted_model, class_posteriors)
    """
    from scipy.special import roots_hermite

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    model = MixtureIRT(
        n_items=n_items,
        n_classes=n_classes,
        base_model=base_model,
    )
    model._initialize_parameters()

    nodes, weights = roots_hermite(n_quadpts)
    weights = weights / np.sqrt(np.pi)
    nodes = nodes * np.sqrt(2)

    prev_ll = -np.inf

    for iteration in range(max_iter):
        class_posteriors = np.zeros((n_persons, n_classes))

        for k in range(n_classes):
            log_like_k = np.zeros(n_persons)

            for q in range(n_quadpts):
                theta_q = np.full((n_persons, 1), nodes[q])
                probs = model.class_probability(theta_q, k, None)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)

                valid = responses >= 0
                ll_q = np.where(
                    valid,
                    responses * np.log(probs) + (1 - responses) * np.log(1 - probs),
                    0.0,
                ).sum(axis=1)

                log_like_k += weights[q] * np.exp(ll_q)

            class_posteriors[:, k] = model.class_proportions[k] * log_like_k

        row_sums = class_posteriors.sum(axis=1, keepdims=True)
        class_posteriors = class_posteriors / (row_sums + 1e-10)

        model._parameters["class_proportions"] = class_posteriors.mean(axis=0)

        for k in range(n_classes):
            weights_k = class_posteriors[:, k]

            for j in range(n_items):
                valid = responses[:, j] >= 0
                if valid.sum() > 0:
                    weighted_mean = np.average(
                        responses[valid, j], weights=weights_k[valid]
                    )
                    weighted_mean = np.clip(weighted_mean, 0.01, 0.99)
                    model._parameters[f"difficulty_class{k}"][j] = -np.log(
                        weighted_mean / (1 - weighted_mean)
                    )

        current_ll = np.sum(np.log(row_sums + 1e-10))

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if abs(current_ll - prev_ll) < tol:
            break

        prev_ll = current_ll

    model._is_fitted = True

    return model, class_posteriors
