"""Partially compensatory, conjunctive, and disjunctive IRT models.

References
----------
Bolt, D. M., & Lall, V. F. (2003). Estimation of compensatory and
noncompensatory multidimensional item response models using Markov chain
Monte Carlo. Applied Psychological Measurement.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import DichotomousItemModel


def _scalar_fisher_information(
    probability: NDArray[np.float64],
    gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the trace of Bernoulli Fisher information matrices."""
    denominator = probability * (1.0 - probability)
    numerator = np.sum(gradient**2, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(probability, dtype=np.float64),
        where=denominator > 0,
    )


def _matrix_fisher_information(
    probability: NDArray[np.float64],
    gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Bernoulli Fisher information matrices from probability gradients."""
    denominator = probability * (1.0 - probability)
    outer_gradient = gradient[..., :, None] * gradient[..., None, :]
    return np.divide(
        outer_gradient,
        denominator[..., None, None],
        out=np.zeros_like(outer_gradient, dtype=np.float64),
        where=denominator[..., None, None] > 0,
    )


class _LogicalMultidimensionalModel(DichotomousItemModel):
    """Shared vectorized behavior for AND/OR multidimensional models."""

    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        shape = (self.n_items, self.n_factors)
        self._parameters["discrimination"] = np.ones(shape)
        self._parameters["difficulty"] = np.zeros(shape)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def _factor_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> NDArray[np.float64]:
        discrimination = self._parameters["discrimination"]
        difficulty = self._parameters["difficulty"]

        if item_idx is not None:
            z = discrimination[item_idx] * (theta - difficulty[item_idx])
            return sigmoid(z)

        z = discrimination[None, :, :] * (theta[:, None, :] - difficulty[None, :, :])
        return sigmoid(z)

    def _probability_and_gradient(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError

    def _probability_only(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        return self._probability_only(theta, item_idx)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return scalar item information as the Fisher-matrix trace."""
        theta = self._ensure_theta_2d(theta)
        probability, gradient = self._probability_and_gradient(theta, item_idx)
        return _scalar_fisher_information(probability, gradient)

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return item Fisher matrices with shape ``(n_theta, n_factors, n_factors)``."""
        theta = self._ensure_theta_2d(theta)
        probability, gradient = self._probability_and_gradient(theta, item_idx)
        return _matrix_fisher_information(probability, gradient)

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return summed test Fisher matrices across all items."""
        theta = self._ensure_theta_2d(theta)
        probability, gradient = self._probability_and_gradient(theta, None)
        item_matrices = _matrix_fisher_information(probability, gradient)
        return np.sum(item_matrices, axis=1)


class PartiallyCompensatoryModel(_LogicalMultidimensionalModel):
    """Weighted conjunctive multidimensional IRT model.

    The response function is

    ``P(X=1 | theta) = product_k logistic_k(theta) ** w_k``.

    Each compensation weight ``w_k`` lies in ``[0, 1]``. A weight of zero
    fully compensates for (ignores) that dimension, while one fully enforces
    its conjunctive contribution. Setting every weight to one reproduces
    :class:`NoncompensatoryModel`.
    """

    model_name = "PartiallyCompensatory"

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Partially compensatory model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self._parameters["compensation"] = np.full(
            (self.n_items, self.n_factors),
            0.5,
        )

    @property
    def compensation(self) -> NDArray[np.float64]:
        """Return per-item dimension contribution weights."""
        return self._parameters["compensation"]

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        compensation = params.get("compensation")
        if compensation is not None:
            weights = np.asarray(compensation, dtype=np.float64)
            if not np.all(np.isfinite(weights)) or np.any(
                (weights < 0.0) | (weights > 1.0)
            ):
                raise ValueError("compensation weights must be finite values in [0, 1]")
        return super().set_parameters(**params)

    def _probability_and_gradient(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        factor_probability = self._factor_probabilities(theta, item_idx)
        discrimination = self._parameters["discrimination"]
        compensation = self._parameters["compensation"]

        if item_idx is not None:
            discrimination = discrimination[item_idx]
            compensation = compensation[item_idx]

        probability = np.prod(
            np.power(factor_probability, compensation),
            axis=-1,
        )
        gradient = (
            probability[..., None]
            * compensation
            * discrimination
            * (1.0 - factor_probability)
        )
        return probability, gradient

    def _probability_only(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> NDArray[np.float64]:
        discrimination = self._parameters["discrimination"]
        difficulty = self._parameters["difficulty"]
        compensation = self._parameters["compensation"]

        if item_idx is not None:
            factor_probability = sigmoid(
                discrimination[item_idx] * (theta - difficulty[item_idx])
            )
            return np.prod(
                np.power(factor_probability, compensation[item_idx]),
                axis=1,
            )

        probability = np.ones((theta.shape[0], self.n_items), dtype=np.float64)
        for factor_idx in range(self.n_factors):
            factor_probability = sigmoid(
                discrimination[None, :, factor_idx]
                * (theta[:, None, factor_idx] - difficulty[None, :, factor_idx])
            )
            probability *= np.power(
                factor_probability,
                compensation[None, :, factor_idx],
            )
        return probability


class NoncompensatoryModel(_LogicalMultidimensionalModel):
    """Fully noncompensatory (conjunctive) multidimensional IRT model.

    Success requires meeting the threshold on every dimension:

    ``P(X=1 | theta) = product_k logistic_k(theta)``.

    This is the limiting case of :class:`PartiallyCompensatoryModel` with all
    compensation weights equal to one.
    """

    model_name = "Noncompensatory"

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Non-compensatory model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _probability_and_gradient(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        factor_probability = self._factor_probabilities(theta, item_idx)
        discrimination = self._parameters["discrimination"]
        if item_idx is not None:
            discrimination = discrimination[item_idx]

        probability = np.prod(factor_probability, axis=-1)
        gradient = probability[..., None] * discrimination * (1.0 - factor_probability)
        return probability, gradient

    def _probability_only(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> NDArray[np.float64]:
        discrimination = self._parameters["discrimination"]
        difficulty = self._parameters["difficulty"]

        if item_idx is not None:
            factor_probability = sigmoid(
                discrimination[item_idx] * (theta - difficulty[item_idx])
            )
            return np.prod(factor_probability, axis=1)

        probability = np.ones((theta.shape[0], self.n_items), dtype=np.float64)
        for factor_idx in range(self.n_factors):
            probability *= sigmoid(
                discrimination[None, :, factor_idx]
                * (theta[:, None, factor_idx] - difficulty[None, :, factor_idx])
            )
        return probability


class DisjunctiveModel(_LogicalMultidimensionalModel):
    """Disjunctive multidimensional IRT model with multiple paths to success.

    Success requires meeting the threshold on any dimension:

    ``P(X=1 | theta) = 1 - product_k (1 - logistic_k(theta))``.
    """

    model_name = "Disjunctive"

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Disjunctive model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _probability_and_gradient(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        factor_probability = self._factor_probabilities(theta, item_idx)
        discrimination = self._parameters["discrimination"]
        if item_idx is not None:
            discrimination = discrimination[item_idx]

        failure_probability = np.prod(1.0 - factor_probability, axis=-1)
        probability = 1.0 - failure_probability
        gradient = failure_probability[..., None] * discrimination * factor_probability
        return probability, gradient

    def _probability_only(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> NDArray[np.float64]:
        discrimination = self._parameters["discrimination"]
        difficulty = self._parameters["difficulty"]

        if item_idx is not None:
            factor_probability = sigmoid(
                discrimination[item_idx] * (theta - difficulty[item_idx])
            )
            return 1.0 - np.prod(1.0 - factor_probability, axis=1)

        failure_probability = np.ones(
            (theta.shape[0], self.n_items),
            dtype=np.float64,
        )
        for factor_idx in range(self.n_factors):
            factor_probability = sigmoid(
                discrimination[None, :, factor_idx]
                * (theta[:, None, factor_idx] - difficulty[None, :, factor_idx])
            )
            failure_probability *= 1.0 - factor_probability
        return 1.0 - failure_probability
