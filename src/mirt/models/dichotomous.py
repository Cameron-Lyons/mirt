"""Dichotomous IRT models: 1PL, 2PL, 3PL, 4PL."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class TwoParameterLogistic(DichotomousItemModel):
    """Two-Parameter Logistic (2PL) IRT Model.

    The 2PL model includes discrimination (a) and difficulty (b) parameters:

    P(X=1|θ) = 1 / (1 + exp(-a * (θ - b)))

    For multidimensional models, the formula becomes:

    P(X=1|θ) = 1 / (1 + exp(-(a·θ + d)))

    where a is a vector of slopes and d is the intercept.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_factors : int, default=1
        Number of latent factors (dimensions).
    item_names : list of str, optional
        Names for each item.

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters. Shape (n_items,) for unidimensional,
        (n_items, n_factors) for multidimensional.
    difficulty : ndarray of shape (n_items,)
        Item difficulty parameters.

    Examples
    --------
    >>> import numpy as np
    >>> model = TwoParameterLogistic(n_items=10)
    >>> theta = np.linspace(-3, 3, 100)
    >>> probs = model.probability(theta)
    >>> print(probs.shape)
    (100, 10)
    """

    model_name = "2PL"
    n_params_per_item = 2
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            # For multidimensional, initialize with small random loadings
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination (slope) parameters."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulty (location) parameters."""
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute P(X=1|θ) for items.

        Parameters
        ----------
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.
        item_idx : int, optional
            If specified, compute probability for this item only.

        Returns
        -------
        ndarray
            Shape (n_persons,) if item_idx specified, else (n_persons, n_items).
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if self.n_factors == 1:
            # Unidimensional case
            theta_1d = theta.ravel()

            if item_idx is not None:
                z = a[item_idx] * (theta_1d - b[item_idx])
                return 1.0 / (1.0 + np.exp(-z))

            # All items: theta_1d[:, None] broadcasts to (n_persons, n_items)
            z = a[None, :] * (theta_1d[:, None] - b[None, :])
            return 1.0 / (1.0 + np.exp(-z))

        else:
            # Multidimensional case: z = a·θ - d
            # a shape: (n_items, n_factors), theta shape: (n_persons, n_factors)
            # Result should be (n_persons, n_items)

            if item_idx is not None:
                # Single item: a[item_idx] is (n_factors,)
                z = np.dot(theta, a[item_idx]) - a[item_idx].sum() * b[item_idx]
                return 1.0 / (1.0 + np.exp(-z))

            # All items: compute θ @ a.T to get (n_persons, n_items)
            # For slope-intercept form: use d = -a·b as intercept
            z = np.dot(theta, a.T) - np.sum(a, axis=1) * b
            return 1.0 / (1.0 + np.exp(-z))

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for given responses.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Binary response matrix (0 or 1). Missing values coded as -1.
        theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
            Latent trait values.

        Returns
        -------
        ndarray of shape (n_persons,)
            Log-likelihood for each person.
        """
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        if responses.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}"
            )

        p = self.probability(theta)

        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        # Handle missing responses (coded as -1)
        valid = responses >= 0

        # Log-likelihood: X*log(P) + (1-X)*log(1-P)
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information.

        For 2PL: I(θ) = a² × P(θ) × (1 - P(θ))

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int, optional
            If specified, compute information for this item only.

        Returns
        -------
        ndarray
            Fisher information values.
        """
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            if self.n_factors == 1:
                a_val = a[item_idx]
            else:
                # For multidimensional, use sum of squared slopes
                a_val = np.sqrt(np.sum(a[item_idx] ** 2))
            return (a_val ** 2) * p * q

        # All items
        if self.n_factors == 1:
            return (a[None, :] ** 2) * p * q
        else:
            a_sq = np.sum(a ** 2, axis=1)  # (n_items,)
            return a_sq[None, :] * p * q

    def test_information(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute total test information.

        Parameters
        ----------
        theta : ndarray
            Latent trait values.

        Returns
        -------
        ndarray of shape (n_persons,)
            Total test information at each theta.
        """
        item_info = self.information(theta)
        return item_info.sum(axis=1)


class OneParameterLogistic(TwoParameterLogistic):
    """One-Parameter Logistic (1PL/Rasch) Model.

    The 1PL model (also known as the Rasch model) has only difficulty parameters,
    with all discriminations fixed to 1:

    P(X=1|θ) = 1 / (1 + exp(-(θ - b)))

    This is a constrained version of the 2PL model.

    Parameters
    ----------
    n_items : int
        Number of items.
    item_names : list of str, optional
        Names for each item.

    Notes
    -----
    The 1PL model does not support multidimensional analysis.
    """

    model_name = "1PL"
    n_params_per_item = 1
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("1PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters. Discrimination fixed to 1."""
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    def set_parameters(self, **params: NDArray[np.float64]) -> "OneParameterLogistic":
        """Set parameters. Discrimination cannot be changed in 1PL."""
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in 1PL model (fixed to 1)")
        return super().set_parameters(**params)


class ThreeParameterLogistic(DichotomousItemModel):
    """Three-Parameter Logistic (3PL) Model.

    The 3PL model adds a lower asymptote (guessing) parameter:

    P(X=1|θ) = c + (1-c) / (1 + exp(-a * (θ - b)))

    where:
    - a: discrimination parameter
    - b: difficulty parameter
    - c: lower asymptote (pseudo-guessing) parameter, 0 ≤ c < 1

    Parameters
    ----------
    n_items : int
        Number of items.
    item_names : list of str, optional
        Names for each item.

    Notes
    -----
    The 3PL model is typically used for multiple-choice items where
    guessing is possible. It does not support multidimensional analysis.
    """

    model_name = "3PL"
    n_params_per_item = 3
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("3PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulty parameters."""
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        """Lower asymptote (guessing) parameters."""
        return self._parameters["guessing"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute P(X=1|θ) for 3PL model.

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int, optional
            If specified, compute for this item only.

        Returns
        -------
        ndarray
            Response probabilities.
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_star = 1.0 / (1.0 + np.exp(-z))
            return c[item_idx] + (1.0 - c[item_idx]) * p_star

        # All items
        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = 1.0 / (1.0 + np.exp(-z))
        return c[None, :] + (1.0 - c[None, :]) * p_star

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for 3PL model."""
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        if responses.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}"
            )

        p = self.probability(theta)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information for 3PL model.

        I(θ) = a² × [(P - c)² / ((1-c)² × P × (1-P))]

        where P is the probability of correct response.
        """
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        a = self._parameters["discrimination"]
        c = self._parameters["guessing"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            numerator = (a_val ** 2) * ((p - c_val) ** 2)
            denominator = ((1 - c_val) ** 2) * p * (1 - p) + 1e-10
            return numerator / denominator

        # All items
        numerator = (a[None, :] ** 2) * ((p - c[None, :]) ** 2)
        denominator = ((1 - c[None, :]) ** 2) * p * (1 - p) + 1e-10
        return numerator / denominator


class FourParameterLogistic(DichotomousItemModel):
    """Four-Parameter Logistic (4PL) Model.

    The 4PL model includes both lower and upper asymptotes:

    P(X=1|θ) = c + (d-c) / (1 + exp(-a * (θ - b)))

    where:
    - a: discrimination parameter
    - b: difficulty parameter
    - c: lower asymptote (guessing), 0 ≤ c < 1
    - d: upper asymptote (carelessness/inattention), c < d ≤ 1

    Parameters
    ----------
    n_items : int
        Number of items.
    item_names : list of str, optional
        Names for each item.

    Notes
    -----
    The 4PL model is rarely used in practice due to estimation difficulties.
    It can model both guessing behavior and careless errors.
    """

    model_name = "4PL"
    n_params_per_item = 4
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("4PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)
        self._parameters["upper"] = np.ones(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulty parameters."""
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        """Lower asymptote (guessing) parameters."""
        return self._parameters["guessing"]

    @property
    def upper(self) -> NDArray[np.float64]:
        """Upper asymptote parameters."""
        return self._parameters["upper"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute P(X=1|θ) for 4PL model."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_star = 1.0 / (1.0 + np.exp(-z))
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        # All items
        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = 1.0 / (1.0 + np.exp(-z))
        return c[None, :] + (d[None, :] - c[None, :]) * p_star

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for 4PL model."""
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        if responses.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}"
            )

        p = self.probability(theta)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information for 4PL model.

        I(θ) = a² × [(P - c)² × (d - P)²] / [(d-c)² × P × (1-P)]
        """
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        a = self._parameters["discrimination"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            d_val = d[item_idx]
            numerator = (a_val ** 2) * ((p - c_val) ** 2) * ((d_val - p) ** 2)
            denominator = ((d_val - c_val) ** 2) * p * (1 - p) + 1e-10
            return numerator / denominator

        # All items
        numerator = (a[None, :] ** 2) * ((p - c[None, :]) ** 2) * ((d[None, :] - p) ** 2)
        denominator = ((d[None, :] - c[None, :]) ** 2) * p * (1 - p) + 1e-10
        return numerator / denominator


# Aliases for convenience
Rasch = OneParameterLogistic
