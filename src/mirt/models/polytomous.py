"""Polytomous IRT models: GRM, GPCM, PCM, NRM."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import PolytomousItemModel


class GradedResponseModel(PolytomousItemModel):
    """Graded Response Model (GRM) - Samejima (1969).

    The GRM is a cumulative logit model for ordered polytomous responses.
    It models the probability of responding in category k or higher:

    P*(X ≥ k|θ) = 1 / (1 + exp(-a(θ - b_k)))

    The probability of responding in exactly category k is:

    P(X = k|θ) = P*(X ≥ k|θ) - P*(X ≥ k+1|θ)

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list of int
        Number of response categories per item. Categories are 0, 1, ..., K-1.
    n_factors : int, default=1
        Number of latent factors.
    item_names : list of str, optional
        Names for each item.

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination (slope) parameters.
    thresholds : ndarray
        Threshold (difficulty) parameters for each category boundary.

    Examples
    --------
    >>> # 10 items with 5 categories each (e.g., Likert scale 1-5)
    >>> model = GradedResponseModel(n_items=10, n_categories=5)
    >>> theta = np.linspace(-3, 3, 100)
    >>> probs = model.probability(theta, item_idx=0)
    >>> print(probs.shape)  # (100, 5)
    """

    model_name = "GRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        # Discrimination parameters
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        # Threshold parameters: K-1 thresholds for K categories
        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            # Initialize with evenly spaced thresholds
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Item threshold parameters."""
        return self._parameters["thresholds"]

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        """Compute P*(X ≥ k|θ) - cumulative probability.

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int
            Item index.
        threshold_idx : int
            Threshold index (0 to n_categories-2).

        Returns
        -------
        ndarray of shape (n_persons,)
            Cumulative probability.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        b = self._parameters["thresholds"][item_idx, threshold_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            z = a_item * (theta.ravel() - b)
        else:
            a_item = a[item_idx]  # (n_factors,)
            z = np.dot(theta, a_item) - np.sum(a_item) * b

        return 1.0 / (1.0 + np.exp(-z))

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute P(X = k|θ) for a specific category.

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int
            Item index.
        category : int
            Category index (0 to n_categories-1).

        Returns
        -------
        ndarray of shape (n_persons,)
            Category probability.
        """
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        if category == 0:
            # P(X=0) = 1 - P*(X≥1)
            return 1.0 - self.cumulative_probability(theta, item_idx, 0)
        elif category == n_cat - 1:
            # P(X=K-1) = P*(X≥K-1)
            return self.cumulative_probability(theta, item_idx, category - 1)
        else:
            # P(X=k) = P*(X≥k) - P*(X≥k+1)
            p_upper = self.cumulative_probability(theta, item_idx, category - 1)
            p_lower = self.cumulative_probability(theta, item_idx, category)
            return p_upper - p_lower

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities.

        Parameters
        ----------
        theta : ndarray
            Latent trait values.
        item_idx : int, optional
            If specified, return probabilities for one item.
            Otherwise, return for all items.

        Returns
        -------
        ndarray
            If item_idx specified: shape (n_persons, n_categories).
            Otherwise: shape (n_persons, n_items, max_categories).
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        # All items
        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for given responses.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix. Responses should be 0, 1, ..., K-1.
            Missing values coded as -1.
        theta : ndarray
            Latent trait values.

        Returns
        -------
        ndarray of shape (n_persons,)
            Log-likelihood for each person.
        """
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:  # Valid response
                    prob = self.category_probability(theta[person : person + 1], i, resp)
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information.

        For GRM, information is computed as:
        I(θ) = Σ_k [P'_k(θ)]² / P_k(θ)

        where P'_k is the derivative of category probability.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        # Total test information
        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute information for a single item."""
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        # Get category probabilities
        probs = self.probability(theta, item_idx)  # (n_persons, n_cat)

        # Compute cumulative probabilities
        cum_probs = np.zeros((n_persons, n_cat + 1))
        cum_probs[:, 0] = 1.0
        for k in range(n_cat - 1):
            cum_probs[:, k + 1] = self.cumulative_probability(theta, item_idx, k)
        cum_probs[:, n_cat] = 0.0

        # Information formula for GRM
        info = np.zeros(n_persons)
        for k in range(n_cat):
            p_k = probs[:, k]
            p_star_k = cum_probs[:, k]
            p_star_k1 = cum_probs[:, k + 1]

            # Derivative of category probability
            dp_k = a_val * (p_star_k * (1 - p_star_k) - p_star_k1 * (1 - p_star_k1))

            # Add to information (avoid division by zero)
            info += np.where(p_k > 1e-10, (dp_k ** 2) / p_k, 0.0)

        return info


class GeneralizedPartialCredit(PolytomousItemModel):
    """Generalized Partial Credit Model (GPCM) - Muraki (1992).

    The GPCM is an adjacent-category logit model with item-specific
    discrimination parameters:

    P(X = k|θ) = exp(Σ_{v=0}^{k} a(θ - b_v)) / Σ_{c=0}^{K-1} exp(Σ_{v=0}^{c} a(θ - b_v))

    where b_0 = 0 by convention.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list of int
        Number of response categories per item.
    n_factors : int, default=1
        Number of latent factors.
    item_names : list of str, optional
        Names for each item.

    Examples
    --------
    >>> model = GeneralizedPartialCredit(n_items=10, n_categories=5)
    """

    model_name = "GPCM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        # Discrimination
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        # Step parameters: K-1 steps for K categories
        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def steps(self) -> NDArray[np.float64]:
        """Step difficulty parameters."""
        return self._parameters["steps"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute P(X = k|θ) for a specific category."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"]
        b = self._parameters["steps"][item_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            theta_1d = theta.ravel()
        else:
            a_item = a[item_idx]
            theta_1d = np.dot(theta, a_item)
            a_item = np.sqrt(np.sum(a_item ** 2))

        # Compute numerators for all categories
        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            # Sum of a*(theta - b_v) for v = 1 to k
            cumsum = 0.0
            for v in range(k):
                cumsum += a_item * (theta_1d - b[v])
            numerators[:, k] = np.exp(cumsum)

        # Denominator is sum of all numerators
        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for given responses."""
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(theta[person : person + 1], i, resp)
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information for GPCM."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute information for a single item."""
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        probs = self.probability(theta, item_idx)  # (n_persons, n_cat)

        # Expected category: E[X]
        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)

        # Expected squared category: E[X²]
        expected_sq = np.sum(probs * (categories ** 2), axis=1)

        # Variance = E[X²] - E[X]²
        variance = expected_sq - expected ** 2

        # Information = a² × Var(X)
        return (a_val ** 2) * variance


class PartialCreditModel(GeneralizedPartialCredit):
    """Partial Credit Model (PCM) - Masters (1982).

    The PCM is a special case of the GPCM with all discriminations
    fixed to 1. It is an adjacent-category Rasch model.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list of int
        Number of response categories per item.
    item_names : list of str, optional
        Names for each item.
    """

    model_name = "PCM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: Optional[list[str]] = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("PCM only supports unidimensional analysis")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters. Discrimination fixed to 1."""
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    def set_parameters(self, **params: NDArray[np.float64]) -> "PartialCreditModel":
        """Set parameters. Discrimination cannot be changed."""
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in PCM (fixed to 1)")
        return super().set_parameters(**params)


class NominalResponseModel(PolytomousItemModel):
    """Nominal Response Model (NRM) - Bock (1972).

    The NRM is designed for nominal (unordered) polytomous responses.
    Each category has its own slope and intercept:

    P(X = k|θ) = exp(a_k × θ + c_k) / Σ_j exp(a_j × θ + c_j)

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list of int
        Number of response categories per item.
    n_factors : int, default=1
        Number of latent factors.
    item_names : list of str, optional
        Names for each item.

    Notes
    -----
    The NRM requires identification constraints. By default, the first
    category's parameters are fixed to zero.
    """

    model_name = "NRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        max_cats = max(self._n_categories)

        # Category slopes (a_k): first category fixed to 0
        if self.n_factors == 1:
            slopes = np.zeros((self.n_items, max_cats))
            for i, n_cat in enumerate(self._n_categories):
                # Linear trend for initial values
                slopes[i, 1:n_cat] = np.linspace(0.5, 1.5, n_cat - 1)
        else:
            slopes = np.zeros((self.n_items, max_cats, self.n_factors))
            for i, n_cat in enumerate(self._n_categories):
                for f in range(self.n_factors):
                    slopes[i, 1:n_cat, f] = np.linspace(0.5, 1.5, n_cat - 1)

        self._parameters["slopes"] = slopes

        # Category intercepts (c_k): first category fixed to 0
        intercepts = np.zeros((self.n_items, max_cats))
        for i, n_cat in enumerate(self._n_categories):
            intercepts[i, 1:n_cat] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["intercepts"] = intercepts

    @property
    def slopes(self) -> NDArray[np.float64]:
        """Category slope parameters."""
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        """Category intercept parameters."""
        return self._parameters["intercepts"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute P(X = k|θ) for a specific category."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["slopes"]
        c = self._parameters["intercepts"]

        # Compute all category numerators
        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            if self.n_factors == 1:
                z = a[item_idx, k] * theta.ravel() + c[item_idx, k]
            else:
                z = np.dot(theta, a[item_idx, k]) + c[item_idx, k]
            numerators[:, k] = np.exp(z)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for given responses."""
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(theta[person : person + 1], i, resp)
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information for NRM."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute information for a single item."""
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        probs = self.probability(theta, item_idx)  # (n_persons, n_cat)

        if self.n_factors == 1:
            # Slopes for this item
            a_item = a[item_idx, :n_cat]  # (n_cat,)

            # E[a] = sum(a_k * P_k)
            expected_a = np.sum(probs * a_item, axis=1)

            # E[a²] = sum(a_k² * P_k)
            expected_a_sq = np.sum(probs * (a_item ** 2), axis=1)

            # Information = E[a²] - E[a]²
            info = expected_a_sq - expected_a ** 2
        else:
            # Multidimensional: compute trace of information matrix
            info = np.zeros(n_persons)
            for f in range(self.n_factors):
                a_f = a[item_idx, :n_cat, f]
                expected_a = np.sum(probs * a_f, axis=1)
                expected_a_sq = np.sum(probs * (a_f ** 2), axis=1)
                info += expected_a_sq - expected_a ** 2

        return info
