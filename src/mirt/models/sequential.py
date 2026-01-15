"""Sequential response models for polytomous items.

Sequential models treat polytomous responses as a series of dichotomous
decisions. This is appropriate when the response process involves
sequential stages (e.g., completion models, mastery learning).

References:
    Tutz, G. (1990). Sequential item response models with an ordered
        response. British Journal of Mathematical and Statistical Psychology.
    Verhelst, N. D., et al. (1997). A logistic model for time-limit tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import PolytomousItemModel


class SequentialResponseModel(PolytomousItemModel):
    """Sequential Response Model for polytomous items.

    In the Sequential model, responding in category k requires "passing"
    all k-1 previous thresholds. This is appropriate for processes where
    higher categories require cumulative mastery.

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int or list of int
        Number of response categories per item
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters
    thresholds : ndarray
        Sequential threshold parameters for each step

    Notes
    -----
    The probability of category k is:

        P(X = k) = P(pass 1) * P(pass 2) * ... * P(pass k) * P(fail k+1)

    where P(pass j) = 1 / (1 + exp(-a(θ - b_j)))

    This differs from GRM/GPCM where responses don't require sequential
    threshold crossing.
    """

    model_name = "Sequential"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Sequential model only supports unidimensional analysis")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._parameters["thresholds"]

    def _step_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        step_idx: int,
    ) -> NDArray[np.float64]:
        """Probability of passing step k given reached step k."""
        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["thresholds"][item_idx, step_idx]

        z = a * (theta - b)
        return sigmoid(z)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute probability of responding in category k.

        P(X = k) = prod_{j<k} P(pass j) * P(fail k)
        where P(fail k) = 1 for k = K-1 (highest category)
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        step_probs = []
        for j in range(n_cat - 1):
            step_probs.append(self._step_probability(theta_1d, item_idx, j))

        if category == 0:
            return 1.0 - step_probs[0]

        prob = np.ones_like(theta_1d)

        for j in range(category):
            prob *= step_probs[j]

        if category < n_cat - 1:
            prob *= 1.0 - step_probs[category]

        return prob

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute item information function."""
        n_cat = self._n_categories[item_idx]
        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)
        expected_sq = np.sum(probs * (categories**2), axis=1)
        variance = expected_sq - expected**2

        a = self._parameters["discrimination"][item_idx]
        return (a**2) * variance


class ContinuationRatioModel(PolytomousItemModel):
    """Continuation Ratio Model for ordinal responses.

    The CR model is appropriate when responses represent a stopping process
    (e.g., how far someone progresses before stopping).

    Each step models P(X >= k | X >= k-1).

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int or list of int
        Number of categories per item
    item_names : list of str, optional
        Names for items
    """

    model_name = "ContinuationRatio"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Continuation Ratio only supports unidimensional")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._parameters["thresholds"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute probability of category k using continuation ratios."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["thresholds"][item_idx]

        cr_probs = []
        for k in range(n_cat - 1):
            z = a * (theta_1d - b[k])
            cr_probs.append(sigmoid(z))

        if category == 0:
            return 1.0 - cr_probs[0]

        p_geq_k = np.ones_like(theta_1d)
        for j in range(category):
            p_geq_k *= cr_probs[j]

        if category == n_cat - 1:
            return p_geq_k
        else:
            return p_geq_k * (1.0 - cr_probs[category])

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute item information."""
        n_cat = self._n_categories[item_idx]
        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)
        expected_sq = np.sum(probs * (categories**2), axis=1)
        variance = expected_sq - expected**2

        a = self._parameters["discrimination"][item_idx]
        return (a**2) * variance


class AdjacentCategoryModel(PolytomousItemModel):
    """Adjacent Category Model for ordinal responses.

    Models the log-odds of adjacent categories:
    log(P(X=k) / P(X=k-1)) = a(theta - b_k)

    This is sometimes called the "adjacent categories logit" model.
    """

    model_name = "AdjacentCategory"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Adjacent Category only supports unidimensional")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._parameters["thresholds"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute category probability using adjacent category formulation."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_persons = len(theta_1d)
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["thresholds"][item_idx]

        log_ratios = np.zeros((n_persons, n_cat))
        for k in range(1, n_cat):
            log_ratios[:, k] = log_ratios[:, k - 1] + a * (theta_1d - b[k - 1])

        log_ratios_max = np.max(log_ratios, axis=1, keepdims=True)
        exp_ratios = np.exp(log_ratios - log_ratios_max)
        probs = exp_ratios / np.sum(exp_ratios, axis=1, keepdims=True)

        return probs[:, category]

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute item information."""
        n_cat = self._n_categories[item_idx]
        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)
        expected_sq = np.sum(probs * (categories**2), axis=1)
        variance = expected_sq - expected**2

        a = self._parameters["discrimination"][item_idx]
        return (a**2) * variance
