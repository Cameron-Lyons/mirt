import numpy as np
from numpy.typing import NDArray

from mirt.models.base import PolytomousItemModel


class GradedResponseModel(PolytomousItemModel):
    model_name = "GRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

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

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["discrimination"]
        b = self._parameters["thresholds"][item_idx, threshold_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            z = a_item * (theta.ravel() - b)
        else:
            a_item = a[item_idx]
            z = np.dot(theta, a_item) - np.sum(a_item) * b

        return 1.0 / (1.0 + np.exp(-z))

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        if category == 0:
            return 1.0 - self.cumulative_probability(theta, item_idx, 0)
        elif category == n_cat - 1:
            return self.cumulative_probability(theta, item_idx, category - 1)
        else:
            p_upper = self.cumulative_probability(theta, item_idx, category - 1)
            p_lower = self.cumulative_probability(theta, item_idx, category)
            return p_upper - p_lower

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        probs = self.probability(theta, item_idx)

        cum_probs = np.zeros((n_persons, n_cat + 1))
        cum_probs[:, 0] = 1.0
        for k in range(n_cat - 1):
            cum_probs[:, k + 1] = self.cumulative_probability(theta, item_idx, k)
        cum_probs[:, n_cat] = 0.0

        info = np.zeros(n_persons)
        for k in range(n_cat):
            p_k = probs[:, k]
            p_star_k = cum_probs[:, k]
            p_star_k1 = cum_probs[:, k + 1]

            dp_k = a_val * (p_star_k * (1 - p_star_k) - p_star_k1 * (1 - p_star_k1))

            info += np.where(p_k > 1e-10, (dp_k**2) / p_k, 0.0)

        return info


class GeneralizedPartialCredit(PolytomousItemModel):
    model_name = "GPCM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def steps(self) -> NDArray[np.float64]:
        return self._parameters["steps"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
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
            a_item = np.sqrt(np.sum(a_item**2))

        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            cumsum = 0.0
            for v in range(k):
                cumsum += a_item * (theta_1d - b[v])
            numerators[:, k] = np.exp(cumsum)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = a[item_idx]
        else:
            a_val = np.sqrt(np.sum(a[item_idx] ** 2))

        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)

        expected_sq = np.sum(probs * (categories**2), axis=1)

        variance = expected_sq - expected**2

        return (a_val**2) * variance


class PartialCreditModel(GeneralizedPartialCredit):
    model_name = "PCM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("PCM only supports unidimensional analysis")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    def set_parameters(self, **params: NDArray[np.float64]) -> "PartialCreditModel":
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in PCM (fixed to 1)")
        return super().set_parameters(**params)


class RatingScaleModel(PolytomousItemModel):
    """Rating Scale Model (RSM) for polytomous items.

    The RSM is a special case of the Partial Credit Model where step
    parameters are constrained to be equal across all items. This is
    appropriate when all items share the same rating scale structure
    (e.g., Likert scales with the same response options).

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int
        Number of response categories (must be same for all items)
    item_names : list of str, optional
        Names for each item

    Attributes
    ----------
    difficulty : ndarray of shape (n_items,)
        Item location/difficulty parameters
    thresholds : ndarray of shape (n_categories - 1,)
        Step thresholds shared across all items

    Notes
    -----
    The probability of responding in category k for item j is:

        P(X_j = k | theta) = exp(sum_{v=0}^{k} (theta - b_j - tau_v)) /
                             sum_{c=0}^{K} exp(sum_{v=0}^{c} (theta - b_j - tau_v))

    where b_j is the item difficulty and tau_v are the shared thresholds.

    The RSM reduces the number of parameters compared to GPCM/PCM,
    which can be beneficial when the assumption of equal thresholds
    is reasonable.

    References
    ----------
    Andrich, D. (1978). A rating formulation for ordered response categories.
        Psychometrika, 43(4), 561-573.
    """

    model_name = "RSM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("RSM only supports unidimensional analysis")

        # RSM requires all items to have the same number of categories
        if isinstance(n_categories, list):
            if len(set(n_categories)) != 1:
                raise ValueError(
                    "RSM requires all items to have the same number of categories"
                )
            n_categories = n_categories[0]

        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)
        self._n_cats = n_categories

    def _initialize_parameters(self) -> None:
        # Item difficulties
        self._parameters["difficulty"] = np.zeros(self.n_items)

        # Shared threshold parameters (n_categories - 1)
        n_thresholds = self._n_cats - 1
        self._parameters["thresholds"] = np.linspace(-1, 1, n_thresholds)

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulty/location parameters."""
        return self._parameters["difficulty"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Shared step threshold parameters."""
        return self._parameters["thresholds"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute probability of responding in a specific category.

        Parameters
        ----------
        theta : ndarray
            Ability values
        item_idx : int
            Item index
        category : int
            Response category (0 to n_categories - 1)

        Returns
        -------
        ndarray
            Probability of category response for each theta
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_cats

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        b_j = self._parameters["difficulty"][item_idx]
        tau = self._parameters["thresholds"]
        theta_1d = theta.ravel()

        # Compute numerators for all categories
        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            cumsum = 0.0
            for v in range(k):
                cumsum += theta_1d - b_j - tau[v]
            numerators[:, k] = np.exp(cumsum)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute item information function.

        Uses the variance of the item score as the information.
        """
        n_cat = self._n_cats
        probs = self.probability(theta, item_idx)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)
        expected_sq = np.sum(probs * (categories**2), axis=1)
        variance = expected_sq - expected**2

        # Information is variance (for Rasch-family models)
        return variance

    def set_parameters(self, **params: NDArray[np.float64]) -> "RatingScaleModel":
        """Set model parameters.

        Parameters
        ----------
        difficulty : ndarray of shape (n_items,)
            Item difficulty parameters
        thresholds : ndarray of shape (n_categories - 1,)
            Shared threshold parameters

        Returns
        -------
        self
        """
        for name, values in params.items():
            if name not in self._parameters:
                raise ValueError(f"Unknown parameter: {name}")
            values = np.asarray(values)
            if name == "difficulty" and values.shape != (self.n_items,):
                raise ValueError(f"difficulty must have shape ({self.n_items},)")
            if name == "thresholds" and values.shape != (self._n_cats - 1,):
                raise ValueError(f"thresholds must have shape ({self._n_cats - 1},)")
            self._parameters[name] = values

        self._is_fitted = True
        return self


class NominalResponseModel(PolytomousItemModel):
    model_name = "NRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        max_cats = max(self._n_categories)

        if self.n_factors == 1:
            slopes = np.zeros((self.n_items, max_cats))
            for i, n_cat in enumerate(self._n_categories):
                slopes[i, 1:n_cat] = np.linspace(0.5, 1.5, n_cat - 1)
        else:
            slopes = np.zeros((self.n_items, max_cats, self.n_factors))
            for i, n_cat in enumerate(self._n_categories):
                for f in range(self.n_factors):
                    slopes[i, 1:n_cat, f] = np.linspace(0.5, 1.5, n_cat - 1)

        self._parameters["slopes"] = slopes

        intercepts = np.zeros((self.n_items, max_cats))
        for i, n_cat in enumerate(self._n_categories):
            intercepts[i, 1:n_cat] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["intercepts"] = intercepts

    @property
    def slopes(self) -> NDArray[np.float64]:
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["slopes"]
        c = self._parameters["intercepts"]

        numerators = np.zeros((n_persons, n_cat))

        for k in range(n_cat):
            if self.n_factors == 1:
                z = a[item_idx, k] * theta.ravel() + c[item_idx, k]
            else:
                z = np.dot(theta, a[item_idx, k]) + c[item_idx, k]
            numerators[:, k] = np.exp(z)

        denominator = numerators.sum(axis=1)

        return numerators[:, category] / denominator

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        probs = self.probability(theta, item_idx)

        if self.n_factors == 1:
            a_item = a[item_idx, :n_cat]

            expected_a = np.sum(probs * a_item, axis=1)

            expected_a_sq = np.sum(probs * (a_item**2), axis=1)

            info = expected_a_sq - expected_a**2
        else:
            info = np.zeros(n_persons)
            for f in range(self.n_factors):
                a_f = a[item_idx, :n_cat, f]
                expected_a = np.sum(probs * a_f, axis=1)
                expected_a_sq = np.sum(probs * (a_f**2), axis=1)
                info += expected_a_sq - expected_a**2

        return info
