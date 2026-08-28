import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt._rust_backend import (
    RUST_AVAILABLE,
    compute_log_likelihoods_gpcm,
    compute_log_likelihoods_grm,
)
from mirt.constants import PROB_EPSILON
from mirt.models.base import PolytomousItemModel


def _stable_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute row-wise softmax without overflowing exponentials."""
    weights = logits - np.max(logits, axis=1, keepdims=True)
    np.exp(weights, out=weights)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _graded_information(
    probabilities: NDArray[np.float64],
    discrimination: float,
) -> NDArray[np.float64]:
    """Compute graded-response information from existing category curves."""
    n_persons, n_categories = probabilities.shape
    cumulative = np.empty((n_persons, n_categories + 1), dtype=np.float64)
    cumulative[:, 0] = 1.0
    cumulative[:, -1] = 0.0
    np.cumsum(
        probabilities[:, :0:-1],
        axis=1,
        out=cumulative[:, -2:0:-1],
    )

    np.multiply(cumulative, 1.0 - cumulative, out=cumulative)
    derivatives = cumulative[:, :-1] - cumulative[:, 1:]
    derivatives *= discrimination
    np.square(derivatives, out=derivatives)
    valid = probabilities > PROB_EPSILON
    np.divide(
        derivatives,
        probabilities,
        out=derivatives,
        where=valid,
    )
    derivatives[~valid] = 0.0
    return derivatives.sum(axis=1)


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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        threshold_mask = np.zeros_like(self.thresholds, dtype=np.bool_)
        for item_idx, n_categories in enumerate(self._n_categories):
            threshold_mask[item_idx, : n_categories - 1] = True
        masks["thresholds"] = threshold_mask
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "thresholds":
            for item_idx, n_categories in enumerate(self._n_categories):
                canonical[item_idx, n_categories - 1 :] = 0.0
        return canonical

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

        return sigmoid(z)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GRM category probabilities in one vectorized pass."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]
        a_item = self._parameters["discrimination"][item_idx]
        thresholds = self._parameters["thresholds"][item_idx, : n_cat - 1]

        if self.n_factors == 1:
            logits = a_item * (theta.ravel()[:, None] - thresholds[None, :])
        else:
            logits = np.dot(theta, a_item)[:, None] - np.sum(a_item) * thresholds

        cumulative = sigmoid(logits)
        probabilities = np.empty((n_persons, n_cat), dtype=np.float64)
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        probabilities[:, 1:-1] = cumulative[:, :-1] - cumulative[:, 1:]
        probabilities[:, -1] = cumulative[:, -1]
        return probabilities

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = float(a[item_idx])
        else:
            a_val = float(np.linalg.norm(a[item_idx]))

        probabilities = self._category_probabilities(theta, item_idx)
        return _graded_information(probabilities, a_val)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if RUST_AVAILABLE and self.n_factors == 1:
            responses = self._validate_polytomous_responses(responses)
            theta = self._ensure_theta_2d(theta)
            quad_points = theta.ravel() if theta.ndim == 2 else theta
            disc = self._parameters["discrimination"]
            thresh = self._parameters["thresholds"]
            n_cats = np.array(self._n_categories, dtype=np.int32)
            return compute_log_likelihoods_grm(
                responses, quad_points, disc, thresh, n_cats
            )
        return super().log_likelihood_batch(responses, theta)


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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        step_mask = np.zeros_like(self.steps, dtype=np.bool_)
        for item_idx, n_categories in enumerate(self._n_categories):
            step_mask[item_idx, : n_categories - 1] = True
        masks["steps"] = step_mask
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "steps":
            for item_idx, n_categories in enumerate(self._n_categories):
                canonical[item_idx, n_categories - 1 :] = 0.0
        return canonical

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GPCM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        steps = self._parameters["steps"][item_idx, : n_cat - 1]

        if self.n_factors == 1:
            a_item = a[item_idx]
            increments = a_item * (theta.ravel()[:, None] - steps[None, :])
        else:
            a_item = a[item_idx]
            scale = np.sqrt(np.sum(a_item**2))
            projected_theta = np.dot(theta, a_item)
            increments = scale * (projected_theta[:, None] - steps[None, :])

        logits = np.empty((n_persons, n_cat), dtype=np.float64)
        logits[:, 0] = 0.0
        np.cumsum(increments, axis=1, out=logits[:, 1:])
        return _stable_softmax(logits)

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

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if RUST_AVAILABLE and self.n_factors == 1:
            responses = self._validate_polytomous_responses(responses)
            theta = self._ensure_theta_2d(theta)
            quad_points = theta.ravel() if theta.ndim == 2 else theta
            disc = self._parameters["discrimination"]
            steps_full = np.zeros((self.n_items, max(self._n_categories)))
            for i, n_cat in enumerate(self._n_categories):
                steps_full[i, 1:n_cat] = self._parameters["steps"][i, : n_cat - 1]
            n_cats = np.array(self._n_categories, dtype=np.int32)
            return compute_log_likelihoods_gpcm(
                responses, quad_points, disc, steps_full, n_cats
            )
        return super().log_likelihood_batch(responses, theta)


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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["discrimination"] = np.zeros_like(self.discrimination, dtype=np.bool_)
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "discrimination":
            canonical.fill(1.0)
        return canonical

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

        if isinstance(n_categories, list):
            if len(set(n_categories)) != 1:
                raise ValueError(
                    "RSM requires all items to have the same number of categories"
                )
            n_categories = n_categories[0]

        self._n_cats = n_categories
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["difficulty"] = np.zeros(self.n_items)

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
        n_cat = self._n_cats

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all RSM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_cats

        b_j = self._parameters["difficulty"][item_idx]
        tau = self._parameters["thresholds"]
        increments = theta.ravel()[:, None] - b_j - tau[None, :]

        logits = np.empty((n_persons, n_cat), dtype=np.float64)
        logits[:, 0] = 0.0
        np.cumsum(increments, axis=1, out=logits[:, 1:])
        return _stable_softmax(logits)

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


class GradedRatingScaleModel(PolytomousItemModel):
    """Graded Rating Scale Model (GRSM) for polytomous items.

    The GRSM is a constrained GRM where discrimination parameters are
    equal across all items. This is the graded response analog of the
    Rating Scale Model.

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
    discrimination : float
        Common discrimination parameter for all items
    difficulty : ndarray of shape (n_items,)
        Item location parameters
    thresholds : ndarray of shape (n_categories - 1,)
        Category threshold parameters (relative to item location)

    Notes
    -----
    The GRSM cumulative probability is:

        P(X >= k | theta) = 1 / (1 + exp(-a * (theta - b_j - tau_k)))

    where a is the common discrimination, b_j is item location, and
    tau_k are shared category thresholds.

    References
    ----------
    Muraki, E. (1990). Fitting a polytomous item response model to
        Likert-type data. Applied Psychological Measurement, 14, 59-71.
    """

    model_name = "GRSM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("GRSM only supports unidimensional analysis")

        if isinstance(n_categories, list):
            if len(set(n_categories)) != 1:
                raise ValueError(
                    "GRSM requires all items to have the same number of categories"
                )
            n_categories = n_categories[0]

        self._n_cats = n_categories
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.array([1.0])
        self._parameters["difficulty"] = np.zeros(self.n_items)
        n_thresholds = self._n_cats - 1
        self._parameters["thresholds"] = np.linspace(-2, 2, n_thresholds)

    @property
    def discrimination(self) -> float:
        """Common discrimination parameter."""
        return float(self._parameters["discrimination"][0])

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item location parameters."""
        return self._parameters["difficulty"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Shared category threshold parameters."""
        return self._parameters["thresholds"]

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"][0]
        b_j = self._parameters["difficulty"][item_idx]
        tau_k = self._parameters["thresholds"][threshold_idx]

        z = a * (theta_1d - b_j - tau_k)
        return sigmoid(z)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_cats

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GRSM category probabilities in one vectorized pass."""
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]
        n_cat = self._n_cats
        a = self._parameters["discrimination"][0]
        b_j = self._parameters["difficulty"][item_idx]
        thresholds = self._parameters["thresholds"]

        logits = a * (theta.ravel()[:, None] - b_j - thresholds[None, :])
        cumulative = sigmoid(logits)
        probabilities = np.empty((n_persons, n_cat), dtype=np.float64)
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        probabilities[:, 1:-1] = cumulative[:, :-1] - cumulative[:, 1:]
        probabilities[:, -1] = cumulative[:, -1]
        return probabilities

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        discrimination = float(self._parameters["discrimination"][0])
        probabilities = self._category_probabilities(theta, item_idx)
        return _graded_information(probabilities, discrimination)

    def set_parameters(
        self,
        discrimination: float | None = None,
        difficulty: NDArray[np.float64] | None = None,
        thresholds: NDArray[np.float64] | None = None,
    ) -> "GradedRatingScaleModel":
        if discrimination is not None:
            self._parameters["discrimination"] = np.array([float(discrimination)])
        if difficulty is not None:
            difficulty = np.asarray(difficulty)
            if difficulty.shape != (self.n_items,):
                raise ValueError(f"difficulty must have shape ({self.n_items},)")
            self._parameters["difficulty"] = difficulty
        if thresholds is not None:
            thresholds = np.asarray(thresholds)
            if thresholds.shape != (self._n_cats - 1,):
                raise ValueError(f"thresholds must have shape ({self._n_cats - 1},)")
            self._parameters["thresholds"] = thresholds

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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        slope_mask = np.zeros_like(self.slopes, dtype=np.bool_)
        intercept_mask = np.zeros_like(self.intercepts, dtype=np.bool_)
        for item_idx, n_categories in enumerate(self._n_categories):
            slope_mask[item_idx, 1:n_categories, ...] = True
            intercept_mask[item_idx, 1:n_categories] = True
        masks["slopes"] = slope_mask
        masks["intercepts"] = intercept_mask
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name not in {"slopes", "intercepts"}:
            return canonical

        for item_idx, n_categories in enumerate(self._n_categories):
            reference = canonical[item_idx, 0].copy()
            canonical[item_idx, :n_categories] -= reference
            canonical[item_idx, n_categories:] = 0.0
        return canonical

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all NRM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        c = self._parameters["intercepts"]

        if self.n_factors == 1:
            logits = (
                theta.ravel()[:, None] * a[item_idx, None, :n_cat]
                + c[item_idx, None, :n_cat]
            )
        else:
            logits = np.dot(theta, a[item_idx, :n_cat].T) + c[item_idx, None, :n_cat]

        return _stable_softmax(logits)

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
