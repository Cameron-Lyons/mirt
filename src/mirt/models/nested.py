"""Nested Logit Models for multiple-choice items.

These models handle multiple-choice items where distractor responses
provide information about ability. The nested structure models the
probability of choosing each distractor conditional on not selecting
the correct response.

References:
    Suh, Y., & Bolt, D. M. (2010). Nested logit models for multiple-choice
        item response data. Psychometrika, 75(3), 454-473.
    Thissen, D., & Steinberg, L. (1984). A response model for multiple choice
        items. Psychometrika, 49(4), 501-519.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import PolytomousItemModel


class TwoPLNestedLogit(PolytomousItemModel):
    """Two-Parameter Logistic Nested Logit Model (2PL-NRM).

    For multiple-choice items, this model separates the probability of
    selecting the correct answer from the probability of choosing among
    distractors given an incorrect response.

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int or list of int
        Number of response options (including correct answer)
    correct_response : int or list of int
        Index of correct response for each item (0-indexed)
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Discrimination parameters for correct vs incorrect
    difficulty : ndarray
        Difficulty parameters (location of correct response)
    distractor_slopes : ndarray
        Slopes for distractor choice (conditional on incorrect)
    distractor_intercepts : ndarray
        Intercepts for distractor choice

    Notes
    -----
    P(correct) = 1 / (1 + exp(-a(θ - b)))
    P(distractor k | incorrect) = exp(ak*θ + ck) / sum_j exp(aj*θ + cj)
    """

    model_name = "2PLNRM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        correct_response: int | list[int] = 0,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("2PLNRM only supports unidimensional analysis")

        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

        if isinstance(correct_response, int):
            self._correct = [correct_response] * n_items
        else:
            self._correct = list(correct_response)

    def _initialize_parameters(self) -> None:
        # Parameters for correct/incorrect
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

        # Parameters for distractor choice
        max_cats = max(self._n_categories)
        self._parameters["distractor_slopes"] = np.zeros((self.n_items, max_cats))
        self._parameters["distractor_intercepts"] = np.zeros((self.n_items, max_cats))

        for i, n_cat in enumerate(self._n_categories):
            n_dist = n_cat - 1  # Number of distractors
            if n_dist > 0:
                self._parameters["distractor_intercepts"][i, :n_cat] = np.linspace(
                    -0.5, 0.5, n_cat
                )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_cat = self._n_categories[item_idx]
        correct_idx = self._correct[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["difficulty"][item_idx]

        # Probability of correct response
        z = a * (theta_1d - b)
        p_correct = 1.0 / (1.0 + np.exp(-z))

        if category == correct_idx:
            return p_correct

        # Probability of this distractor given incorrect
        d_slopes = self._parameters["distractor_slopes"][item_idx, :n_cat]
        d_intercepts = self._parameters["distractor_intercepts"][item_idx, :n_cat]

        # Compute distractor logits (excluding correct response)
        distractor_logits = d_slopes * theta_1d[:, None] + d_intercepts

        # Set correct response logit to -inf (won't be chosen in distractor model)
        distractor_logits[:, correct_idx] = -np.inf

        # Softmax for distractors
        max_logits = np.max(distractor_logits, axis=1, keepdims=True)
        exp_logits = np.exp(distractor_logits - max_logits)
        p_distractor_given_incorrect = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        return (1.0 - p_correct) * p_distractor_given_incorrect[:, category]

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]
        probs = self.probability(theta, item_idx)

        # Score for correct response
        correct_idx = self._correct[item_idx]
        scores = np.zeros(n_cat)
        scores[correct_idx] = 1

        expected = np.sum(probs * scores, axis=1)
        expected_sq = np.sum(probs * (scores**2), axis=1)
        variance = expected_sq - expected**2

        a = self._parameters["discrimination"][item_idx]
        return (a**2) * variance


class ThreePLNestedLogit(TwoPLNestedLogit):
    """Three-Parameter Logistic Nested Logit Model (3PL-NRM).

    Extends 2PL-NRM with a guessing parameter.
    """

    model_name = "3PLNRM"

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self._parameters["guessing"] = np.full(self.n_items, 0.2)

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_cat = self._n_categories[item_idx]
        correct_idx = self._correct[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["difficulty"][item_idx]
        c = self._parameters["guessing"][item_idx]

        # 3PL probability of correct
        z = a * (theta_1d - b)
        p_star = 1.0 / (1.0 + np.exp(-z))
        p_correct = c + (1.0 - c) * p_star

        if category == correct_idx:
            return p_correct

        # Distractor probabilities
        d_slopes = self._parameters["distractor_slopes"][item_idx, :n_cat]
        d_intercepts = self._parameters["distractor_intercepts"][item_idx, :n_cat]

        distractor_logits = d_slopes * theta_1d[:, None] + d_intercepts
        distractor_logits[:, correct_idx] = -np.inf

        max_logits = np.max(distractor_logits, axis=1, keepdims=True)
        exp_logits = np.exp(distractor_logits - max_logits)
        p_distractor_given_incorrect = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        return (1.0 - p_correct) * p_distractor_given_incorrect[:, category]


class FourPLNestedLogit(ThreePLNestedLogit):
    """Four-Parameter Logistic Nested Logit Model (4PL-NRM).

    Extends 3PL-NRM with an upper asymptote parameter.
    """

    model_name = "4PLNRM"

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self._parameters["upper"] = np.ones(self.n_items)

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_cat = self._n_categories[item_idx]
        correct_idx = self._correct[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        a = self._parameters["discrimination"][item_idx]
        b = self._parameters["difficulty"][item_idx]
        c = self._parameters["guessing"][item_idx]
        d = self._parameters["upper"][item_idx]

        # 4PL probability of correct
        z = a * (theta_1d - b)
        p_star = 1.0 / (1.0 + np.exp(-z))
        p_correct = c + (d - c) * p_star

        if category == correct_idx:
            return p_correct

        # Distractor probabilities
        d_slopes = self._parameters["distractor_slopes"][item_idx, :n_cat]
        d_intercepts = self._parameters["distractor_intercepts"][item_idx, :n_cat]

        distractor_logits = d_slopes * theta_1d[:, None] + d_intercepts
        distractor_logits[:, correct_idx] = -np.inf

        max_logits = np.max(distractor_logits, axis=1, keepdims=True)
        exp_logits = np.exp(distractor_logits - max_logits)
        p_distractor_given_incorrect = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        return (1.0 - p_correct) * p_distractor_given_incorrect[:, category]
