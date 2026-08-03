"""Multilevel IRT models with hierarchical and crossed random effects.

This module provides:
- Two-level models (persons nested in groups)
- Three-level models (persons in level-2 units in level-3 units)
- Crossed random effects (person x item x rater)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _validate_identifiers(values: object, name: str) -> NDArray[np.intp]:
    """Return a copied vector of non-negative integer identifiers."""
    values_array = np.asarray(values)
    if values_array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if values_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if values_array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer identifiers")
    if values_array.dtype.kind == "u" and np.any(values_array > np.iinfo(np.intp).max):
        raise ValueError(f"{name} contains identifiers outside the supported range")
    if np.any(values_array < 0):
        raise ValueError(f"{name} must contain non-negative identifiers")
    return values_array.astype(np.intp, copy=True)


def _encode_identifiers(
    values: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Map arbitrary external identifiers to compact, sorted positions."""
    identifiers, positions = np.unique(values, return_inverse=True)
    return (
        identifiers.astype(np.intp, copy=False),
        positions.astype(np.intp, copy=False),
    )


def _validate_labels(
    labels: list[str] | None,
    identifiers: NDArray[np.intp],
    prefix: str,
) -> list[str]:
    if labels is None:
        return [f"{prefix}_{identifier}" for identifier in identifiers]
    if len(labels) != len(identifiers):
        raise ValueError(
            f"labels length ({len(labels)}) must match number of units "
            f"({len(identifiers)})"
        )
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must contain non-empty strings")
    return list(labels)


def _finite_vector(
    values: object,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} != {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _finite_variance(value: float, name: str, *, positive: bool) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} variance must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} variance must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} variance must be non-negative")
    return result


def _validate_index(index: int, size: int, name: str) -> int:
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(index)
    if result < 0 or result >= size:
        raise IndexError(f"{name} {result} out of range [0, {size})")
    return result


def _expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the logistic function without overflow at either tail."""
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _shift_log_odds(
    probabilities: NDArray[np.float64],
    effects: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        PROB_EPSILON,
        1.0 - PROB_EPSILON,
    )
    log_odds = np.log(probabilities) - np.log1p(-probabilities)
    return _expit(log_odds + effects)


@dataclass
class RandomEffectSpec:
    """Specification for a nested or crossed random effect."""

    name: str
    type: Literal["nested", "crossed"]
    n_levels: int
    variance_prior: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.type not in ("nested", "crossed"):
            raise ValueError("type must be 'nested' or 'crossed'")
        if (
            isinstance(self.n_levels, (bool, np.bool_))
            or not isinstance(self.n_levels, (int, np.integer))
            or self.n_levels <= 0
        ):
            raise ValueError("n_levels must be a positive integer")
        if len(self.variance_prior) != 2:
            raise ValueError("variance_prior must contain shape and scale")
        shape, scale = (float(value) for value in self.variance_prior)
        if not np.isfinite(shape) or shape <= 0:
            raise ValueError("variance_prior shape must be finite and positive")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("variance_prior scale must be finite and positive")
        self.n_levels = int(self.n_levels)
        self.variance_prior = (shape, scale)


@dataclass
class MultilevelIRTResult:
    """Results from multilevel IRT estimation."""

    model: MultilevelIRTModel
    fixed_effects: dict[str, NDArray[np.float64]]
    random_effects: dict[str, NDArray[np.float64]]
    variance_components: dict[str, float]
    icc: dict[str, float]
    log_likelihood: float
    dic: float
    n_iterations: int
    converged: bool


class MultilevelIRTModel:
    """Two-level IRT model with persons nested in groups.

    Group identifiers may be sparse or non-contiguous. Parameters and labels
    follow the ascending order returned by :attr:`group_ids`.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        group_membership: NDArray[np.int_],
        group_labels: list[str] | None = None,
    ) -> None:
        membership = _validate_identifiers(group_membership, "group_membership")
        group_ids, group_positions = _encode_identifiers(membership)

        self._base_model = base_model.copy()
        self._group_membership = membership
        self._group_ids = group_ids
        self._group_positions = group_positions
        self._n_persons = membership.size
        self._n_groups = group_ids.size
        self._group_labels = _validate_labels(group_labels, group_ids, "Group")

        self._group_means = np.zeros(self._n_groups)
        self._between_variance = 0.25
        self._within_variance = 1.0
        self._is_fitted = False

    @property
    def base_model(self) -> BaseItemModel:
        return self._base_model

    @property
    def n_persons(self) -> int:
        return self._n_persons

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @property
    def n_items(self) -> int:
        return self._base_model.n_items

    @property
    def group_membership(self) -> NDArray[np.intp]:
        return self._group_membership.copy()

    @property
    def group_ids(self) -> NDArray[np.intp]:
        """External group identifiers in parameter order."""
        return self._group_ids.copy()

    @property
    def group_labels(self) -> list[str]:
        return self._group_labels.copy()

    @property
    def group_means(self) -> NDArray[np.float64]:
        return self._group_means.copy()

    @property
    def between_variance(self) -> float:
        return self._between_variance

    @property
    def within_variance(self) -> float:
        return self._within_variance

    @property
    def icc(self) -> float:
        """Intraclass correlation coefficient."""
        total = self._between_variance + self._within_variance
        return self._between_variance / total

    def set_group_means(self, means: NDArray[np.float64]) -> Self:
        self._group_means = _finite_vector(means, (self._n_groups,), "means")
        return self

    def set_variance_components(self, between: float, within: float = 1.0) -> Self:
        new_between = _finite_variance(between, "between", positive=False)
        new_within = _finite_variance(within, "within", positive=True)
        self._between_variance = new_between
        self._within_variance = new_within
        return self

    def person_prior_mean(self) -> NDArray[np.float64]:
        """Get the prior mean for each person based on group membership."""
        return self._group_means[self._group_positions]

    def person_prior_variance(self) -> float:
        """Get the prior variance for person abilities within groups."""
        return self._within_variance

    def group_prior_variance(self) -> float:
        """Get the prior variance for group means."""
        return self._between_variance

    def group_sizes(self) -> NDArray[np.intp]:
        """Get group sizes in :attr:`group_ids` order."""
        return np.bincount(self._group_positions, minlength=self._n_groups).astype(
            np.intp, copy=False
        )

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute the response log-likelihood given person abilities."""
        return float(np.sum(self._base_model.log_likelihood(responses, theta)))

    def copy(self) -> Self:
        new_model = MultilevelIRTModel(
            base_model=self._base_model,
            group_membership=self._group_membership,
            group_labels=self._group_labels,
        )
        new_model._group_means = self._group_means.copy()
        new_model._between_variance = self._between_variance
        new_model._within_variance = self._within_variance
        new_model._is_fitted = self._is_fitted
        return new_model


class ThreeLevelIRTModel:
    """Three-level IRT model for persons nested in two higher levels.

    ``level3_membership`` supplies one external level-3 identifier for each
    level-2 unit, ordered by ascending :attr:`level2_ids`.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        level2_membership: NDArray[np.int_],
        level3_membership: NDArray[np.int_],
        level2_labels: list[str] | None = None,
        level3_labels: list[str] | None = None,
    ) -> None:
        membership2 = _validate_identifiers(level2_membership, "level2_membership")
        membership3 = _validate_identifiers(level3_membership, "level3_membership")
        level2_ids, level2_positions = _encode_identifiers(membership2)
        if membership3.size != level2_ids.size:
            raise ValueError(
                f"level3_membership length ({membership3.size}) must match "
                f"number of level-2 units ({level2_ids.size})"
            )
        level3_ids, level3_positions = _encode_identifiers(membership3)

        self._base_model = base_model.copy()
        self._level2_membership = membership2
        self._level3_membership = membership3
        self._level2_ids = level2_ids
        self._level3_ids = level3_ids
        self._level2_positions = level2_positions
        self._level3_positions = level3_positions

        self._n_persons = membership2.size
        self._n_level2 = level2_ids.size
        self._n_level3 = level3_ids.size
        self._level2_labels = _validate_labels(level2_labels, level2_ids, "L2")
        self._level3_labels = _validate_labels(level3_labels, level3_ids, "L3")

        self._level2_effects = np.zeros(self._n_level2)
        self._level3_effects = np.zeros(self._n_level3)
        self._level2_variance = 0.15
        self._level3_variance = 0.10
        self._within_variance = 1.0
        self._is_fitted = False

    @property
    def base_model(self) -> BaseItemModel:
        return self._base_model

    @property
    def n_persons(self) -> int:
        return self._n_persons

    @property
    def n_items(self) -> int:
        return self._base_model.n_items

    @property
    def n_level2_units(self) -> int:
        return self._n_level2

    @property
    def n_level3_units(self) -> int:
        return self._n_level3

    @property
    def level2_membership(self) -> NDArray[np.intp]:
        return self._level2_membership.copy()

    @property
    def level3_membership(self) -> NDArray[np.intp]:
        return self._level3_membership.copy()

    @property
    def level2_ids(self) -> NDArray[np.intp]:
        return self._level2_ids.copy()

    @property
    def level3_ids(self) -> NDArray[np.intp]:
        return self._level3_ids.copy()

    @property
    def level2_labels(self) -> list[str]:
        return self._level2_labels.copy()

    @property
    def level3_labels(self) -> list[str]:
        return self._level3_labels.copy()

    @property
    def level2_effects(self) -> NDArray[np.float64]:
        return self._level2_effects.copy()

    @property
    def level3_effects(self) -> NDArray[np.float64]:
        return self._level3_effects.copy()

    @property
    def variance_components(self) -> dict[str, float]:
        return {
            "within": self._within_variance,
            "level2": self._level2_variance,
            "level3": self._level3_variance,
        }

    def set_level_effects(
        self,
        level2: NDArray[np.float64] | None = None,
        level3: NDArray[np.float64] | None = None,
    ) -> Self:
        """Set either or both vectors of level effects atomically."""
        new_level2 = (
            self._level2_effects.copy()
            if level2 is None
            else _finite_vector(level2, (self._n_level2,), "level2 effects")
        )
        new_level3 = (
            self._level3_effects.copy()
            if level3 is None
            else _finite_vector(level3, (self._n_level3,), "level3 effects")
        )
        self._level2_effects = new_level2
        self._level3_effects = new_level3
        return self

    def set_variance_components(
        self,
        level2: float,
        level3: float,
        within: float = 1.0,
    ) -> Self:
        """Set all variance components atomically."""
        new_level2 = _finite_variance(level2, "level2", positive=False)
        new_level3 = _finite_variance(level3, "level3", positive=False)
        new_within = _finite_variance(within, "within", positive=True)
        self._level2_variance = new_level2
        self._level3_variance = new_level3
        self._within_variance = new_within
        return self

    def icc(self, level: Literal["level2", "level3", "total"] = "total") -> float:
        """Compute the variance share at a requested hierarchy level."""
        if level not in ("level2", "level3", "total"):
            raise ValueError("level must be 'level2', 'level3', or 'total'")
        total = self._within_variance + self._level2_variance + self._level3_variance
        if level == "level2":
            return self._level2_variance / total
        if level == "level3":
            return self._level3_variance / total
        return (self._level2_variance + self._level3_variance) / total

    def person_prior_mean(self) -> NDArray[np.float64]:
        """Get the combined level-2 and level-3 prior mean per person."""
        level2_effect = self._level2_effects[self._level2_positions]
        person_level3_positions = self._level3_positions[self._level2_positions]
        level3_effect = self._level3_effects[person_level3_positions]
        return level2_effect + level3_effect

    def level2_sizes(self) -> NDArray[np.intp]:
        """Get person counts in :attr:`level2_ids` order."""
        return np.bincount(self._level2_positions, minlength=self._n_level2).astype(
            np.intp, copy=False
        )

    def level3_sizes(self) -> NDArray[np.intp]:
        """Get level-2 unit counts in :attr:`level3_ids` order."""
        return np.bincount(self._level3_positions, minlength=self._n_level3).astype(
            np.intp, copy=False
        )

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute the response log-likelihood given person abilities."""
        return float(np.sum(self._base_model.log_likelihood(responses, theta)))

    def copy(self) -> Self:
        new_model = ThreeLevelIRTModel(
            base_model=self._base_model,
            level2_membership=self._level2_membership,
            level3_membership=self._level3_membership,
            level2_labels=self._level2_labels,
            level3_labels=self._level3_labels,
        )
        new_model._level2_effects = self._level2_effects.copy()
        new_model._level3_effects = self._level3_effects.copy()
        new_model._level2_variance = self._level2_variance
        new_model._level3_variance = self._level3_variance
        new_model._within_variance = self._within_variance
        new_model._is_fitted = self._is_fitted
        return new_model


class CrossedRandomEffectsModel:
    """IRT model with crossed person, item, and rater effects.

    Rater assignments are stored as a person-by-item matrix. A flattened
    assignment vector is accepted and reshaped using the base model's item
    count. Rater effects shift the base model probability on the log-odds
    scale.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        n_raters: int,
        rater_assignments: NDArray[np.int_] | None = None,
        include_item_effects: bool = True,
        include_rater_effects: bool = True,
    ) -> None:
        if (
            isinstance(n_raters, (bool, np.bool_))
            or not isinstance(n_raters, (int, np.integer))
            or n_raters <= 0
        ):
            raise ValueError("n_raters must be a positive integer")
        if not isinstance(include_item_effects, (bool, np.bool_)):
            raise TypeError("include_item_effects must be boolean")
        if not isinstance(include_rater_effects, (bool, np.bool_)):
            raise TypeError("include_rater_effects must be boolean")
        if base_model.is_polytomous:
            raise ValueError(
                "CrossedRandomEffectsModel requires a dichotomous base_model"
            )

        self._base_model = base_model.copy()
        self._n_raters = int(n_raters)
        self._rater_assignments = self._prepare_assignments(rater_assignments)
        self._include_item_effects = bool(include_item_effects)
        self._include_rater_effects = bool(include_rater_effects)

        self._rater_effects = np.zeros(self._n_raters)
        self._person_variance = 1.0
        self._item_variance = 0.5
        self._rater_variance = 0.25
        self._is_fitted = False

    def _prepare_assignments(
        self, assignments: NDArray[np.int_] | None
    ) -> NDArray[np.intp] | None:
        if assignments is None:
            return None
        result = np.asarray(assignments)
        if result.ndim == 1:
            if result.size == 0 or result.size % self._base_model.n_items != 0:
                raise ValueError(
                    "flattened rater_assignments length must be a positive "
                    "multiple of n_items"
                )
            result = result.reshape(-1, self._base_model.n_items)
        elif result.ndim == 2:
            if result.shape[0] == 0 or result.shape[1] != self._base_model.n_items:
                raise ValueError(
                    "rater_assignments must have shape (n_persons, n_items)"
                )
        else:
            raise ValueError("rater_assignments must be 1D or 2D")

        flattened = _validate_identifiers(result.reshape(-1), "rater_assignments")
        if np.any(flattened >= self._n_raters):
            raise ValueError(f"rater_assignments must be in [0, {self._n_raters})")
        return flattened.reshape(result.shape)

    @property
    def base_model(self) -> BaseItemModel:
        return self._base_model

    @property
    def n_raters(self) -> int:
        return self._n_raters

    @property
    def n_persons(self) -> int | None:
        if self._rater_assignments is None:
            return None
        return self._rater_assignments.shape[0]

    @property
    def rater_assignments(self) -> NDArray[np.intp] | None:
        if self._rater_assignments is None:
            return None
        return self._rater_assignments.copy()

    @property
    def rater_effects(self) -> NDArray[np.float64]:
        return self._rater_effects.copy()

    @property
    def variance_components(self) -> dict[str, float]:
        components = {"person": self._person_variance}
        if self._include_item_effects:
            components["item"] = self._item_variance
        if self._include_rater_effects:
            components["rater"] = self._rater_variance
        return components

    @property
    def variance_partition(self) -> dict[str, float]:
        """Return each enabled variance component as a share of the total."""
        components = self.variance_components
        total = sum(components.values())
        return {name: value / total for name, value in components.items()}

    def set_rater_effects(self, effects: NDArray[np.float64]) -> Self:
        self._rater_effects = _finite_vector(effects, (self._n_raters,), "effects")
        return self

    def set_variance_components(
        self,
        person: float = 1.0,
        item: float | None = None,
        rater: float | None = None,
    ) -> Self:
        """Set supplied variance components atomically."""
        new_person = _finite_variance(person, "person", positive=True)
        new_item = (
            self._item_variance
            if item is None
            else _finite_variance(item, "item", positive=False)
        )
        new_rater = (
            self._rater_variance
            if rater is None
            else _finite_variance(rater, "rater", positive=False)
        )
        self._person_variance = new_person
        self._item_variance = new_item
        self._rater_variance = new_rater
        return self

    def get_rater_for_observation(self, person_idx: int, item_idx: int) -> int | None:
        """Get the rater identifier for a person-item observation."""
        if self._rater_assignments is None:
            return None
        person_idx = _validate_index(
            person_idx, self._rater_assignments.shape[0], "person_idx"
        )
        item_idx = _validate_index(item_idx, self._base_model.n_items, "item_idx")
        return int(self._rater_assignments[person_idx, item_idx])

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
        rater_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute probabilities with an optional common rater effect."""
        base_probability = np.asarray(
            self._base_model.probability(theta, item_idx), dtype=np.float64
        )
        if rater_idx is None or not self._include_rater_effects:
            return base_probability
        rater_idx = _validate_index(rater_idx, self._n_raters, "rater_idx")
        return _shift_log_odds(base_probability, self._rater_effects[rater_idx])

    def assigned_probability(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute all person-item probabilities using stored assignments."""
        if self._rater_assignments is None:
            raise ValueError("rater_assignments are required")
        base_probability = np.asarray(
            self._base_model.probability(theta), dtype=np.float64
        )
        if base_probability.shape != self._rater_assignments.shape:
            raise ValueError(
                f"theta produces probability shape {base_probability.shape}; "
                f"expected {self._rater_assignments.shape}"
            )
        if not self._include_rater_effects:
            return base_probability
        effects = self._rater_effects[self._rater_assignments]
        return _shift_log_odds(base_probability, effects)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute log-likelihood, applying stored rater assignments if present."""
        if self._rater_assignments is None:
            return float(np.sum(self._base_model.log_likelihood(responses, theta)))

        responses_array = np.asarray(responses)
        if responses_array.shape != self._rater_assignments.shape:
            raise ValueError(
                f"responses shape {responses_array.shape} != "
                f"{self._rater_assignments.shape}"
            )
        if responses_array.dtype.kind not in "biuf":
            raise ValueError("responses must contain numeric response codes")
        if responses_array.dtype.kind == "f" and (
            not np.all(np.isfinite(responses_array))
            or not np.all(responses_array == np.trunc(responses_array))
        ):
            raise ValueError("responses must contain finite integer response codes")
        valid = responses_array >= 0
        if np.any(valid & (responses_array != 0) & (responses_array != 1)):
            raise ValueError(
                "responses must contain only 0, 1, or negative missing codes"
            )

        probabilities = np.clip(
            self.assigned_probability(theta),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        log_likelihood = np.where(
            valid,
            responses_array * np.log(probabilities)
            + (1 - responses_array) * np.log1p(-probabilities),
            0.0,
        )
        return float(np.sum(log_likelihood))

    def copy(self) -> Self:
        new_model = CrossedRandomEffectsModel(
            base_model=self._base_model,
            n_raters=self._n_raters,
            rater_assignments=self._rater_assignments,
            include_item_effects=self._include_item_effects,
            include_rater_effects=self._include_rater_effects,
        )
        new_model._rater_effects = self._rater_effects.copy()
        new_model._person_variance = self._person_variance
        new_model._item_variance = self._item_variance
        new_model._rater_variance = self._rater_variance
        new_model._is_fitted = self._is_fitted
        return new_model


@dataclass
class NestedHierarchy:
    """Describe ordered, nested membership transitions.

    Each membership vector is ordered by the ascending external identifiers
    from the preceding transition. The first vector is ordered by lowest-level
    unit position.
    """

    levels: list[str]
    memberships: list[NDArray[np.int_]]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError("levels must contain at least two hierarchy levels")
        if any(not isinstance(level, str) or not level for level in self.levels):
            raise ValueError("levels must contain non-empty strings")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be unique")
        if len(self.memberships) != len(self.levels) - 1:
            raise ValueError(
                f"Need {len(self.levels) - 1} membership arrays for "
                f"{len(self.levels)} levels"
            )

        validated: list[NDArray[np.intp]] = []
        for index, membership in enumerate(self.memberships):
            result = _validate_identifiers(membership, f"memberships[{index}]")
            if index > 0:
                expected = np.unique(validated[index - 1]).size
                if result.size != expected:
                    raise ValueError(
                        f"memberships[{index}] length ({result.size}) must match "
                        f"the preceding level's unit count ({expected})"
                    )
            validated.append(result)
        self.levels = list(self.levels)
        self.memberships = validated

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def get_unit_counts(self) -> dict[str, int]:
        """Get the unit count at every level."""
        counts = {self.levels[0]: len(self.memberships[0])}
        for index, membership in enumerate(self.memberships):
            counts[self.levels[index + 1]] = int(np.unique(membership).size)
        return counts

    def get_full_path(self, unit_idx: int, level: int = 0) -> list[int]:
        """Get external identifiers from one unit position to the top level."""
        level = _validate_index(level, self.n_levels, "level")
        unit_count = self.get_unit_counts()[self.levels[level]]
        position = _validate_index(unit_idx, unit_count, "unit_idx")
        path = [position]
        for index in range(level, len(self.memberships)):
            membership = self.memberships[index]
            identifier = int(membership[position])
            path.append(identifier)
            if index + 1 < len(self.memberships):
                preceding_ids = np.unique(membership)
                position = int(np.searchsorted(preceding_ids, identifier))
        return path
