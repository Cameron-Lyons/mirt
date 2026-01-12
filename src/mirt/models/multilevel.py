"""Multilevel IRT models with hierarchical and crossed random effects.

This module provides:
- Two-level models (persons nested in groups)
- Three-level models (persons in classrooms in schools)
- Crossed random effects (person × item × rater)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class RandomEffectSpec:
    """Specification for a random effect.

    Attributes
    ----------
    name : str
        Name of the random effect (e.g., 'school', 'rater').
    type : str
        Type of random effect: 'nested' or 'crossed'.
    n_levels : int
        Number of levels (e.g., number of schools).
    variance_prior : tuple
        Prior for variance component (shape, scale) for inverse-gamma.
    """

    name: str
    type: Literal["nested", "crossed"]
    n_levels: int
    variance_prior: tuple[float, float] = (1.0, 1.0)


@dataclass
class MultilevelIRTResult:
    """Results from multilevel IRT estimation.

    Attributes
    ----------
    model : MultilevelIRTModel
        Fitted model.
    fixed_effects : dict
        Estimated fixed effects (item parameters).
    random_effects : dict
        Estimated random effects (group means, etc.).
    variance_components : dict
        Estimated variance components.
    icc : dict
        Intraclass correlation coefficients.
    log_likelihood : float
        Log-likelihood at convergence.
    dic : float
        Deviance Information Criterion.
    n_iterations : int
        Number of iterations.
    converged : bool
        Whether estimation converged.
    """

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

    Supports hierarchical IRT where ability varies both within and
    between higher-level units (e.g., schools, classrooms).

    Parameters
    ----------
    base_model : BaseItemModel
        IRT model for item responses.
    group_membership : ndarray of shape (n_persons,)
        Group assignment for each person.
    group_labels : list of str, optional
        Labels for groups.

    Attributes
    ----------
    group_means : ndarray of shape (n_groups,)
        Mean ability in each group.
    between_variance : float
        Between-group variance in ability.
    within_variance : float
        Within-group variance in ability (typically fixed to 1).
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        group_membership: NDArray[np.int_],
        group_labels: list[str] | None = None,
    ) -> None:
        group_membership = np.asarray(group_membership)
        if group_membership.ndim != 1:
            raise ValueError("group_membership must be 1D")

        self._base_model = base_model.copy()
        self._group_membership = group_membership
        self._n_persons = len(group_membership)
        self._n_groups = len(np.unique(group_membership))

        if group_labels is None:
            self._group_labels = [f"Group_{i}" for i in range(self._n_groups)]
        else:
            if len(group_labels) != self._n_groups:
                raise ValueError(
                    f"group_labels length ({len(group_labels)}) must match "
                    f"n_groups ({self._n_groups})"
                )
            self._group_labels = list(group_labels)

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
    def group_membership(self) -> NDArray[np.int_]:
        return self._group_membership.copy()

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
        return self._between_variance / total if total > 0 else 0.0

    def set_group_means(self, means: NDArray[np.float64]) -> Self:
        means = np.asarray(means, dtype=np.float64)
        if means.shape != (self._n_groups,):
            raise ValueError(f"means shape {means.shape} != ({self._n_groups},)")
        self._group_means = means
        return self

    def set_variance_components(self, between: float, within: float = 1.0) -> Self:
        if between < 0:
            raise ValueError("between variance must be non-negative")
        if within <= 0:
            raise ValueError("within variance must be positive")
        self._between_variance = between
        self._within_variance = within
        return self

    def person_prior_mean(self) -> NDArray[np.float64]:
        """Get prior mean for each person based on group membership."""
        return self._group_means[self._group_membership]

    def person_prior_variance(self) -> float:
        """Get prior variance for person abilities (within-group)."""
        return self._within_variance

    def group_prior_variance(self) -> float:
        """Get prior variance for group means (between-group)."""
        return self._between_variance

    def group_sizes(self) -> NDArray[np.int_]:
        """Get size of each group."""
        return np.bincount(self._group_membership, minlength=self._n_groups)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute log-likelihood of responses given theta."""
        return float(np.sum(self._base_model.log_likelihood(responses, theta)))

    def copy(self) -> Self:
        new_model = MultilevelIRTModel(
            base_model=self._base_model.copy(),
            group_membership=self._group_membership.copy(),
            group_labels=self._group_labels.copy(),
        )
        new_model._group_means = self._group_means.copy()
        new_model._between_variance = self._between_variance
        new_model._within_variance = self._within_variance
        new_model._is_fitted = self._is_fitted
        return new_model


class ThreeLevelIRTModel:
    """Three-level IRT model (persons in classrooms in schools).

    Parameters
    ----------
    base_model : BaseItemModel
        IRT model for item responses.
    level2_membership : ndarray of shape (n_persons,)
        Level-2 unit assignment (e.g., classroom).
    level3_membership : ndarray of shape (n_level2_units,)
        Level-3 unit assignment (e.g., school for each classroom).
    level2_labels : list of str, optional
        Labels for level-2 units.
    level3_labels : list of str, optional
        Labels for level-3 units.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        level2_membership: NDArray[np.int_],
        level3_membership: NDArray[np.int_],
        level2_labels: list[str] | None = None,
        level3_labels: list[str] | None = None,
    ) -> None:
        level2_membership = np.asarray(level2_membership)
        level3_membership = np.asarray(level3_membership)

        self._base_model = base_model.copy()
        self._level2_membership = level2_membership
        self._level3_membership = level3_membership

        self._n_persons = len(level2_membership)
        self._n_level2 = len(np.unique(level2_membership))
        self._n_level3 = len(np.unique(level3_membership))

        if len(level3_membership) != self._n_level2:
            raise ValueError(
                f"level3_membership length ({len(level3_membership)}) must match "
                f"number of level-2 units ({self._n_level2})"
            )

        self._level2_labels = level2_labels or [
            f"L2_{i}" for i in range(self._n_level2)
        ]
        self._level3_labels = level3_labels or [
            f"L3_{i}" for i in range(self._n_level3)
        ]

        self._level2_effects = np.zeros(self._n_level2)
        self._level3_effects = np.zeros(self._n_level3)
        self._level2_variance = 0.15
        self._level3_variance = 0.10
        self._within_variance = 1.0
        self._is_fitted = False

    @property
    def n_persons(self) -> int:
        return self._n_persons

    @property
    def n_level2_units(self) -> int:
        return self._n_level2

    @property
    def n_level3_units(self) -> int:
        return self._n_level3

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

    def icc(self, level: Literal["level2", "level3", "total"] = "total") -> float:
        """Compute intraclass correlation.

        Parameters
        ----------
        level : str
            Which ICC to compute:
            - 'level2': proportion of variance at level 2 (classroom)
            - 'level3': proportion of variance at level 3 (school)
            - 'total': proportion of variance at level 2 + 3
        """
        total = self._within_variance + self._level2_variance + self._level3_variance
        if total == 0:
            return 0.0

        if level == "level2":
            return self._level2_variance / total
        elif level == "level3":
            return self._level3_variance / total
        else:
            return (self._level2_variance + self._level3_variance) / total

    def person_prior_mean(self) -> NDArray[np.float64]:
        """Get prior mean for each person."""
        l2_effect = self._level2_effects[self._level2_membership]
        l3_effect = self._level3_effects[
            self._level3_membership[self._level2_membership]
        ]
        return l2_effect + l3_effect


class CrossedRandomEffectsModel:
    """IRT model with crossed random effects (person × item × rater).

    Supports non-nested random effects where the same item can be
    rated by multiple raters, and the same rater can rate multiple items.

    Parameters
    ----------
    base_model : BaseItemModel
        IRT model for item responses.
    n_raters : int
        Number of raters.
    rater_assignments : ndarray of shape (n_observations,) or (n_persons, n_items)
        Rater ID for each observation.
    include_item_effects : bool, default=True
        Whether to include random item effects.
    include_rater_effects : bool, default=True
        Whether to include random rater effects.

    Notes
    -----
    The model is:

        logit(P(X=1)) = θ_person - b_item + γ_rater

    where γ_rater represents rater severity/leniency.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        n_raters: int,
        rater_assignments: NDArray[np.int_] | None = None,
        include_item_effects: bool = True,
        include_rater_effects: bool = True,
    ) -> None:
        self._base_model = base_model.copy()
        self._n_raters = n_raters
        self._rater_assignments = rater_assignments
        self._include_item_effects = include_item_effects
        self._include_rater_effects = include_rater_effects

        self._rater_effects = np.zeros(n_raters)
        self._person_variance = 1.0
        self._item_variance = 0.5
        self._rater_variance = 0.25
        self._is_fitted = False

    @property
    def base_model(self) -> BaseItemModel:
        return self._base_model

    @property
    def n_raters(self) -> int:
        return self._n_raters

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

    def set_rater_effects(self, effects: NDArray[np.float64]) -> Self:
        effects = np.asarray(effects, dtype=np.float64)
        if effects.shape != (self._n_raters,):
            raise ValueError(f"effects shape {effects.shape} != ({self._n_raters},)")
        self._rater_effects = effects
        return self

    def set_variance_components(
        self,
        person: float = 1.0,
        item: float | None = None,
        rater: float | None = None,
    ) -> Self:
        if person <= 0:
            raise ValueError("person variance must be positive")
        self._person_variance = person

        if item is not None:
            if item < 0:
                raise ValueError("item variance must be non-negative")
            self._item_variance = item

        if rater is not None:
            if rater < 0:
                raise ValueError("rater variance must be non-negative")
            self._rater_variance = rater

        return self

    def get_rater_for_observation(
        self,
        person_idx: int,
        item_idx: int,
    ) -> int | None:
        """Get rater ID for a specific observation."""
        if self._rater_assignments is None:
            return None

        if self._rater_assignments.ndim == 1:
            obs_idx = person_idx * self._base_model.n_items + item_idx
            if obs_idx >= len(self._rater_assignments):
                return None
            return int(self._rater_assignments[obs_idx])
        else:
            return int(self._rater_assignments[person_idx, item_idx])

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
        rater_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability with rater effect.

        Parameters
        ----------
        theta : ndarray
            Person abilities.
        item_idx : int, optional
            Item index.
        rater_idx : int, optional
            Rater index. If None, uses rater effect of 0.

        Returns
        -------
        ndarray
            Response probabilities.
        """
        base_prob = self._base_model.probability(theta, item_idx)

        if rater_idx is not None and self._include_rater_effects:
            rater_effect = self._rater_effects[rater_idx]

            if base_prob.ndim == 1:
                log_odds = np.log(base_prob / (1 - base_prob + 1e-10))
                adjusted_log_odds = log_odds + rater_effect
                base_prob = 1 / (1 + np.exp(-adjusted_log_odds))
            else:
                log_odds = np.log(base_prob / (1 - base_prob + 1e-10))
                adjusted_log_odds = log_odds + rater_effect
                base_prob = 1 / (1 + np.exp(-adjusted_log_odds))

        return base_prob


@dataclass
class NestedHierarchy:
    """Describes a nested hierarchical structure.

    Attributes
    ----------
    levels : list of str
        Names of hierarchy levels from lowest to highest.
        E.g., ['student', 'classroom', 'school', 'district']
    memberships : list of ndarray
        Membership arrays for each level transition.
        memberships[0] maps students to classrooms,
        memberships[1] maps classrooms to schools, etc.
    """

    levels: list[str]
    memberships: list[NDArray[np.int_]]

    def __post_init__(self):
        if len(self.memberships) != len(self.levels) - 1:
            raise ValueError(
                f"Need {len(self.levels) - 1} membership arrays for "
                f"{len(self.levels)} levels"
            )

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def get_unit_counts(self) -> dict[str, int]:
        """Get count of units at each level."""
        counts = {self.levels[0]: len(self.memberships[0])}
        for i, membership in enumerate(self.memberships):
            counts[self.levels[i + 1]] = len(np.unique(membership))
        return counts

    def get_full_path(self, unit_idx: int, level: int = 0) -> list[int]:
        """Get full path from lowest level unit to highest."""
        path = [unit_idx]
        current = unit_idx
        for i in range(level, len(self.memberships)):
            current = self.memberships[i][current]
            path.append(current)
        return path
