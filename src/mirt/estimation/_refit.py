"""Recipes for refitting a model with the estimator settings of an earlier fit.

Resampling utilities (bootstrap, jackknife, likelihood-ratio bootstrap and
multi-start fitting) refit copies of a model many times. A fit that records a
:class:`RefitRecipe` is refitted by the estimator class that produced it, with
the same latent density, item priors, equality constraints, quadrature and
acceleration, so every refit estimates the same quantity as the original fit.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.base import BaseEstimator
    from mirt.estimation.latent_density import LatentDensity
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Constructor settings that define how an estimator fits. What it reports
# (``verbose`` and ``compute_standard_errors``) and its worker threads
# (``n_jobs``), which would multiply with the process workers of resampling
# utilities, are chosen by every refitting caller itself.
_EM_SETTINGS = (
    "n_quadpts",
    "max_iter",
    "tol",
    "prob_epsilon",
    "item_optim_maxiter",
    "item_optim_ftol",
    "se_step_size",
    "use_gpu",
    "use_rust",
    "accelerate",
    "item_priors",
    "se_method",
    "constraints",
)
_BIFACTOR_SETTINGS = ("n_quadpts", "max_iter", "tol", "se_method")


@dataclass(frozen=True, eq=False)
class RefitRecipe:
    """The estimator class and settings that produced a fit.

    Attributes
    ----------
    estimator : type
        Estimator class, such as ``EMEstimator`` or ``BifactorEMEstimator``.
    options : dict
        Constructor keyword arguments of the fit. ``verbose``,
        ``compute_standard_errors`` and ``n_jobs`` are left to each refit.
    latent_density : LatentDensity, optional
        Latent density at the end of the fit, or ``None`` for the fixed
        standard normal. Every refit gets its own copy, so an estimated
        density is re-estimated starting from the original estimates.
    """

    estimator: type[BaseEstimator]
    options: dict[str, Any] = field(default_factory=dict)
    latent_density: LatentDensity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", dict(self.options))

    def build(self, **overrides: Any) -> Any:
        """Return a new estimator with the recorded settings and ``overrides``.

        Raises
        ------
        MirtValidationError
            If an override is not a setting of the recorded estimator.
        """
        options = {**self.options, **overrides}
        if self.latent_density is not None and "latent_density" not in overrides:
            options["latent_density"] = self.latent_density
        return _construct(self.estimator, options)


def _construct(estimator: type[BaseEstimator], options: dict[str, Any]) -> Any:
    """Build ``estimator`` after checking that it takes every option.

    Each estimator gets its own copy of a latent density instance, because EM
    updates the density in place.
    """
    from mirt.estimation.latent_density import LatentDensity

    accepted = inspect.signature(estimator).parameters
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        raise MirtValidationError(
            f"{estimator.__name__} does not take {', '.join(unknown)}",
            parameter=unknown[0],
            expected=", ".join(accepted),
        )
    density = options.get("latent_density")
    if isinstance(density, LatentDensity):
        options = {**options, "latent_density": deepcopy(density)}
    return estimator(**options)


def recipe_for(estimator: object) -> RefitRecipe | None:
    """Record how ``estimator`` fits, or ``None`` when it cannot be rebuilt.

    :class:`~mirt.estimation.em.EMEstimator`, subclasses that keep its
    constructor such as ``MixedFormatEMEstimator``, and
    :class:`~mirt.estimation.bifactor_em.BifactorEMEstimator` record a recipe.
    Call it after the fit, when the estimator holds the final latent density.
    """
    from mirt.estimation.bifactor_em import BifactorEMEstimator
    from mirt.estimation.em import EMEstimator

    cls = type(estimator)
    density = None
    if isinstance(estimator, EMEstimator) and cls.__init__ is EMEstimator.__init__:
        names = _EM_SETTINGS
        fitted = estimator._latent_density
        if fitted is not None and not _is_standard_normal(fitted):
            density = deepcopy(fitted)
    elif (
        isinstance(estimator, BifactorEMEstimator)
        and cls.__init__ is BifactorEMEstimator.__init__
    ):
        names = _BIFACTOR_SETTINGS
    else:
        return None
    return RefitRecipe(cls, {name: getattr(estimator, name) for name in names}, density)


def _is_standard_normal(density: LatentDensity) -> bool:
    """Whether ``density`` is the fixed standard normal of a default fit."""
    from mirt.estimation.latent_density import GaussianDensity

    return (
        type(density) is GaussianDensity
        and not density.estimate_mean
        and not density.estimate_cov
        and not np.any(density.mean)
        and np.array_equal(density.cov, np.eye(density.n_dimensions))
    )


def gaussian_population(
    density: LatentDensity | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Return the mean and covariance of a Gaussian latent population.

    A :class:`~mirt.estimation.latent_density.GaussianDensity`, including a
    ``FactorCovarianceDensity`` and the estimated population of fixed-item
    calibration, yields copies of its final mean and covariance. A zero mean
    or an identity covariance is returned as ``None``, the standard-normal
    default of the consumers of ``FitResult.latent_mean`` and
    ``FitResult.latent_covariance``. Other densities yield ``(None, None)``.
    """
    from mirt.estimation.latent_density import GaussianDensity

    if not isinstance(density, GaussianDensity):
        return None, None
    mean = np.array(density.mean, dtype=np.float64) if np.any(density.mean) else None
    cov = None
    if not np.array_equal(density.cov, np.eye(density.n_dimensions)):
        cov = np.array(density.cov, dtype=np.float64)
    return mean, cov


def _is_builtin_bifactor(model: object) -> bool:
    from mirt.estimation.bifactor_em import _require_bifactor_model

    try:
        _require_bifactor_model(model)
    except MirtModelError:
        return False
    return True


def em_estimator_for(
    model_or_result: BaseItemModel | FitResult,
    *,
    recipe: RefitRecipe | None = None,
    **options: Any,
) -> Any:
    """Return an estimator that refits ``model_or_result`` with ``options``.

    Parameters
    ----------
    model_or_result : BaseItemModel or FitResult
        Model to refit, or a fit whose ``refit_recipe`` supplies the
        estimator.
    recipe : RefitRecipe, optional
        Recipe to use instead of the one recorded on a ``FitResult``.
    **options
        Estimator settings that override the recipe, such as ``max_iter`` or
        ``compute_standard_errors``.

    Returns
    -------
    BaseEstimator
        With a recipe, the recorded estimator class and settings. Otherwise
        ``MixedFormatEMEstimator`` for mixed-format models,
        ``BifactorEMEstimator`` for built-in bifactor models when it takes
        every option, and ``EMEstimator`` for other models.

    Raises
    ------
    MirtValidationError
        If the chosen estimator does not take an option.
    """
    from mirt.estimation.bifactor_em import BifactorEMEstimator
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.mixed_format_em import MixedFormatEMEstimator
    from mirt.models.mixed_format import MixedItemModel
    from mirt.results.fit_result import FitResult

    model: object = model_or_result
    if isinstance(model_or_result, FitResult):
        model = model_or_result.model
        if recipe is None:
            recipe = model_or_result.refit_recipe
    if recipe is not None:
        return recipe.build(**options)
    if isinstance(model, MixedItemModel):
        return _construct(MixedFormatEMEstimator, options)
    # Dimension reduction integrates bifactor models on two-dimensional grids
    # instead of the exponential product grid of EMEstimator.
    if _is_builtin_bifactor(model) and set(options) <= set(
        inspect.signature(BifactorEMEstimator).parameters
    ):
        return _construct(BifactorEMEstimator, options)
    return _construct(EMEstimator, options)
