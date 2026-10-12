from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._backend_config import should_use_rust
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust.multigroup import (
    multigroup_e_step_2pl,
    multigroup_e_step_3pl,
    multigroup_e_step_gpcm,
    multigroup_e_step_grm,
    multigroup_e_step_nrm,
)
from mirt.constants import PROB_EPSILON
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.base import _initialize_free_parameters, _parameter_bounds
from mirt.estimation.quadrature import (
    GaussHermiteQuadrature,
    QuadratureRule,
    RectangularQuadrature,
    build_quadrature,
    resolve_quadpts,
    validate_quadpts,
    validate_quadrature_rule,
)
from mirt.exceptions import MirtValidationError
from mirt.multigroup._identification import (
    factor_correlations_identified,
    infer_latent_identification,
)
from mirt.multigroup.invariance import InvarianceSpec, parse_invariance
from mirt.multigroup.latent import MultigroupLatentDensity
from mirt.multigroup.results import MultigroupFitResult
from mirt.utils.data import validate_responses
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.model import MultigroupModel

# A pure item objective, its full parameter vector, the free vector positions
# and the optimizer variables placed there.
_BinaryTerm = tuple[
    Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    NDArray[np.float64],
    NDArray[np.intp],
    NDArray[np.intp],
]
# Largest relative L-BFGS-B tolerance for a joint binary item solve. Item
# losses are large sums, so looser values stop on flat 3PL/4PL directions
# long before the item optimum, and EM then crawls toward the maximum.
_JOINT_ITEM_FTOL = 1e-10
# A rectangular grid grows when a group puts more than this normal mass
# beyond it (about 3.7 SDs), until every group's mean +/- 6 SDs is covered.
_GRID_TAIL_MASS = 1e-4
_GRID_COVERAGE_SD = 6.0
# Parameters with one coordinate per response category, which a group that
# never uses a category cannot identify.
_CATEGORY_PARAMETERS = ("thresholds", "steps", "intercepts", "slopes")
_ACCELERATIONS = ("none", "squarem")
# Growth factor of the largest SQUAREM step length after an accepted
# extrapolation at that length, as in single-group EM.
_SQUAREM_STEP_FACTOR = 4.0
# Largest violation of the latent-mean order that the ordered update treats
# as SLSQP roundoff and removes.
_ORDER_SLACK = 1e-6


def _resolve_correlation_mask(
    model: MultigroupModel, free_correlations: bool | NDArray[np.bool_] | None
) -> NDArray[np.bool_] | None:
    """Return the factor correlations to estimate, validated against the pattern."""
    n_factors = model.n_factors
    if free_correlations is None:
        if not factor_correlations_identified(model):
            return None
        return ~np.eye(n_factors, dtype=np.bool_)
    if isinstance(free_correlations, (bool, np.bool_)):
        if not free_correlations:
            return None
        mask = ~np.eye(n_factors, dtype=np.bool_)
    else:
        mask = np.asarray(free_correlations)
        if mask.dtype != np.bool_ or mask.shape != (n_factors, n_factors):
            raise MirtValidationError(
                f"free_correlations must be a boolean or a Boolean array of "
                f"shape ({n_factors}, {n_factors})",
                parameter="free_correlations",
                value=getattr(mask, "shape", free_correlations),
            )
        if not np.array_equal(mask, mask.T):
            raise MirtValidationError(
                "free_correlations must be symmetric", parameter="free_correlations"
            )
        mask = mask & ~np.eye(n_factors, dtype=np.bool_)
        if not np.any(mask):
            return None
    if n_factors < 2:
        raise MirtValidationError(
            "free_correlations requires a multidimensional model",
            parameter="free_correlations",
            value=free_correlations,
        )
    if not factor_correlations_identified(model):
        raise MirtValidationError(
            "the loading pattern does not identify factor correlations: every "
            "factor needs items that do not load on it, with loadings of full "
            "rank on the other factors; fix loadings at zero with "
            "set_free_parameter_masks or model syntax, or keep the factors "
            "uncorrelated (free_correlations=False)",
            parameter="free_correlations",
            value=free_correlations,
        )
    return mask


def _warm_start_distribution(
    current: GroupLatentDistribution, start: GroupLatentDistribution
) -> None:
    """Copy the moments that ``current`` estimates from a previous fit."""
    free = current.free_covariance_entries()
    if not current.estimate_mean and free is None:
        return
    mean = start.mean if current.estimate_mean else current.mean
    covariance = current.cov
    if free is not None:
        covariance = np.where(free, start.cov, current.cov)
        if not current.estimate_cov:
            # Fixed variances keep the starting correlations.
            scale = np.sqrt(np.diag(start.cov))
            variances = np.sqrt(np.diag(current.cov))
            correlation = start.cov / np.outer(scale, scale)
            covariance = np.where(
                free, correlation * np.outer(variances, variances), current.cov
            )
    try:
        current._set_parameters(np.asarray(mean), covariance)
    except ValueError:
        # A covariance pattern that the start does not fit keeps defaults.
        current._set_parameters(np.asarray(mean), current.cov)


def _tail_masses(
    density: MultigroupLatentDensity, limits: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return each group's largest normal mass outside ``limits``, means and SDs."""
    from scipy.stats import norm

    low, high = limits
    means = np.array(
        [np.asarray(d.mean, dtype=np.float64) for d in density.distributions]
    )
    sds = np.sqrt(np.array([np.diag(d.cov) for d in density.distributions]))
    tails = np.maximum(norm.cdf((low - means) / sds), norm.sf((high - means) / sds))
    return tails.max(axis=1), means, sds


def _covering_grid(
    quadrature: RectangularQuadrature,
    density: MultigroupLatentDensity,
    max_points: int,
) -> RectangularQuadrature | None:
    """Extend a rectangular grid that truncates a group's density, or None.

    A grid centered on the reference group truncates a group several SDs
    away, which shrinks its estimated mean and SD. When a group puts more
    than ``_GRID_TAIL_MASS`` outside the grid, whole node steps are added
    until every group's mean +/- ``_GRID_COVERAGE_SD`` SDs is covered,
    keeping the node spacing and the existing nodes, with at most
    ``max_points`` per dimension. Grids only grow, so EM rebuilds them a few
    times at most.
    """
    tails, means, sds = _tail_masses(density, quadrature.limits)
    if np.all(tails <= _GRID_TAIL_MASS):
        return None
    low, high = quadrature.limits
    spacing = (high - low) / (quadrature.n_points - 1)
    below = max(0.0, low - float(np.min(means - _GRID_COVERAGE_SD * sds)))
    above = max(0.0, float(np.max(means + _GRID_COVERAGE_SD * sds)) - high)
    steps_below, steps_above = np.ceil(below / spacing), np.ceil(above / spacing)
    room = max_points - quadrature.n_points
    if steps_below + steps_above > room:
        scale = room / (steps_below + steps_above)
        steps_below, steps_above = (
            np.floor(steps_below * scale),
            np.floor(steps_above * scale),
        )
    if steps_below + steps_above == 0:
        return None
    return RectangularQuadrature(
        n_points=quadrature.n_points + int(steps_below + steps_above),
        n_dimensions=quadrature.n_dimensions,
        limits=(low - steps_below * spacing, high + steps_above * spacing),
    )


def _warn_outside_grid(
    model: MultigroupModel,
    density: MultigroupLatentDensity,
    limits: tuple[float, float],
) -> None:
    """Warn when a fitted group density still reaches beyond the grid.

    The grid grows to cover the groups (see :func:`_covering_grid`) up to
    twice its default number of points; a group with more than 0.1% of its
    normal mass outside the final grid is truncated there.
    """
    from scipy.stats import norm

    low, high = limits
    for label, distribution in zip(
        model.group_labels, density.distributions, strict=True
    ):
        mean = np.asarray(distribution.mean, dtype=np.float64)
        sd = np.sqrt(np.diag(distribution.cov))
        outside = float(
            np.max(norm.cdf((low - mean) / sd) + norm.sf((high - mean) / sd))
        )
        if outside > 1e-3:
            warnings.warn(
                f"group {label!r} has {outside:.2%} of its fitted latent "
                f"distribution (means {np.round(mean, 3).tolist()}, SDs "
                f"{np.round(sd, 3).tolist()}) outside the quadrature grid "
                f"[{low:g}, {high:g}] of the reference group's scale; refit "
                "with a reference group nearer the middle of the groups or "
                "with more quadrature points",
                UserWarning,
                stacklevel=3,
            )


def _check_group_support(
    model: MultigroupModel, responses: Sequence[NDArray[np.int_]]
) -> None:
    """Warn about unobserved items and reject unidentified category parameters.

    An item without responses in a group leaves its group-specific
    parameters there without information: the model excludes them from
    estimation and the parameter count (see
    :meth:`MultigroupModel.unobserved_items`), and this warns. A response
    category that a group never uses leaves group-specific category
    parameters at infinity, so it is rejected, as R's ``multipleGroup``
    does. Call after the invariance links are applied.
    """
    observed_items = [np.any(data >= 0, axis=0) for data in responses]
    item_major = [
        name for name in model.parameter_names if model._parameter_is_item_major(name)
    ]
    for g, observed in enumerate(observed_items):
        masks = model._raw_free_parameter_masks(g)
        dropped = [
            model.item_names[item]
            for item in np.flatnonzero(~observed)
            if any(
                not model.is_item_parameter_shared(name, int(item))
                and np.any(masks[name][item])
                for name in item_major
            )
        ]
        if dropped:
            warnings.warn(
                f"group {model.group_labels[g]!r} has no responses to items "
                f"{dropped}; their group-specific parameters in that group are "
                "not estimated, not counted, and reported as NaN",
                UserWarning,
                stacklevel=3,
            )
    if not model.is_polytomous:
        return
    category_names = [name for name in _CATEGORY_PARAMETERS if name in item_major]
    for g, (data, observed) in enumerate(zip(responses, observed_items, strict=True)):
        group = model.get_group_model(g)
        masks = model.effective_free_parameter_masks(g)
        for item in np.flatnonzero(observed):
            column = data[:, item]
            used = np.unique(column[column >= 0])
            n_categories = int(group._n_categories[item])
            if used.size == n_categories:
                continue
            estimated = [
                name
                for name in category_names
                if not model.is_item_parameter_shared(name, int(item))
                and np.any(masks[name][item])
            ]
            if estimated:
                missing = sorted(set(range(n_categories)) - {int(k) for k in used})
                raise MirtValidationError(
                    f"group {model.group_labels[g]!r} never uses categories "
                    f"{missing} of item {model.item_names[item]!r}, whose "
                    f"{', '.join(estimated)} are estimated separately in each "
                    "group; collapse unused categories or constrain these "
                    "parameters across groups (R's multipleGroup also stops)",
                    parameter="responses",
                )


class MultigroupEMEstimator:
    """EM estimator for simultaneous multigroup IRT estimation.

    This estimator fits IRT models across multiple groups simultaneously,
    with support for various invariance constraints and group-specific
    latent distributions.

    Every group is integrated on one grid with its own normal prior masses,
    so a group whose latent mean or variance differs from the reference
    group's is integrated as accurately as the reference group. The grid and
    its defaults are those of :class:`~mirt.estimation.em.EMEstimator` and
    R's ``multipleGroup``.

    Parameters
    ----------
    n_quadpts : int, optional
        Quadrature points per latent dimension, at least 5. The default
        depends on the number of factors, as in single-group EM and R mirt:
        61 points for one factor, 31 for two, 15 for three, 9 for four, 7
        for five and 5 for six (see
        :func:`~mirt.estimation.quadrature.default_quadpts`).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood change.
    verbose : bool
        Print iteration progress.
    prob_epsilon : float
        Minimum probability for numerical stability.
    item_optim_maxiter : int
        Maximum iterations for item parameter optimization.
    item_optim_ftol : float
        Relative tolerance for item parameter optimization. Built-in 1PL-4PL
        items, whose free coordinates are solved jointly, use at most 1e-10.
    quadrature : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule. ``"rectangular"`` places ``n_quadpts`` equally
        spaced nodes on ``[-6, 6]`` of the reference group's metric in every
        dimension, as R mirt does, and weights them by each group's normal
        density; groups far from the reference group keep as many effective
        nodes as it has. When a group puts more than 0.01% of its normal mass
        beyond the grid, which happens about 3.7 SDs from an edge, EM adds
        nodes at the same spacing until every group's mean +/- 6 SDs is
        covered, with at most twice ``n_quadpts`` points per dimension; R's
        fixed grid instead truncates such a group and shrinks its estimated
        mean and SD. A group still extending beyond the largest grid warns.
        ``"gauss-hermite"`` uses standard-normal Gauss-Hermite nodes, which
        suit coarse grids but leave few effective nodes to a group whose mean
        lies several standard deviations away. ``"auto"`` is rectangular for
        15 or more points per dimension and Gauss-Hermite otherwise.
    accelerate : {"none", "squarem"}, default="none"
        EM acceleration. ``"squarem"`` applies SQUAREM extrapolation (Varadhan
        and Roland, 2008) to every group's free item coordinates and latent
        moments at once, with step-length adaptation, projection onto the
        parameter boxes and a monotone fallback to the plain EM step, as
        :class:`~mirt.estimation.em.EMEstimator` does. Latent means drift
        slowly along flat ridges of the likelihood when groups are linked
        only through a few anchor items, as in vertical scaling, where it
        needs several times fewer E-steps. Extrapolation needs precise
        M-steps, so item optimizers then use a relative tolerance of at most
        ``1e-10``. ``max_iter`` still bounds the number of M-steps, while
        ``n_iterations`` counts E-steps, including the evaluation of each
        extrapolated point, and ``convergence_history`` records the
        log-likelihood at accepted iterates only.
    compute_standard_errors : bool, default=False
        Compute observed-information standard errors and the covariance of
        every free item parameter and latent moment, as R's
        ``multipleGroup(SE=TRUE)``. Each group's information is the exact
        Louis information of its item parameters under its own prior, with
        latent mean and covariance columns from central differences of the
        exact score (Oakes' identity), and the groups' informations are
        summed over the shared coordinates (see
        :class:`~mirt.multigroup.MultigroupFitResult`). Latent moments are
        estimated in their natural metric, means and covariances, as R
        reports them. Item coordinates on an optimizer bound and the slopes
        that identify an exploratory rotation get ``NaN`` errors. Ordered
        means (``mean_order``) are treated as unconstrained.

    References
    ----------
    Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods
    for accelerating the convergence of any EM algorithm. *Scandinavian
    Journal of Statistics*, 35(2), 335-353.
    """

    def __init__(
        self,
        n_quadpts: int | None = None,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        prob_epsilon: float = 1e-10,
        item_optim_maxiter: int = 50,
        item_optim_ftol: float = 1e-6,
        quadrature: QuadratureRule = "auto",
        accelerate: Literal["none", "squarem"] = "none",
        compute_standard_errors: bool = False,
    ) -> None:
        if not isinstance(compute_standard_errors, (bool, np.bool_)):
            raise MirtValidationError(
                "compute_standard_errors must be a boolean",
                parameter="compute_standard_errors",
                value=compute_standard_errors,
                expected="bool",
            )
        if not isinstance(accelerate, str) or accelerate not in _ACCELERATIONS:
            raise MirtValidationError(
                "accelerate must be 'none' or 'squarem'",
                parameter="accelerate",
                value=accelerate,
                expected="'none' or 'squarem'",
            )
        for name, value, minimum in (
            ("max_iter", max_iter, 1),
            ("item_optim_maxiter", item_optim_maxiter, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name, value in (("tol", tol), ("item_optim_ftol", item_optim_ftol)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(prob_epsilon) or not 0 < prob_epsilon < 0.5:
            raise ValueError("prob_epsilon must be between 0 and 0.5")

        self.n_quadpts = None if n_quadpts is None else validate_quadpts(n_quadpts)
        self.quadrature = validate_quadrature_rule(quadrature)
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol
        self.accelerate = accelerate
        self.compute_standard_errors = bool(compute_standard_errors)
        self._precise_m_steps = False

        self._quadrature: GaussHermiteQuadrature | RectangularQuadrature | None = None
        self._latent_density: MultigroupLatentDensity | None = None
        self._convergence_history: list[float] = []

    def fit(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        invariance: InvarianceSpec | str = "configural",
        reference_group: int = 0,
        *,
        fixed_parameters: Mapping[str, Mapping[int, float | NDArray[np.float64]]]
        | None = None,
        mean_order: Sequence[int] | None = None,
        initial_latent: Sequence[GroupLatentDistribution] | None = None,
        free_correlations: bool | NDArray[np.bool_] | None = None,
    ) -> MultigroupFitResult:
        """Fit multigroup model with simultaneous EM.

        Parameters
        ----------
        model : MultigroupModel
            The multigroup model to fit.
        responses : list of ndarray
            Response matrices, one per group. Each has shape (n_persons_g, n_items).
            An item without responses in a group has no group-specific
            parameters there: they are neither estimated nor counted, and
            :meth:`MultigroupFitResult.coef` reports them as ``NaN``.
        invariance : InvarianceSpec or str
            Invariance level or custom specification. Its ``free_means`` and
            ``free_variances`` options fix or free the nonreference latent
            moments independently of the level.
        reference_group : int
            Index of reference group (fixed mean 0 and unit variances).
        fixed_parameters : mapping, optional
            Stored parameter name -> item index -> fixed scalar or parameter row.
            These values are applied to all groups and retained throughout fitting.
        mean_order : sequence of int, optional
            Permutation of all group indices in nondecreasing population-mean
            order. Supported for unidimensional Gaussian latent distributions.
        initial_latent : sequence of GroupLatentDistribution, optional
            Starting latent distributions, one per group, such as those of a
            previous fit. Only means and covariances that this fit estimates
            are copied. Together with already fitted group models this
            warm-starts nested refits.
        free_correlations : bool or ndarray of bool, optional
            Factor correlations of a multidimensional model, estimated in
            every group, including the reference group, whose variances stay
            at one; R's ``multipleGroup`` estimates them for ``COV`` terms of
            the model syntax. ``None`` (the default) estimates every
            correlation when the loading pattern identifies them, as a
            confirmatory pattern set with ``set_free_parameter_masks`` does
            when every factor has items that do not load on it, with
            loadings of full rank on the other factors. Otherwise it keeps
            the factors of the reference group uncorrelated, which
            identifies the rotation of an exploratory model. ``True`` estimates every correlation, ``False`` none, and a
            symmetric Boolean ``(n_factors, n_factors)`` array the selected
            pairs. Nonreference groups that estimate variances estimate the
            covariances of these pairs too, or every covariance when the
            reference factors are uncorrelated.

        Returns
        -------
        MultigroupFitResult
            Fitted model results.

        Raises
        ------
        MirtValidationError
            If ``free_correlations`` frees correlations that the loading
            pattern does not identify, ``free_means`` or ``free_variances``
            is True for moments without anchors, or a group never uses a
            response category of an item whose category parameters it
            estimates separately.
        """
        if len(responses) != model.n_groups:
            raise ValueError(
                f"Number of response matrices ({len(responses)}) must match "
                f"n_groups ({model.n_groups})"
            )

        responses = [validate_responses(r, n_items=model.n_items) for r in responses]
        for g, r in enumerate(responses):
            category_counts = (
                np.asarray(model.get_group_model(g)._n_categories)
                if model.is_polytomous
                else np.full(model.n_items, 2)
            )
            if np.any(r >= category_counts[None, :]):
                raise ValueError(f"Group {g} contains invalid response categories")
            if not np.any(r >= 0):
                raise ValueError(f"Group {g} must contain observed responses")
        mean_order = self._validate_mean_order(mean_order, model)

        inv_spec = parse_invariance(invariance)

        # One grid for every group: each group's prior reweights the fixed
        # nodes, as in R, rather than shifting them.
        base_points = resolve_quadpts(self.n_quadpts, model.n_factors)
        self._quadrature = build_quadrature(
            base_points, model.n_factors, self.quadrature
        )

        self._latent_density = MultigroupLatentDensity(
            n_groups=model.n_groups,
            n_factors=model.n_factors,
            reference_group=reference_group,
        )
        if fixed_parameters is not None:
            model.fix_item_parameters(fixed_parameters)
        model._set_unobserved_items(
            [np.flatnonzero(np.all(data < 0, axis=0)).tolist() for data in responses]
        )
        # Fixed zero loadings decide which factor correlations are identified.
        correlation_mask = _resolve_correlation_mask(model, free_correlations)
        for distribution in self._latent_density.distributions:
            distribution.correlation_mask = correlation_mask
        identification = infer_latent_identification(
            model, inv_spec, reference_group, mean_order=mean_order
        )
        for distribution, (estimate_mean, estimate_cov) in zip(
            self._latent_density.distributions, identification, strict=True
        ):
            distribution.estimate_mean = estimate_mean
            distribution.estimate_cov = estimate_cov
        if initial_latent is not None:
            if len(initial_latent) != model.n_groups:
                raise ValueError(
                    "initial_latent must contain one distribution per group"
                )
            for current, start in zip(
                self._latent_density.distributions, initial_latent, strict=True
            ):
                _warm_start_distribution(current, start)
        inv_spec.apply_to_model(model)
        _check_group_support(model, responses)

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            if not group_model._is_fitted:
                _initialize_free_parameters(group_model)
        model.synchronize_shared_parameters()

        self._convergence_history = []
        squarem = self.accelerate == "squarem"
        run = self._run_squarem if squarem else self._run_em
        # Extrapolation needs precise M-steps, so SQUAREM tightens the item
        # optimizer tolerance while it runs.
        self._precise_m_steps = squarem
        try:
            posterior_weights, group_lls, converged, n_iterations = run(
                model, responses, inv_spec, mean_order, base_points
            )
        finally:
            self._precise_m_steps = False

        for g in range(model.n_groups):
            model.get_group_model(g)._is_fitted = True

        final_ll = float(sum(group_lls))
        if isinstance(self._quadrature, RectangularQuadrature):
            _warn_outside_grid(model, self._latent_density, self._quadrature.limits)

        group_n = [r.shape[0] for r in responses]
        total_n = sum(group_n)

        n_item_params = model.n_parameters
        n_latent_params = self._latent_density.n_parameters
        n_params = n_item_params + n_latent_params

        aic = -2 * final_ll + 2 * n_params
        bic = -2 * final_ll + np.log(total_n) * n_params

        errors: dict[str, Any] = {}
        if self.compute_standard_errors:
            from mirt.multigroup._standard_errors import multigroup_covariance

            try:
                estimate = multigroup_covariance(
                    model,
                    responses,
                    self._quadrature,
                    self._latent_density,
                    self._parameter_bound,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                warnings.warn(
                    f"standard errors could not be computed: {error}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                errors = {
                    "standard_errors": estimate.standard_errors,
                    "latent_standard_errors": estimate.latent_standard_errors,
                    "vcov": estimate.covariance,
                    "vcov_labels": estimate.labels,
                }

        return MultigroupFitResult(
            model=model,
            invariance=inv_spec.level,
            log_likelihood=final_ll,
            n_iterations=n_iterations,
            converged=converged,
            group_log_likelihoods=group_lls,
            group_n_observations=group_n,
            latent_distributions=[d.copy() for d in self._latent_density.distributions],
            aic=aic,
            bic=bic,
            n_parameters=n_params,
            n_observations=total_n,
            **errors,
        )

    def _em_update(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        invariance: InvarianceSpec,
        mean_order: tuple[int, ...] | None,
    ) -> None:
        """Run the M-step of the items and then of the latent distributions."""
        self._m_step(model, responses, posterior_weights, invariance)
        if mean_order is None:
            for g in range(model.n_groups):
                n_k = posterior_weights[g].sum(axis=0)
                self._latent_density.update(self._quadrature.nodes, n_k, g)
        else:
            self._update_ordered_latent_density(posterior_weights, mean_order)

    def _grow_grid(self, base_points: int) -> bool:
        """Extend a rectangular grid that truncates a group; return whether it grew."""
        if not isinstance(self._quadrature, RectangularQuadrature):
            return False
        grid = _covering_grid(self._quadrature, self._latent_density, 2 * base_points)
        if grid is None:
            return False
        self._quadrature = grid
        return True

    def _log_iteration(self, log_likelihood: float) -> None:
        """Record one E-step's log-likelihood and print it when verbose."""
        self._convergence_history.append(log_likelihood)
        if self.verbose:
            print(
                f"Iteration {len(self._convergence_history)}: LL = {log_likelihood:.4f}"
            )

    def _run_em(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        invariance: InvarianceSpec,
        mean_order: tuple[int, ...] | None,
        base_points: int,
    ) -> tuple[list[NDArray[np.float64]], list[float], bool, int]:
        """Iterate plain EM; return the final posteriors, group LLs and status."""
        prev_ll = -np.inf
        n_iterations = 0
        converged = False
        state_is_current = False

        for iteration in range(self.max_iter):
            posterior_weights, group_lls = self._e_step(model, responses)
            state_is_current = True

            current_ll = float(sum(group_lls))
            self._log_iteration(current_ll)

            if abs(current_ll - prev_ll) < self.tol:
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                n_iterations = iteration + 1
                break

            prev_ll = current_ll
            n_iterations = iteration + 1

            self._em_update(model, responses, posterior_weights, invariance, mean_order)
            state_is_current = False
            if self._grow_grid(base_points):
                # The likelihood is now integrated more accurately, so the
                # next change does not measure convergence.
                prev_ll = -np.inf

        if not state_is_current:
            posterior_weights, group_lls = self._e_step(model, responses)
            final_ll = float(sum(group_lls))
            self._convergence_history.append(final_ll)
            converged = abs(final_ll - prev_ll) < self.tol
        return posterior_weights, group_lls, converged, n_iterations

    def _run_squarem(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        invariance: InvarianceSpec,
        mean_order: tuple[int, ...] | None,
        base_points: int,
    ) -> tuple[list[NDArray[np.float64]], list[float], bool, int]:
        """Iterate SQUAREM cycles over item parameters and latent moments.

        Each cycle takes two EM steps from the current iterate, extrapolates
        along the SqS3 step length (Varadhan and Roland, 2008) and stabilizes
        the extrapolated point with one more EM step, as
        :class:`~mirt.estimation.em.EMEstimator` does. The point moves every
        group's free item coordinates and latent moments together (see
        :class:`~mirt.multigroup._acceleration.MultigroupParameters`). It is
        kept only if its log-likelihood is at least that of the first EM
        step, so the fit stays monotone; otherwise the cycle keeps the second
        EM step. Convergence uses the plain EM rule on the first step of each
        cycle. A grid that grows to cover a group restarts the cycle from
        that step, whose log-likelihood then does not measure convergence.
        """
        from mirt.estimation._acceleration import squarem_point, squarem_step_length
        from mirt.multigroup._acceleration import MultigroupParameters

        parameters = MultigroupParameters(
            model, self._latent_density, self._parameter_bound, mean_order
        )
        n_e_steps = n_m_steps = 0

        def evaluate() -> tuple[list[NDArray[np.float64]], list[float], float]:
            nonlocal n_e_steps
            n_e_steps += 1
            posterior, group_lls = self._e_step(model, responses)
            return posterior, group_lls, float(sum(group_lls))

        def update(posterior: list[NDArray[np.float64]]) -> NDArray[np.float64]:
            nonlocal n_m_steps
            n_m_steps += 1
            self._em_update(model, responses, posterior, invariance, mean_order)
            return parameters.get()

        step_max = 1.0
        state = evaluate()
        self._log_iteration(state[2])
        converged = False
        while n_m_steps < self.max_iter:
            start_ll = state[2]
            start = parameters.get()
            first = update(state[0])
            grew = self._grow_grid(base_points)
            state = evaluate()
            first_ll = state[2]
            self._log_iteration(first_ll)
            converged = not grew and abs(first_ll - start_ll) < self.tol
            if converged or n_m_steps >= self.max_iter:
                if converged and self.verbose:
                    print(f"Converged after {n_e_steps} E-steps")
                break
            if grew:
                continue

            second = update(state[0])
            # A unit step length reproduces the second EM step itself.
            alpha = squarem_step_length(start, first, second, step_max)
            accepted = alpha == 1.0
            if not accepted:
                point = squarem_point(
                    start, first, second, alpha, parameters.lower, parameters.upper
                )
                if parameters.set(point):
                    trial = evaluate()
                    if np.isfinite(trial[2]) and n_m_steps < self.max_iter:
                        update(trial[0])
                        trial = evaluate()
                    accepted = bool(np.isfinite(trial[2]) and trial[2] >= first_ll)
                    if accepted:
                        state = trial
                    else:
                        parameters.set(second)
            if alpha == step_max:
                step_max = (
                    step_max * _SQUAREM_STEP_FACTOR
                    if accepted
                    else max(1.0, step_max / _SQUAREM_STEP_FACTOR)
                )
            if alpha == 1.0 or not accepted:
                state = evaluate()
            self._log_iteration(state[2])
            if self._grow_grid(base_points):
                state = evaluate()
                self._log_iteration(state[2])
        return state[0], state[1], converged, n_e_steps

    def _e_step(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step: compute posterior weights for each group.

        Returns
        -------
        posterior_weights : list of ndarray
            Posterior weights per group, shape (n_persons_g, n_quad).
        group_lls : list of float
            Marginal log-likelihood per group.
        """
        if should_use_rust() and self._can_use_rust_e_step(model):
            return self._e_step_rust(model, responses)
        return self._e_step_python(model, responses)

    def _can_use_rust_e_step(self, model: MultigroupModel) -> bool:
        """Check if Rust E-step can be used for this model."""
        if model.n_factors != 1:
            return False
        if not all(
            uses_builtin_model_hooks(group_model, likelihood=True)
            for group_model in model.group_models
        ):
            return False
        model_name = model.model_name
        if model.is_polytomous:
            return model_name in ("GRM", "GPCM", "PCM", "NRM")
        return model_name in ("2PL", "1PL", "3PL")

    def _e_step_rust(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for parallel processing."""
        quad_points = self._quadrature.nodes.ravel()
        quad_weights = self._quadrature.weights

        prior_means = np.array(
            [
                self._latent_density.distributions[g].mean[0]
                for g in range(model.n_groups)
            ]
        )
        prior_vars = np.array(
            [
                self._latent_density.distributions[g].cov[0, 0]
                for g in range(model.n_groups)
            ]
        )

        if model.model_name == "GRM":
            return self._e_step_rust_grm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        if model.model_name in ("GPCM", "PCM"):
            return self._e_step_rust_gpcm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        if model.model_name == "NRM":
            return self._e_step_rust_nrm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        disc_list = []
        diff_list = []
        guess_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            diff = params.get("difficulty", np.zeros(model.n_items))

            disc_list.append(disc.ravel())
            diff_list.append(diff.ravel())

            if "guessing" in params:
                guess_list.append(params["guessing"].ravel())

        if model.model_name in ("2PL", "1PL"):
            result = multigroup_e_step_2pl(
                responses,
                quad_points,
                quad_weights,
                disc_list,
                diff_list,
                prior_means,
                prior_vars,
            )
        else:
            result = multigroup_e_step_3pl(
                responses,
                quad_points,
                quad_weights,
                disc_list,
                diff_list,
                guess_list,
                prior_means,
                prior_vars,
            )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_grm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for GRM models."""
        disc_list = []
        thresh_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            thresh = params.get("thresholds", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            disc_list.append(disc.ravel())
            thresh_list.append(thresh)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_grm(
            responses,
            quad_points,
            quad_weights,
            disc_list,
            thresh_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_gpcm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for GPCM models."""
        disc_list = []
        steps_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            steps = params.get("steps", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            max_cats = max(group_model._n_categories)
            steps_full = np.zeros((model.n_items, max_cats))
            for j, nc in enumerate(group_model._n_categories):
                steps_full[j, 1:nc] = steps[j, : nc - 1]

            disc_list.append(disc.ravel())
            steps_list.append(steps_full)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_gpcm(
            responses,
            quad_points,
            quad_weights,
            disc_list,
            steps_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_nrm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for NRM models."""
        slopes_list = []
        intercepts_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            slopes = params.get("slopes", np.zeros((model.n_items, 1)))
            intercepts = params.get("intercepts", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            max_cats = max(group_model._n_categories)
            slopes_full = np.zeros((model.n_items, max_cats))
            intercepts_full = np.zeros((model.n_items, max_cats))
            for j, nc in enumerate(group_model._n_categories):
                slopes_full[j, :nc] = slopes[j, :nc]
                intercepts_full[j, :nc] = intercepts[j, :nc]

            slopes_list.append(slopes_full)
            intercepts_list.append(intercepts_full)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_nrm(
            responses,
            quad_points,
            quad_weights,
            slopes_list,
            intercepts_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_python(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """Fallback Python E-step implementation."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_quad = len(quad_weights)

        posterior_weights = []
        group_lls = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            group_responses = responses[g]
            n_persons = group_responses.shape[0]

            if hasattr(group_model, "log_likelihood_batch"):
                log_likelihoods = group_model.log_likelihood_batch(
                    group_responses, quad_points
                )
            else:
                log_likelihoods = np.zeros((n_persons, n_quad))
                for q in range(n_quad):
                    theta_q = quad_points[q : q + 1]
                    log_likelihoods[:, q] = group_model.log_likelihood(
                        group_responses, theta_q
                    )

            log_prior_mass = self._latent_density.log_quadrature_mass(
                quad_points, quad_weights, g
            )
            log_joint = log_likelihoods + log_prior_mass[None, :]
            log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
            log_posterior = log_joint - log_marginal

            post_w = np.exp(log_posterior)
            posterior_weights.append(post_w)
            group_ll = np.sum(log_marginal)
            group_lls.append(group_ll)

        return posterior_weights, group_lls

    def _m_step(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        invariance: InvarianceSpec,
    ) -> None:
        """M-step: update parameters respecting constraints.

        Shared parameters are fitted to the expected counts of every group at
        once, and group-specific parameters to each group's own counts.
        Built-in 1PL/2PL items whose parameters are all shared or all
        group-specific are solved together by batched Newton steps (see
        ``_newton_m_step``); the other items are optimized one at a time
        (see ``_optimize_item``).
        """
        quad_points = self._quadrature.nodes
        # Item updates never change which coordinates are free.
        masks = [model.effective_free_parameter_masks(g) for g in range(model.n_groups)]
        counts = [
            self._all_item_counts(group, data, posterior)
            for group, data, posterior in zip(
                model.group_models, responses, posterior_weights, strict=True
            )
        ]
        items: Sequence[int] = range(model.n_items)
        if self._uses_newton_m_step(model):
            items = self._newton_m_step(model, counts, masks, quad_points)
        for item_idx in items:
            self._optimize_item(
                model,
                item_idx,
                responses,
                posterior_weights,
                quad_points,
                masks=masks,
                counts=[group_counts[item_idx] for group_counts in counts],
            )

        model.synchronize_shared_parameters()

    def _uses_newton_m_step(self, model: MultigroupModel) -> bool:
        """Return whether every group holds unrestricted built-in 1PL/2PL items.

        The batched Newton solver maximizes the unclipped item objective,
        which matches the clipped one only for the default probability
        clipping. Estimators that customize item optimization keep it.
        """
        from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

        first = model.get_group_model(0)
        return (
            type(first) in (OneParameterLogistic, TwoParameterLogistic)
            and 0.0 < self.prob_epsilon <= PROB_EPSILON
            and not any(
                name in vars(self)
                or getattr(type(self), name) is not getattr(MultigroupEMEstimator, name)
                for name in ("_optimize_item", "_optimize_binary_block")
            )
            and all(
                type(group) is type(first)
                and uses_builtin_model_hooks(group)
                and not group._free_parameter_restrictions
                and tuple(group._parameters) == ("discrimination", "difficulty")
                for group in model.group_models
            )
        )

    def _newton_m_step(
        self,
        model: MultigroupModel,
        counts: Sequence[NDArray[np.float64]],
        masks: Sequence[Mapping[str, NDArray[np.bool_]]],
        quad_points: NDArray[np.float64],
    ) -> list[int]:
        """Solve 1PL/2PL items by batched Newton steps; return the other items.

        ``counts`` holds each group's expected category counts, of shape
        ``(n_items, n_points, 2)``. An item whose free parameters are all
        shared is one logistic regression on the counts pooled over groups,
        and an item whose parameters are all group-specific is one regression
        per group that observed it. A 2PL item with shared slopes and
        group-specific difficulties, as under metric invariance, is one
        logistic regression on every group's counts with the shared slopes
        and one intercept per group (see :meth:`_newton_shared_slopes`).
        Items with fixed coordinates or other partly shared parameters, and
        items with a solve that does not converge or leaves the optimizer
        box, are returned for the itemwise path.
        """
        from mirt.estimation._logistic_newton import newton_logistic_items
        from mirt.models.dichotomous import OneParameterLogistic

        groups = model.group_models
        n_items = model.n_items
        estimate_slopes = type(groups[0]) is not OneParameterLogistic
        names = ("discrimination", "difficulty") if estimate_slopes else ("difficulty",)
        values = {
            name: np.stack(
                [np.reshape(group.parameters[name], (n_items, -1)) for group in groups]
            )
            for name in ("discrimination", "difficulty")
        }
        group_counts = np.stack(counts)
        observed = np.any(group_counts, axis=(2, 3))
        free = np.all(
            [
                np.reshape(group_masks[name], (n_items, -1)).all(axis=1)
                for group_masks in masks
                for name in names
            ],
            axis=0,
        )
        shared = np.zeros((len(names), n_items), dtype=np.bool_)
        for row, name in enumerate(names):
            shared[row, model.get_shared_items(name)] = True
        equal = np.all(
            [np.all(array == array[0], axis=(0, 2)) for array in values.values()],
            axis=0,
        )
        pooled = free & shared.all(axis=0) & equal
        separate = free & ~shared.any(axis=0)
        slopes_shared = np.zeros(n_items, dtype=np.bool_)
        if estimate_slopes:
            slopes_shared = (
                free
                & shared[0]
                & ~shared[1]
                & np.all(
                    values["discrimination"] == values["discrimination"][0], (0, 2)
                )
                & observed.all(axis=0)
            )
        remaining = ~(pooled | separate | slopes_shared)
        pooled &= observed.any(axis=0)
        if np.any(slopes_shared):
            remaining |= self._newton_shared_slopes(
                groups, np.flatnonzero(slopes_shared), group_counts, values, quad_points
            )

        # One problem per pooled item (group -1) and per observed group of a
        # group-specific item.
        separate_groups, separate_items = np.nonzero(separate & observed)
        items = np.concatenate([np.flatnonzero(pooled), separate_items])
        members = np.concatenate(
            [np.full(int(pooled.sum()), -1), separate_groups]
        ).astype(np.intp)
        if items.size:
            source = np.maximum(members, 0)
            problem_counts = np.where(
                (members < 0)[:, None, None],
                group_counts.sum(axis=0)[items],
                group_counts[source, items],
            )
            slopes = values["discrimination"][source, items]
            difficulty = values["difficulty"][source, items, 0]
            new_slopes, intercepts, accepted = newton_logistic_items(
                quad_points,
                problem_counts[:, :, 1],
                problem_counts.sum(axis=2),
                slopes,
                -slopes.sum(axis=1) * difficulty,
                estimate_slopes=estimate_slopes,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                new_difficulty = -intercepts / new_slopes.sum(axis=1)
            for name, estimates in (
                ("difficulty", new_difficulty[:, None]),
                ("discrimination", new_slopes),
            ):
                low, high = self._parameter_bound(groups[0], name)
                accepted &= np.all((estimates >= low) & (estimates <= high), axis=1)
            remaining[items[~accepted]] = True
            keep = ~remaining[items]
            for g, group in enumerate(groups):
                rows = keep & ((members < 0) | (members == g))
                if not np.any(rows):
                    continue
                updated = {
                    name: np.array(group.parameters[name], copy=True) for name in names
                }
                updated["difficulty"][items[rows]] = new_difficulty[rows]
                if estimate_slopes:
                    slope_values = updated["discrimination"]
                    slope_values[items[rows]] = new_slopes[rows].reshape(
                        (-1, *slope_values.shape[1:])
                    )
                group.set_parameters(**updated)
        return [int(item) for item in np.flatnonzero(remaining)]

    def _newton_shared_slopes(
        self,
        groups: Sequence[BaseItemModel],
        items: NDArray[np.intp],
        group_counts: NDArray[np.float64],
        values: Mapping[str, NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Solve 2PL items with shared slopes and group-specific difficulties.

        Every group ``g`` has logits ``theta @ a + c_g`` at the nodes. Writing
        ``c_g = c_0 + delta_g`` makes this one logistic regression of each
        item on all groups' counts, with design rows ``(theta_q, e_g)`` for
        the group indicators ``e_g`` of the nonfirst groups, which the batched
        Newton solver handles like any other. Difficulties are
        ``-c_g / sum(a)``, as for the other Newton problems.

        Returns
        -------
        ndarray of bool
            Item flags, ``True`` for items whose solve did not converge or left
            the optimizer box; those keep their values for the itemwise path.
        """
        from mirt.estimation._logistic_newton import newton_logistic_items

        n_groups, n_points = len(groups), quad_points.shape[0]
        n_factors = quad_points.shape[1]
        design = np.hstack(
            (
                np.tile(quad_points, (n_groups, 1)),
                np.repeat(np.eye(n_groups)[:, 1:], n_points, axis=0),
            )
        )
        counts = np.moveaxis(group_counts[:, items], 0, 1)
        correct = counts[..., 1].reshape(items.size, n_groups * n_points)
        observed = counts.sum(axis=3).reshape(items.size, n_groups * n_points)
        slopes = values["discrimination"][0, items]
        intercepts = -slopes.sum(axis=1) * values["difficulty"][:, items, 0]
        coefficients, base, accepted = newton_logistic_items(
            design,
            correct,
            observed,
            np.hstack((slopes, (intercepts[1:] - intercepts[0]).T)),
            intercepts[0],
        )
        new_slopes = coefficients[:, :n_factors]
        new_intercepts = base[None, :] + np.vstack(
            (np.zeros(items.size), coefficients[:, n_factors:].T)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            new_difficulty = -new_intercepts / new_slopes.sum(axis=1)
        for name, estimates in (
            ("discrimination", new_slopes),
            ("difficulty", new_difficulty.T),
        ):
            low, high = self._parameter_bound(groups[0], name)
            accepted &= np.all((estimates >= low) & (estimates <= high), axis=1)
        if np.any(accepted):
            for g, group in enumerate(groups):
                difficulty = np.array(group.parameters["difficulty"], copy=True)
                slope_values = np.array(group.parameters["discrimination"], copy=True)
                difficulty[items[accepted]] = new_difficulty[g, accepted]
                slope_values[items[accepted]] = new_slopes[accepted].reshape(
                    (-1, *slope_values.shape[1:])
                )
                group.set_parameters(discrimination=slope_values, difficulty=difficulty)
        failed = np.zeros(group_counts.shape[1], dtype=np.bool_)
        failed[items[~accepted]] = True
        return failed

    @staticmethod
    def _validate_mean_order(
        mean_order: Sequence[int] | None, model: MultigroupModel
    ) -> tuple[int, ...] | None:
        if mean_order is None:
            return None
        if model.n_factors != 1:
            raise ValueError("ordered group means require a unidimensional model")
        if isinstance(mean_order, (str, bytes)):
            raise ValueError("mean_order must be a permutation of all group indices")
        try:
            order = tuple(mean_order)
        except TypeError as error:
            raise ValueError(
                "mean_order must be a permutation of all group indices"
            ) from error
        if (
            len(order) != model.n_groups
            or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                for index in order
            )
            or set(order) != set(range(model.n_groups))
        ):
            raise ValueError("mean_order must be a permutation of all group indices")
        return tuple(int(index) for index in order)

    def _update_ordered_latent_density(
        self,
        posterior_weights: list[NDArray[np.float64]],
        order: tuple[int, ...],
    ) -> None:
        """Maximize the discretized Gaussian prior Q under mean inequalities.

        Prior masses include the quadrature reference-density correction and
        their normalizer. Gradients compare posterior and candidate prior
        moments, so the update optimizes the same grid used by the E-step.
        The reference mean and covariance never enter the optimization vector.
        """
        density = self._latent_density
        nodes = self._quadrature.nodes[:, 0]
        log_base = np.log(self._quadrature.weights) + 0.5 * nodes**2
        free = [g for g in range(density.n_groups) if g != density.reference_group]
        n_free = len(free)
        covariance_slots = {
            group: n_free + slot
            for slot, group in enumerate(
                group for group in free if density.distributions[group].estimate_cov
            )
        }
        counts = [weights.sum(axis=0) for weights in posterior_weights]
        x0 = np.array(
            [density.distributions[g].mean[0] for g in free]
            + [np.log(density.distributions[g].cov[0, 0]) for g in covariance_slots]
        )

        def objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            loss = 0.0
            gradient = np.zeros_like(params)
            for slot, group in enumerate(free):
                centered = nodes - params[slot]
                variance = (
                    np.exp(params[covariance_slots[group]])
                    if group in covariance_slots
                    else density.distributions[group].cov[0, 0]
                )
                log_mass = log_base - centered**2 / (2 * variance)
                log_mass -= logsumexp(log_mass)
                mass = np.exp(log_mass)
                n_k = counts[group]
                total = n_k.sum()
                loss -= float(n_k @ log_mass)
                gradient[slot] = -(n_k @ nodes - total * (mass @ nodes)) / variance
                if group in covariance_slots:
                    gradient[covariance_slots[group]] = -(
                        n_k @ (centered**2) - total * (mass @ (centered**2))
                    ) / (2 * variance)
            return loss, gradient

        order_matrix = np.zeros((len(order) - 1, len(x0)))
        for row, (left, right) in enumerate(zip(order[:-1], order[1:], strict=True)):
            if left in free:
                order_matrix[row, free.index(left)] = -1
            if right in free:
                order_matrix[row, free.index(right)] = 1
        result = minimize(
            objective,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=[(None, None)] * n_free + [(-14.0, 14.0)] * len(covariance_slots),
            constraints={
                "type": "ineq",
                "fun": lambda params: order_matrix @ params,
                "jac": lambda params: order_matrix,
            },
            options={"maxiter": max(100, self.item_optim_maxiter), "ftol": 1e-9},
        )
        # SLSQP allows small feasibility tolerances, and a line-search
        # failure next to tied means can leave slightly larger ones. Remove
        # only such numerical violations while retaining the reference mean
        # exactly at zero.
        if (
            np.all(np.isfinite(result.x))
            and np.min(order_matrix @ result.x) >= -_ORDER_SLACK
        ):
            means = np.zeros(density.n_groups)
            means[free] = result.x[:n_free]
            ordered_means = np.maximum.accumulate(means[list(order)])
            ordered_means -= ordered_means[order.index(density.reference_group)]
            means[list(order)] = ordered_means
            result.x[:n_free] = means[free]
        initial_loss = objective(x0)[0]
        final_loss = objective(result.x)[0]
        # SLSQP can stop on a line-search failure next to active order
        # constraints; a feasible point that does not raise the loss is
        # still a valid generalized EM step.
        if (
            not np.all(np.isfinite(result.x))
            or not np.isfinite(final_loss)
            or np.min(order_matrix @ result.x) < -1e-8
            or final_loss > initial_loss + 1e-7 * max(1.0, abs(initial_loss))
        ):
            raise RuntimeError(
                f"ordered latent-density optimization failed: {result.message}"
            )
        for slot, group in enumerate(free):
            density.set_group_distribution(
                group,
                mean=np.array([result.x[slot]]),
                cov=(
                    np.array([[np.exp(result.x[covariance_slots[group]])]])
                    if group in covariance_slots
                    else None
                ),
            )

    def _optimize_item(
        self,
        model: MultigroupModel,
        item_idx: int,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
        *,
        masks: Sequence[Mapping[str, NDArray[np.bool_]]] | None = None,
        counts: Sequence[NDArray[np.float64]] | None = None,
    ) -> None:
        """Optimize parameters for a single item across all groups.

        Built-in binary and GRM, GPCM, PCM and NRM items update every free
        coordinate in one block with analytic gradients (see
        ``_optimize_binary_block`` and ``_optimize_polytomous_block``). Other
        items update one parameter name at a time, jointly across groups when
        it is shared. ``masks`` supplies
        each group's effective free-parameter masks and ``counts`` each
        group's expected category counts of the item when already computed.
        """
        groups = model.group_models
        if counts is None:
            counts = [
                self._expected_item_counts(group, item_idx, data, posterior)
                for group, data, posterior in zip(
                    groups, responses, posterior_weights, strict=True
                )
            ]
        if masks is None:
            masks = [
                model.effective_free_parameter_masks(g) for g in range(model.n_groups)
            ]
        item_masks = {
            name: [group_masks[name][item_idx].ravel() for group_masks in masks]
            for name in model.parameter_names
        }
        shared = {
            name
            for name in item_masks
            if model.is_item_parameter_shared(name, item_idx)
        }
        if self._optimize_binary_block(
            groups, item_idx, item_masks, shared, counts, quad_points
        ) or self._optimize_polytomous_block(
            groups, item_idx, item_masks, shared, counts, quad_points
        ):
            return
        for param_name, name_masks in item_masks.items():
            if param_name in shared:
                self._optimize_parameter_block(
                    groups, name_masks, counts, item_idx, param_name, quad_points
                )
                continue
            for group, mask, group_counts in zip(
                groups, name_masks, counts, strict=True
            ):
                self._optimize_parameter_block(
                    [group], [mask], [group_counts], item_idx, param_name, quad_points
                )

    @staticmethod
    def _expected_item_counts(
        group_model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reduce category indicators without copying posterior rows."""
        n_categories = (
            group_model._n_categories[item_idx] if group_model.is_polytomous else 2
        )
        return EMFitContext(responses).expected_category_counts(
            item_idx, n_categories, posterior
        )

    @staticmethod
    def _all_item_counts(
        group_model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior: NDArray[np.float64],
    ) -> Sequence[NDArray[np.float64]]:
        """Return every item's ``_expected_item_counts`` with blocked products.

        Binary counts form one array of shape ``(n_items, n_points, 2)``.
        """
        context = EMFitContext(responses)
        if group_model.is_polytomous:
            return context.category_counts(group_model._n_categories, posterior)
        correct, observed = context.expected_counts(posterior)
        return np.stack((observed - correct, correct), axis=-1)

    def _optimize_binary_block(
        self,
        groups: Sequence[BaseItemModel],
        item_idx: int,
        masks: Mapping[str, Sequence[NDArray[np.bool_]]],
        shared: Collection[str],
        counts: Sequence[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> bool:
        """Maximize a built-in binary item's free coordinates with one solver.

        ``masks`` maps parameter names to each group's free coordinates of the
        item; omitted names stay fixed. A free coordinate of a ``shared`` name
        is one variable for every group that frees it, and other coordinates
        are one variable per group. Without shared variables each group is
        solved on its own, and groups whose item curves and variables
        coincide pool their counts. A solve is kept only when it improves the
        expected log-likelihood, as generalized EM requires.

        Returns
        -------
        bool
            ``False``, with every model unchanged, when the item needs the
            per-parameter path: polytomous, custom, or restricted models.
        """
        first = groups[0]
        # Pooled groups share one kernel, so every group needs the built-in
        # curve, not only the group whose kernel is prepared.
        if first.is_polytomous or any(
            type(group) is not type(first) or not uses_builtin_model_hooks(group)
            for group in groups
        ):
            return False
        fixed_slope = first.model_name == "1PL"
        if fixed_slope and np.any(masks.get("discrimination", False)):
            return False
        parameters = [group.parameters for group in groups]
        names = [
            name
            for name in parameters[0]
            if not (fixed_slope and name == "discrimination")
        ]
        rows = [
            np.concatenate([np.ravel(values[name][item_idx]) for name in names])
            for values in parameters
        ]
        observed = [bool(np.any(group_counts)) for group_counts in counts]
        lower: list[float] = []
        upper: list[float] = []
        spans: dict[str, slice] = {}
        positions: list[list[int]] = [[] for _ in groups]
        variables: list[list[int]] = [[] for _ in groups]
        start: list[float] = []
        box: list[tuple[float, float]] = []
        coupled = False
        for name in names:
            offset = len(lower)
            size = np.size(parameters[0][name][item_idx])
            spans[name] = slice(offset, offset + size)
            bound = self._parameter_bound(first, name)
            lower.extend([bound[0]] * size)
            upper.extend([bound[1]] * size)
            name_masks = masks.get(name)
            if name_masks is None:
                continue
            if name in shared:
                for coordinate in np.flatnonzero(np.logical_or.reduce(name_masks)):
                    coupled = True
                    for g, mask in enumerate(name_masks):
                        if mask[coordinate]:
                            positions[g].append(offset + coordinate)
                            variables[g].append(len(start))
                            value = rows[g][offset + coordinate]
                    start.append(value)
                    box.append(bound)
                continue
            for g, mask in enumerate(name_masks):
                if not observed[g]:
                    continue
                for coordinate in np.flatnonzero(mask):
                    positions[g].append(offset + coordinate)
                    variables[g].append(len(start))
                    start.append(rows[g][offset + coordinate])
                    box.append(bound)

        members = [g for g in range(len(groups)) if variables[g]]
        problems = [members] if coupled else [[g] for g in members]
        initial = np.asarray(start, dtype=np.float64)
        solves = []
        for problem in problems:
            terms_groups = [g for g in problem if observed[g]]
            if not terms_groups:
                continue
            used = np.unique(np.concatenate([variables[g] for g in problem]))
            local = {g: np.searchsorted(used, variables[g]) for g in problem}
            reference = terms_groups[0]
            pooled = all(
                positions[g] == positions[reference]
                and variables[g] == variables[reference]
                and all(
                    np.array_equal(values[item_idx], parameters[g][name][item_idx])
                    for name, values in parameters[reference].items()
                )
                for g in terms_groups[1:]
            )
            terms: list[_BinaryTerm] = []
            for term in [terms_groups] if pooled else [[g] for g in terms_groups]:
                g = term[0]
                index = np.asarray(positions[g], dtype=np.intp)
                trial = rows[g].copy()
                trial[index] = initial[variables[g]]
                # The kernel skips overflow guards when every evaluated vector
                # lies in its box, so cover fixed values and the start too.
                kernel_box = list(
                    zip(
                        np.minimum(lower, np.minimum(rows[g], trial)),
                        np.maximum(upper, np.maximum(rows[g], trial)),
                        strict=True,
                    )
                )
                term_counts = np.sum([counts[h] for h in term], axis=0)
                kernel = prepare_dichotomous_objective(
                    groups[g],
                    item_idx,
                    quad_points,
                    term_counts.sum(axis=1),
                    term_counts[:, 1],
                    self.prob_epsilon,
                    kernel_box,
                )
                if kernel is None:
                    return False
                terms.append((kernel, rows[g], index, local[g]))
            solves.append((problem, used, local, terms))

        for problem, used, local, terms in solves:
            solution = self._minimize_item_terms(
                terms, initial[used], [box[index] for index in used]
            )
            if solution is None:
                continue
            for g in problem:
                row = rows[g].copy()
                row[positions[g]] = solution[local[g]]
                for name, span in spans.items():
                    if not np.array_equal(row[span], rows[g][span]):
                        self._set_param(groups[g], item_idx, name, row[span])
        return True

    def _minimize_item_terms(
        self,
        terms: list[_BinaryTerm],
        start: NDArray[np.float64],
        box: list[tuple[float, float]],
        order: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> NDArray[np.float64] | None:
        """Minimize summed kernel losses; keep only an improving solution.

        ``order`` holds a matrix ``A`` and constants ``c`` of graded threshold
        gaps ``A @ x + c``, which must stay at least ``THRESHOLD_GAP``; SLSQP
        then replaces L-BFGS-B. A solution that leaves a gap nonpositive is
        rejected.
        """
        from mirt.estimation._graded_order import THRESHOLD_GAP

        def objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            loss = 0.0
            gradient = np.zeros_like(params)
            for kernel, base, index, variables in terms:
                trial = base.copy()
                trial[index] = params[variables]
                value, full_gradient = kernel(trial)
                loss += value
                gradient[variables] += full_gradient[index]
            return loss, gradient

        initial_loss = objective(start)[0]
        options = {
            "maxiter": self.item_optim_maxiter,
            "ftol": min(self.item_optim_ftol, _JOINT_ITEM_FTOL),
        }
        result = minimize(
            objective,
            x0=start,
            jac=True,
            method="L-BFGS-B",
            bounds=box,
            options=options,
        )
        # Ordered graded thresholds are an interior optimum unless a
        # category is nearly empty, so constraints are added only when the
        # free solve crosses them.
        if order is not None and (
            not np.all(np.isfinite(result.x))
            or np.min(order[0] @ result.x + order[1]) < THRESHOLD_GAP
        ):
            matrix, constants = order
            result = minimize(
                objective,
                x0=start,
                jac=True,
                method="SLSQP",
                bounds=box,
                constraints={
                    "type": "ineq",
                    "fun": lambda params: matrix @ params + constants - THRESHOLD_GAP,
                    "jac": lambda params: matrix,
                },
                options=options,
            )
        if not np.all(np.isfinite(result.x)):
            return None
        if order is not None and np.min(order[0] @ result.x + order[1]) <= 0.0:
            return None
        final_loss = objective(result.x)[0]
        if not np.isfinite(final_loss) or final_loss > initial_loss + 1e-10 * max(
            1.0, abs(initial_loss)
        ):
            return None
        return np.asarray(result.x, dtype=np.float64)

    def _optimize_polytomous_block(
        self,
        groups: Sequence[BaseItemModel],
        item_idx: int,
        masks: Mapping[str, Sequence[NDArray[np.bool_]]],
        shared: Collection[str],
        counts: Sequence[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> bool:
        """Maximize a built-in polytomous item's free coordinates with one solver.

        Variables are formed as in :meth:`_optimize_binary_block`: a free
        coordinate of a ``shared`` name is one variable for every group that
        frees it, other coordinates one variable per group that observed the
        item, and groups whose item curves and variables coincide pool their
        counts. Each group contributes the analytic GRM, GPCM, PCM or NRM
        expected log-likelihood and gradient of
        :func:`~mirt.estimation._polytomous_objective.prepare_polytomous_objective`
        in that family's free layout, as in single-group EM; coordinates that
        ``set_free_parameter_masks`` fixes, such as the zero loadings of a
        confirmatory model, are constants there. Graded thresholds stay
        ordered in every group by linear constraints. A solve is kept only
        when it improves the expected log-likelihood.

        Returns
        -------
        bool
            ``False``, with every model unchanged, when the item needs the
            per-parameter path: binary, custom or other polytomous families,
            a slope shared by all items, or a graded solve that cannot keep
            its thresholds ordered.
        """
        import copy

        from mirt.estimation._polytomous_objective import prepare_polytomous_objective
        from mirt.models.polytomous import GradedResponseModel

        first = groups[0]
        if (
            not first.is_polytomous
            or any(
                type(group) is not type(first) or not uses_builtin_model_hooks(group)
                for group in groups
            )
            or not all(first._item_indexed(name) for name in first.parameters)
        ):
            return False
        parameters = [group.parameters for group in groups]
        names = list(parameters[0])
        # Objectives use the family layout, without user restrictions.
        views = []
        for group in groups:
            view = copy.copy(group)
            view._free_parameter_restrictions = {}
            views.append(view)
        # Each group's item coordinates in the objectives' free layout: per
        # name, its offset there and the flat item indices it holds.
        layouts: list[dict[str, tuple[int, NDArray[np.intp]]]] = []
        rows: list[NDArray[np.float64]] = []
        for view, values in zip(views, parameters, strict=True):
            layout: dict[str, tuple[int, NDArray[np.intp]]] = {}
            chunks = []
            offset = 0
            free_masks = view.free_parameter_masks
            for name in names:
                flat = np.flatnonzero(np.ravel(free_masks[name][item_idx]))
                layout[name] = (offset, flat)
                chunks.append(np.ravel(values[name][item_idx])[flat])
                offset += flat.size
            layouts.append(layout)
            rows.append(
                np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0)
            )

        observed = [bool(np.any(group_counts)) for group_counts in counts]
        positions: list[list[int]] = [[] for _ in groups]
        variables: list[list[int]] = [[] for _ in groups]
        start: list[float] = []
        box: list[tuple[float, float]] = []
        coupled = False

        def position(g: int, name: str, coordinate: int) -> int | None:
            offset, flat = layouts[g][name]
            slot = int(np.searchsorted(flat, coordinate))
            if slot == flat.size or flat[slot] != coordinate:
                return None
            return offset + slot

        for name in names:
            name_masks = masks.get(name)
            if name_masks is None:
                continue
            bound = self._parameter_bound(first, name)
            if name in shared:
                for coordinate in np.flatnonzero(np.logical_or.reduce(name_masks)):
                    coupled = True
                    value = 0.0
                    for g, mask in enumerate(name_masks):
                        if not mask[coordinate]:
                            continue
                        index = position(g, name, int(coordinate))
                        if index is None:
                            return False
                        positions[g].append(index)
                        variables[g].append(len(start))
                        value = rows[g][index]
                    start.append(value)
                    box.append(bound)
                continue
            for g, mask in enumerate(name_masks):
                if not observed[g]:
                    continue
                for coordinate in np.flatnonzero(mask):
                    index = position(g, name, int(coordinate))
                    if index is None:
                        return False
                    positions[g].append(index)
                    variables[g].append(len(start))
                    start.append(rows[g][index])
                    box.append(bound)

        members = [g for g in range(len(groups)) if variables[g]]
        problems = [members] if coupled else [[g] for g in members]
        initial = np.asarray(start, dtype=np.float64)
        graded = isinstance(first, GradedResponseModel)
        solves = []
        for problem in problems:
            terms_groups = [g for g in problem if observed[g]]
            if not terms_groups:
                continue
            used = np.unique(np.concatenate([variables[g] for g in problem]))
            local = {g: np.searchsorted(used, variables[g]) for g in problem}
            reference = terms_groups[0]
            pooled = all(
                positions[g] == positions[reference]
                and variables[g] == variables[reference]
                and all(
                    np.array_equal(values[item_idx], parameters[g][name][item_idx])
                    for name, values in parameters[reference].items()
                )
                for g in terms_groups[1:]
            )
            terms: list[_BinaryTerm] = []
            for term in [terms_groups] if pooled else [[g] for g in terms_groups]:
                g = term[0]
                kernel = prepare_polytomous_objective(
                    views[g],
                    item_idx,
                    quad_points,
                    np.sum([counts[h] for h in term], axis=0),
                    self.prob_epsilon,
                )
                if kernel is None:
                    return False
                terms.append(
                    (kernel, rows[g], np.asarray(positions[g], np.intp), local[g])
                )
            order = (
                self._threshold_gaps(
                    groups, item_idx, problem, layouts, rows, positions, local, used
                )
                if graded
                else None
            )
            solves.append((problem, used, local, terms, order))

        solutions = []
        for problem, used, local, terms, order in solves:
            solution = self._minimize_item_terms(
                terms, initial[used], [box[index] for index in used], order
            )
            if solution is None and order is not None:
                return False
            solutions.append(solution)
        for (problem, _, local, _, _), solution in zip(solves, solutions, strict=True):
            if solution is None:
                continue
            for g in problem:
                row = rows[g].copy()
                row[positions[g]] = solution[local[g]]
                for name in names:
                    offset, flat = layouts[g][name]
                    span = slice(offset, offset + flat.size)
                    if np.array_equal(row[span], rows[g][span]):
                        continue
                    values = np.ravel(parameters[g][name][item_idx]).copy()
                    values[flat] = row[span]
                    self._set_param(groups[g], item_idx, name, values)
        return True

    @staticmethod
    def _threshold_gaps(
        groups: Sequence[BaseItemModel],
        item_idx: int,
        problem: Sequence[int],
        layouts: Sequence[Mapping[str, tuple[int, NDArray[np.intp]]]],
        rows: Sequence[NDArray[np.float64]],
        positions: Sequence[Sequence[int]],
        local: Mapping[int, NDArray[np.intp]],
        used: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return adjacent graded threshold gaps as ``A @ x + c`` over a solve.

        Every group of ``problem`` contributes the gaps of its thresholds,
        with fixed thresholds as constants; gaps without a variable and
        duplicates, such as those of shared thresholds, are dropped.
        """
        gaps: dict[tuple[float, ...], None] = {}
        for g in problem:
            offset, flat = layouts[g]["thresholds"]
            n_thresholds = int(groups[g]._n_categories[item_idx]) - 1
            slots = {
                index: int(variable)
                for index, variable in zip(positions[g], local[g], strict=True)
            }
            for k in range(min(n_thresholds, flat.size) - 1):
                row = np.zeros(used.size + 1)
                for index, sign in ((offset + k, -1.0), (offset + k + 1, 1.0)):
                    if index in slots:
                        row[slots[index]] += sign
                    else:
                        row[-1] += sign * rows[g][index]
                if np.any(row[:-1]):
                    gaps[tuple(row)] = None
        if not gaps:
            return None
        matrix = np.array(list(gaps))
        return matrix[:, :-1], matrix[:, -1]

    def _optimize_parameter_block(
        self,
        models: list[BaseItemModel],
        masks: list[NDArray[np.bool_]],
        counts: list[NDArray[np.float64]],
        item_idx: int,
        param_name: str,
        quad_points: NDArray[np.float64],
    ) -> None:
        """Maximize a free block and restore state after rejected trials.

        Trials go through the models' public probability and setter hooks.
        An incomplete optimizer step is accepted only when it improves the
        actual expected likelihood, as required by generalized EM.
        """
        if not np.any(masks) or not any(
            np.any(group_counts) for group_counts in counts
        ):
            return
        originals = [
            group.parameters[param_name][item_idx].ravel().copy() for group in models
        ]
        active = np.logical_or.reduce(masks)
        current, bounds = self._get_param_and_bounds(models[0], item_idx, param_name)
        for original, mask in zip(originals, masks, strict=True):
            current[mask] = original[mask]
        slots = np.flatnonzero(active)
        evaluations = list(zip(models, originals, masks, counts, strict=True))
        if len(models) > 1 and all(
            type(group) is type(models[0])
            and uses_builtin_model_hooks(group)
            and np.array_equal(mask, masks[0])
            and all(
                np.array_equal(group.parameters[name][item_idx], value[item_idx])
                for name, value in models[0].parameters.items()
            )
            for group, mask in zip(models, masks, strict=True)
        ):
            # Identical curves share sufficient statistics; retain separate
            # likelihood terms whenever another group parameter or hook differs.
            evaluations = [(models[0], originals[0], masks[0], np.sum(counts, axis=0))]

        def evaluate(params: NDArray[np.float64]) -> float:
            loss = 0.0
            for group, original, mask, group_counts in evaluations:
                row = original.copy()
                row[slots[mask[slots]]] = params[mask[slots]]
                self._set_param(group, item_idx, param_name, row)
                probs = group.probability(quad_points, item_idx)
                if not group.is_polytomous:
                    probs = np.column_stack((1 - probs, probs))
                if not np.all(np.isfinite(probs)):
                    return np.inf
                loss -= float(
                    np.sum(
                        group_counts
                        * np.log(
                            np.clip(probs, self.prob_epsilon, 1 - self.prob_epsilon)
                        )
                    )
                )
            return loss

        accepted = False
        try:
            initial_loss = evaluate(current[active])
            ordered_thresholds = param_name == "thresholds" and all(
                group.model_name == "GRM" for group in models
            )
            constraint_matrix = []
            constraint_constants = []
            if ordered_thresholds:
                for group, original, mask in zip(models, originals, masks, strict=True):
                    for first in range(group._n_categories[item_idx] - 2):
                        row = np.zeros(slots.size)
                        constant = original[first + 1] - original[first]
                        for index, sign in ((first, -1), (first + 1, 1)):
                            if mask[index]:
                                column = int(np.flatnonzero(slots == index)[0])
                                row[column] += sign
                                constant -= sign * original[index]
                        if np.any(row):
                            constant -= 1e-7
                        constraint_matrix.append(row)
                        constraint_constants.append(constant)
                ordered_thresholds = bool(constraint_matrix)
            matrix = np.asarray(constraint_matrix)
            constants = np.asarray(constraint_constants)
            result = minimize(
                evaluate,
                x0=current[active],
                method="SLSQP" if ordered_thresholds else "L-BFGS-B",
                bounds=[bounds[index] for index in slots],
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda params: matrix @ params + constants,
                        "jac": lambda params: matrix,
                    }
                    if ordered_thresholds
                    else ()
                ),
                options={
                    "maxiter": self.item_optim_maxiter,
                    "ftol": (
                        min(self.item_optim_ftol, _JOINT_ITEM_FTOL)
                        if self._precise_m_steps
                        else self.item_optim_ftol
                    ),
                },
            )
            if not np.all(np.isfinite(result.x)):
                return
            if ordered_thresholds:
                if np.min(matrix @ result.x + constants) < -1e-9:
                    return
            final_loss = evaluate(result.x)
            if not np.isfinite(final_loss) or final_loss > initial_loss + 1e-10 * max(
                1.0, abs(initial_loss)
            ):
                return
            for group, original, mask in zip(models, originals, masks, strict=True):
                row = original.copy()
                row[slots[mask[slots]]] = result.x[mask[slots]]
                self._set_param(group, item_idx, param_name, row)
            accepted = True
        finally:
            if not accepted:
                for group, original in zip(models, originals, strict=True):
                    self._set_param(group, item_idx, param_name, original)

    @staticmethod
    def _parameter_bound(model: BaseItemModel, param_name: str) -> tuple[float, float]:
        """Return the optimizer box for every coordinate of a parameter.

        It is the single-group EM box (``estimation.base._parameter_bounds``),
        except that multidimensional binary slopes may be negative. Their
        factors are identified only up to rotation, and a positive lower
        bound would exclude rotations with zero or negative loadings.
        """
        if (
            param_name == "discrimination"
            and model.n_factors > 1
            and not model.is_polytomous
        ):
            return (-5.0, 5.0)
        return _parameter_bounds(model, param_name)

    def _get_param_and_bounds(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current parameter value and bounds for optimization."""
        current = np.ravel(model.parameters[param_name][item_idx]).copy()
        return current, [self._parameter_bound(model, param_name)] * current.size

    def _set_param(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        value: NDArray[np.float64],
    ) -> None:
        """Set parameter value for a specific item."""
        values = model._parameters[param_name].copy()
        row_shape = values[item_idx].shape
        values[item_idx] = np.asarray(value, dtype=np.float64).reshape(row_shape)
        canonical = model._canonical_parameter_values(param_name, values)
        row = np.asarray(canonical[item_idx])
        model.set_item_parameter(
            item_idx, param_name, float(row) if row.ndim == 0 else row
        )

    @property
    def convergence_history(self) -> list[float]:
        """Log-likelihood history across iterations."""
        return self._convergence_history.copy()
