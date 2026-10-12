"""Bock-Lieberman (BL) estimation for IRT models.

The BL method is a direct marginal maximum likelihood approach that
jointly optimizes all parameters without the iterative E-M structure.
It can be more efficient for small models but scales less well than EM.

References
----------
Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n
    dichotomously scored items. Psychometrika, 35, 179-197.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt.estimation._bl_objective import prepare_bl_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.base import (
    BaseEstimator,
    _family_tied_parameters,
    _initialize_free_parameters,
    _logistic_bounds,
)
from mirt.estimation.quadrature import (
    GaussHermiteQuadrature,
    QuadratureRule,
    RectangularQuadrature,
    build_quadrature,
    resolve_quadpts,
    validate_quadpts,
    validate_quadrature_rule,
)
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.estimation._bl_objective import PreparedBLObjective
    from mirt.estimation._louis_information import LouisInformation
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Central-difference steps for the marginal Hessian: an analytic gradient
# tolerates a smaller step than second differences of the likelihood.
_GRADIENT_STEP = 1e-5
_LIKELIHOOD_STEP = 1e-4
# scipy.optimize.minimize methods that cannot enforce the parameter boxes,
# and those that BL gives analytic marginal gradients.
_UNBOUNDED_METHODS = (
    "bfgs",
    "cg",
    "newton-cg",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
)
_GRADIENT_METHODS = ("l-bfgs-b", "tnc", "slsqp")
# Projected Newton steps on the exact information that may finish a
# quasi-Newton solution, and the step halvings each may take.
_NEWTON_STEPS = 20
_NEWTON_HALVINGS = 20
# Optimizer boxes by stored parameter name; other parameters get (-10, 10).
_BL_BOUNDS = {
    "discrimination": (0.1, 5.0),
    "slopes": (0.1, 5.0),
    "difficulty": (-6.0, 6.0),
    "intercepts": (-6.0, 6.0),
    "thresholds": (-6.0, 6.0),
    "steps": (-6.0, 6.0),
    "guessing": (0.0, 0.5),
    "upper": (0.5, 1.0),
    "asymmetry": (0.1, 5.0),
}


class BLEstimator(BaseEstimator):
    """Bock-Lieberman marginal maximum likelihood estimator.

    This estimator uses direct numerical optimization of the marginal
    likelihood, jointly estimating all item parameters simultaneously.
    Unlike EM, there is no alternation between E and M steps.

    Parameters
    ----------
    n_quadpts : int, optional
        Quadrature points per latent dimension, at least 5. The default
        depends on the number of factors, as for
        :class:`~mirt.estimation.em.EMEstimator`.
    max_iter : int
        Maximum number of optimization iterations (function evaluations
        for TNC).
    tol : float, default=1e-6
        Convergence tolerance in marginal log-likelihood units, as for
        :class:`~mirt.estimation.em.EMEstimator`: the optimizer stops once
        an iteration improves the log-likelihood by less than ``tol``.
        L-BFGS-B and Powell compare relative changes, so their tolerance
        is ``tol`` divided by the starting objective. With analytic
        gradients and the exact information (see Notes), projected Newton
        steps then finish the solution, and ``FitResult.converged`` reports
        whether the log-likelihood gain predicted for a further Newton
        step, ``g' H^-1 g / 2`` over the coordinates off their bounds, is
        at most ``tol``. Otherwise it reports the optimizer's success.
    verbose : bool
        Print optimization progress.
    method : str, default="L-BFGS-B"
        Bounded optimization method of :func:`scipy.optimize.minimize`.
        Built-in 1PL–4PL, GRM, GPCM, PCM, NRM, MIRT, and bifactor models
        use analytic marginal gradients with L-BFGS-B, TNC, or SLSQP.
        Custom models and likelihoods, and the derivative-free bounded
        methods such as Powell, retain numerical optimization. Methods that
        cannot enforce the parameter boxes, such as BFGS and CG, raise
        ``MirtValidationError``.
    quadrature : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule, as for :class:`~mirt.estimation.em.EMEstimator`.

    Notes
    -----
    Standard errors invert the full Hessian of the marginal log-likelihood.
    Built-in item models whose likelihood is a product of item curves
    (1PL-4PL, GRM, GPCM, PCM, NRM, RSM, GRSM, multidimensional, bifactor and
    the other built-in families such as 5PL, log-log, nested-logit,
    sequential, unfolding and zero-inflated models), and custom item models,
    compute it exactly by Louis's identity, which differences at most each
    item's own curve. Other models with analytic gradients difference the
    gradient, and the remaining models difference the likelihood over
    every pair of parameters, which takes 2P^2 + 1 likelihood evaluations
    for P parameters. Coordinates on an optimizer bound are held fixed and
    receive ``NaN`` standard errors. The covariance is stored in
    ``FitResult.vcov``.

    The BL method directly maximizes:

        L(xi) = prod_i integral P(x_i | theta)^{x_i} Q(x_i | theta)^{1-x_i} g(theta) dtheta

    using numerical quadrature to approximate the integral.

    For dichotomous 2PL models, this reduces to optimizing 2*n_items parameters
    simultaneously. The method can be less stable than EM for complex models
    but may converge faster for simple cases.
    """

    def __init__(
        self,
        n_quadpts: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        method: str = "L-BFGS-B",
        quadrature: QuadratureRule = "auto",
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        self.n_quadpts = None if n_quadpts is None else validate_quadpts(n_quadpts)
        self.quadrature = validate_quadrature_rule(quadrature)
        if isinstance(method, str) and method.lower() in _UNBOUNDED_METHODS:
            # Every coordinate has a box, which these methods would ignore.
            raise MirtValidationError(
                f"method={method!r} cannot enforce the parameter bounds of "
                "BLEstimator; use 'L-BFGS-B', 'TNC' or 'SLSQP'",
                parameter="method",
                value=method,
                expected="'L-BFGS-B', 'TNC', 'SLSQP' or another bounded method",
            )
        self.method = method
        self._quadrature: GaussHermiteQuadrature | RectangularQuadrature | None = None
        # The exact information at the latest point that needed it.
        self._information_cache: (
            tuple[NDArray[np.int_], NDArray[np.float64], LouisInformation] | None
        ) = None

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> FitResult:
        """Fit IRT model using Bock-Lieberman estimation.

        Parameters
        ----------
        model : BaseItemModel
            The IRT model to fit.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.

        Returns
        -------
        FitResult
            Fitted model with parameter estimates and standard errors.
        """
        responses = self._validate_responses(responses, model.n_items, model)
        with EMFitContext(responses) as context:
            return self._fit_prepared(model, responses, context)

    def _fit_prepared(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        context: EMFitContext,
    ) -> FitResult:
        from mirt.results.fit_result import FitResult

        n_persons = responses.shape[0]

        tied = _family_tied_parameters(model)
        if tied:
            # The joint vector would give every item its own value.
            raise MirtModelError(
                f"BLEstimator cannot estimate {', '.join(tied)}, which "
                f"{model.model_name} keeps equal across items; use EMEstimator, "
                "or hold it fixed with set_free_parameter_masks",
                model_type=model.model_name,
            )
        self._quadrature = build_quadrature(
            resolve_quadpts(self.n_quadpts, model.n_factors),
            model.n_factors,
            self.quadrature,
        )

        if not model._is_fitted:
            _initialize_free_parameters(model)

        initial_params, bounds, param_structure = self._flatten_parameters(model)
        self._information_cache = None
        prepared = None
        if (
            type(self) is BLEstimator
            and isinstance(self.method, str)
            and self.method.lower() in _GRADIENT_METHODS
            and not any(
                name in vars(self)
                for name in (
                    "_flatten_parameters",
                    "_unflatten_parameters",
                    "_compute_marginal_log_likelihood",
                )
            )
        ):
            prepared = prepare_bl_objective(
                model,
                context,
                self._quadrature.nodes,
                np.log(self._quadrature.weights),
                param_structure,
                bounds,
                self._unflatten_parameters,
            )

        def neg_log_likelihood(params: NDArray[np.float64]) -> float:
            self._unflatten_parameters(model, params, param_structure)
            ll = self._compute_marginal_log_likelihood(model, responses)
            return -ll

        def _verbose_callback(x: NDArray[np.float64]) -> None:
            value = neg_log_likelihood(x) if prepared is None else prepared.value(x)
            print(f"LL = {-value:.4f}")

        callback = _verbose_callback if self.verbose else None

        if initial_params.size:
            try:
                start_value = (
                    neg_log_likelihood(initial_params)
                    if prepared is None
                    else prepared.value(initial_params)
                )
                result = minimize(
                    neg_log_likelihood if prepared is None else prepared,
                    x0=initial_params,
                    jac=prepared is not None,
                    method=self.method,
                    bounds=bounds,
                    options=self._optimizer_options(start_value),
                    callback=callback,
                )
            except BaseException:
                self._unflatten_parameters(model, initial_params, param_structure)
                raise
        else:
            result = SimpleNamespace(
                x=initial_params,
                fun=neg_log_likelihood(initial_params),
                nit=0,
                success=True,
            )

        params, value = np.asarray(result.x, dtype=np.float64), float(result.fun)
        n_iterations = result.nit if hasattr(result, "nit") else 0
        converged = bool(result.success)
        # Newton steps finish a solution the optimizer accepted; they do not
        # extend a run that ``max_iter`` cut short.
        if converged and prepared is not None and params.size:
            finished = self._finish_with_newton(
                model, responses, prepared, params, bounds, param_structure
            )
            if finished is not None:
                params, value, converged, n_steps = finished
                n_iterations += n_steps

        self._unflatten_parameters(model, params, param_structure)
        model._is_fitted = True

        final_ll = -value

        covariance = None
        if (
            "_compute_standard_errors" in vars(self)
            or type(self)._compute_standard_errors
            is not BLEstimator._compute_standard_errors
        ):
            se = self._compute_standard_errors(
                model, responses, params, param_structure
            )
        else:
            covariance = self._parameter_covariance(
                model,
                responses,
                params,
                param_structure,
                gradient=prepared,
                bounds=bounds,
            )
            se = self._unflatten_standard_errors(model, covariance, param_structure)
            if not _uses_free_layout(model, param_structure):
                covariance = None

        n_params = len(params)
        if _uses_free_layout(model, param_structure):
            # Leaves out the rotational indeterminacy of exploratory factors.
            n_params = model.n_parameters
        aic = -2 * final_ll + 2 * n_params
        bic = -2 * final_ll + np.log(n_persons) * n_params

        return FitResult(
            model=model,
            log_likelihood=final_ll,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors=se,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
            se_method=None if covariance is None else "hessian",
            vcov=covariance,
        )

    def _compute_marginal_log_likelihood(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute marginal log-likelihood using quadrature."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights

        if hasattr(model, "log_likelihood_batch"):
            log_likelihoods = model.log_likelihood_batch(responses, quad_points)
        else:
            n_persons = responses.shape[0]
            n_quad = len(quad_weights)
            log_likelihoods = np.zeros((n_persons, n_quad))
            for q in range(n_quad):
                theta_q = quad_points[q : q + 1]
                log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)

        log_weights = np.log(quad_weights)[None, :]

        log_marginal = logsumexp(log_likelihoods + log_weights, axis=1)

        return float(np.sum(log_marginal))

    def _flatten_parameters(
        self,
        model: BaseItemModel,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]], dict]:
        """Convert model parameters to flat optimization vector."""
        params_list = []
        bounds_list = []
        structure = {}

        idx = 0
        free_masks = model.free_parameter_masks
        for name, values in model.parameters.items():
            canonical = model._canonical_parameter_values(name, values)
            free_mask = np.asarray(free_masks[name], dtype=np.bool_)
            if free_mask.shape != values.shape:
                raise RuntimeError(
                    f"free-parameter mask for {name} has shape {free_mask.shape}, "
                    f"expected {values.shape}"
                )
            free_indices = np.flatnonzero(free_mask.ravel())
            flat = canonical.ravel()[free_indices]
            n_params = free_indices.size

            structure[name] = {
                "start_idx": idx,
                "end_idx": idx + n_params,
                "shape": values.shape,
                "free_indices": free_indices,
                "template": canonical,
            }

            params_list.append(flat)
            bounds_list.extend([_bl_parameter_bounds(model, name)] * n_params)

            idx += n_params

        flattened = (
            np.concatenate(params_list)
            if params_list
            else np.empty(0, dtype=np.float64)
        )
        return flattened, bounds_list, structure

    def _unflatten_parameters(
        self,
        model: BaseItemModel,
        params: NDArray[np.float64],
        structure: dict,
    ) -> None:
        """Set model parameters from flat optimization vector."""
        for name, info in structure.items():
            flat_params = params[info["start_idx"] : info["end_idx"]]
            values = info["template"].copy().ravel()
            values[info["free_indices"]] = flat_params
            model._parameters[name] = model._canonical_parameter_values(
                name, values.reshape(info["shape"])
            )

    def _optimizer_options(self, start_value: float) -> dict[str, float]:
        """Return optimizer options whose tolerance is ``tol`` in LL units.

        L-BFGS-B, the default with bounds, and Powell stop on the relative
        change of the objective, so their tolerance is scaled by the
        starting objective ``start_value``; TNC and SLSQP stop on its
        absolute change, and Nelder-Mead on its absolute spread.
        """
        method = self.method.lower() if isinstance(self.method, str) else None
        relative = self.tol / max(1.0, abs(float(start_value)))
        if self.method is None or method in ("l-bfgs-b", "powell"):
            return {"maxiter": self.max_iter, "ftol": relative}
        if method == "tnc":
            return {"maxfun": self.max_iter, "ftol": self.tol}
        if method == "slsqp":
            return {"maxiter": self.max_iter, "ftol": self.tol}
        if method == "nelder-mead":
            return {"maxiter": self.max_iter, "fatol": self.tol}
        if method is None:
            # A custom minimizer receives the historical options.
            return {"maxiter": self.max_iter, "ftol": self.tol}
        return {"maxiter": self.max_iter}

    def _finish_with_newton(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        objective: PreparedBLObjective,
        params: NDArray[np.float64],
        bounds: list[tuple[float, float]],
        structure: dict,
    ) -> tuple[NDArray[np.float64], float, bool, int] | None:
        """Finish an optimizer solution by projected Newton steps.

        Quasi-Newton methods progress slowly along flat directions, such as
        the guessing parameters of a 3PL, and can stop well short of the
        optimum. The exact observed information ``H`` and the analytic
        gradient ``g`` of the negative log-likelihood give Newton steps on
        the coordinates off their bounds; a coordinate on a bound whose
        gradient points out of the box is held. Steps are halved until the
        objective decreases, and stop once the predicted gain
        ``g' H^-1 g / 2`` is at most ``tol``.

        Returns the point, its objective, whether the predicted gain fell
        to ``tol``, and the number of steps taken, or None without the exact
        information.
        """
        from scipy.linalg import LinAlgError, cho_factor, cho_solve

        from mirt.estimation.standard_errors import _BOUND_TOLERANCE

        limits = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
        value, gradient = objective(params)
        for n_steps in range(_NEWTON_STEPS + 1):
            terms = self._exact_terms(model, responses, params, structure)
            if terms is None:
                return None
            held = ((params <= limits[:, 0] + _BOUND_TOLERANCE) & (gradient > 0.0)) | (
                (params >= limits[:, 1] - _BOUND_TOLERANCE) & (gradient < 0.0)
            )
            free = ~held
            try:
                factor = cho_factor(terms.information[np.ix_(free, free)])
            except LinAlgError:
                # Not a local maximum on the free coordinates.
                return params, value, False, n_steps
            step = cho_solve(factor, -gradient[free])
            if -0.5 * float(gradient[free] @ step) <= self.tol:
                return params, value, True, n_steps
            if n_steps == _NEWTON_STEPS:
                break
            scale = 1.0
            for _ in range(_NEWTON_HALVINGS):
                trial = params.copy()
                trial[free] += scale * step
                trial = np.clip(trial, limits[:, 0], limits[:, 1])
                trial_value, trial_gradient = objective(trial)
                if trial_value < value:
                    break
                scale *= 0.5
            else:
                return params, value, False, n_steps
            params, value, gradient = trial, trial_value, trial_gradient
        return params, value, False, _NEWTON_STEPS

    def _exact_terms(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
    ) -> LouisInformation | None:
        """Louis information and score, when they describe the BL objective.

        The model is left at ``params``. The latest terms are cached, so the
        standard errors at a point that :meth:`_finish_with_newton` checked
        reuse its information.
        """
        from mirt.estimation._louis_information import (
            louis_information,
            supports_louis_information,
        )
        from mirt.estimation.standard_errors import _flatten_parameters

        own_objective = not any(
            name in vars(self)
            or getattr(type(self), name) is not getattr(BLEstimator, name)
            for name in (
                "_compute_marginal_log_likelihood",
                "_flatten_parameters",
                "_unflatten_parameters",
            )
        )
        if (
            not own_objective
            or not supports_louis_information(model)
            or not _uses_free_layout(model, structure)
        ):
            return None
        self._unflatten_parameters(model, params, structure)
        cached = self._information_cache
        if (
            cached is not None
            and cached[0] is responses
            and np.array_equal(cached[1], params)
        ):
            return cached[2]
        _, layouts = _flatten_parameters(model)
        weights = self._quadrature.weights
        terms = louis_information(
            model, responses, self._quadrature.nodes, weights / weights.sum(), layouts
        )
        self._information_cache = (
            responses,
            np.array(params, dtype=np.float64),
            terms,
        )
        return terms

    def _exact_information(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
    ) -> NDArray[np.float64] | None:
        """Louis observed information, when it describes the BL objective."""
        terms = self._exact_terms(model, responses, params, structure)
        return None if terms is None else terms.information

    def _marginal_hessian(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
        *,
        objective: Callable[[NDArray[np.float64]], float] | None = None,
        gradient: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
        | None = None,
    ) -> NDArray[np.float64]:
        """Return the Hessian of the negative marginal log-likelihood.

        Item models accepted by
        :func:`~mirt.estimation._louis_information.supports_louis_information`
        on the estimator's own likelihood use the exact Louis observed
        information, whichever optimizer produced ``params``. Other analytic
        ``gradient`` objectives are differenced once per coordinate, and the
        remaining models, or a caller-supplied ``objective``, difference the
        likelihood over every coordinate pair.
        """
        if objective is None:
            exact = self._exact_information(model, responses, params, structure)
            if exact is not None:
                return exact
        n_params = params.size
        hessian = np.zeros((n_params, n_params))
        if gradient is not None:
            for column in range(n_params):
                plus, minus = params.copy(), params.copy()
                plus[column] += _GRADIENT_STEP
                minus[column] -= _GRADIENT_STEP
                hessian[:, column] = (gradient(plus)[1] - gradient(minus)[1]) / (
                    2.0 * _GRADIENT_STEP
                )
            return (hessian + hessian.T) / 2.0

        def neg_ll(candidate: NDArray[np.float64]) -> float:
            if objective is not None:
                return objective(candidate)
            self._unflatten_parameters(model, candidate, structure)
            return -self._compute_marginal_log_likelihood(model, responses)

        h = _LIKELIHOOD_STEP
        try:
            center = neg_ll(params)
            for row in range(n_params):
                shifted = {}
                for sign in (1.0, -1.0):
                    candidate = params.copy()
                    candidate[row] += sign * h
                    shifted[sign] = neg_ll(candidate)
                hessian[row, row] = (shifted[1.0] - 2.0 * center + shifted[-1.0]) / h**2
                for column in range(row + 1, n_params):
                    total = 0.0
                    for row_sign, column_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        candidate = params.copy()
                        candidate[row] += row_sign * h
                        candidate[column] += column_sign * h
                        total += row_sign * column_sign * neg_ll(candidate)
                    hessian[row, column] = hessian[column, row] = total / (4.0 * h**2)
        finally:
            if objective is None:
                self._unflatten_parameters(model, params, structure)
        return hessian

    def _parameter_covariance(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
        *,
        objective: Callable[[NDArray[np.float64]], float] | None = None,
        gradient: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
        | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> NDArray[np.float64]:
        """Invert the marginal Hessian, holding coordinates on a bound fixed."""
        from mirt.estimation.standard_errors import _BOUND_TOLERANCE, _covariance

        hessian = self._marginal_hessian(
            model,
            responses,
            params,
            structure,
            objective=objective,
            gradient=gradient,
        )
        active = np.ones(params.size, dtype=np.bool_)
        if bounds is not None:
            limits = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
            active &= np.abs(params - limits[:, 0]) > _BOUND_TOLERANCE
            active &= np.abs(params - limits[:, 1]) > _BOUND_TOLERANCE
        return _covariance("oakes", hessian, None, active)

    def _unflatten_standard_errors(
        self,
        model: BaseItemModel,
        covariance: NDArray[np.float64],
        structure: dict,
    ) -> dict[str, NDArray[np.float64]]:
        variances = np.diag(covariance)
        se_flat = np.full(variances.shape, np.nan)
        valid = np.isfinite(variances) & (variances >= 0.0)
        se_flat[valid] = np.sqrt(variances[valid])

        se_dict = {}
        for name, info in structure.items():
            full_se = np.zeros(info["shape"], dtype=np.float64)
            full_se.ravel()[info["free_indices"]] = se_flat[
                info["start_idx"] : info["end_idx"]
            ]
            se_dict[name] = model._expand_parameter_standard_errors(name, full_se)
        return se_dict

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
        *,
        objective: Callable[[NDArray[np.float64]], float] | None = None,
        gradient: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
        | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors from the inverse marginal Hessian."""
        covariance = self._parameter_covariance(
            model,
            responses,
            params,
            structure,
            objective=objective,
            gradient=gradient,
            bounds=bounds,
        )
        return self._unflatten_standard_errors(model, covariance, structure)


def _bl_parameter_bounds(model: BaseItemModel, name: str) -> tuple[float, float]:
    """Optimizer box of every free coordinate of a stored parameter.

    Qualified parameters of a mixed-format model use their component's box.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        component, local = model.parameter_component(name)
        return _bl_parameter_bounds(model.component_models[component], local)
    if name == "slopes" and model.model_name == "NRM":
        return (-5.0, 5.0)
    # Logistic slopes and locations have EM's wide boxes.
    logistic = _logistic_bounds(model, name)
    if logistic is not None:
        return logistic
    return _BL_BOUNDS.get(name, (-10.0, 10.0))


def _uses_free_layout(model: BaseItemModel, structure: dict) -> bool:
    """Whether a BL layout orders coordinates as ``FitResult.vcov`` does."""
    masks = model.free_parameter_masks
    parameters = model.parameters
    return list(structure) == list(parameters) and all(
        np.array_equal(
            info["free_indices"], np.flatnonzero(np.asarray(masks[name]).ravel())
        )
        for name, info in structure.items()
    )
