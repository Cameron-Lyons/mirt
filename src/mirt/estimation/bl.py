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

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class BLEstimator(BaseEstimator):
    """Bock-Lieberman marginal maximum likelihood estimator.

    This estimator uses direct numerical optimization of the marginal
    likelihood, jointly estimating all item parameters simultaneously.
    Unlike EM, there is no alternation between E and M steps.

    Parameters
    ----------
    n_quadpts : int
        Number of quadrature points for numerical integration.
    max_iter : int
        Maximum number of optimization iterations.
    tol : float
        Convergence tolerance for optimizer.
    verbose : bool
        Print optimization progress.
    method : str
        Optimization method for scipy.optimize.minimize.

    Notes
    -----
    The BL method directly maximizes:

        L(xi) = prod_i integral P(x_i | theta)^{x_i} Q(x_i | theta)^{1-x_i} g(theta) dtheta

    using numerical quadrature to approximate the integral.

    For dichotomous 2PL models, this reduces to optimizing 2*n_items parameters
    simultaneously. The method can be less stable than EM for complex models
    but may converge faster for simple cases.
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        method: str = "L-BFGS-B",
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise ValueError("n_quadpts must be at least 5")

        self.n_quadpts = n_quadpts
        self.method = method
        self._quadrature: GaussHermiteQuadrature | None = None

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
        from mirt.results.fit_result import FitResult

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        if not model._is_fitted:
            model._initialize_parameters()

        initial_params, bounds, param_structure = self._flatten_parameters(model)

        def neg_log_likelihood(params: NDArray[np.float64]) -> float:
            self._unflatten_parameters(model, params, param_structure)
            ll = self._compute_marginal_log_likelihood(model, responses)
            return -ll

        def _verbose_callback(x: NDArray[np.float64]) -> None:
            print(f"LL = {-neg_log_likelihood(x):.4f}")

        callback = _verbose_callback if self.verbose else None

        result = minimize(
            neg_log_likelihood,
            x0=initial_params,
            method=self.method,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
            callback=callback,
        )

        self._unflatten_parameters(model, result.x, param_structure)
        model._is_fitted = True

        final_ll = -result.fun
        n_iterations = result.nit if hasattr(result, "nit") else 0
        converged = result.success

        se = self._compute_standard_errors(model, responses, result.x, param_structure)

        n_params = len(result.x)
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

        bounds_map = {
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

        idx = 0
        for name, values in model.parameters.items():
            if model.model_name == "1PL" and name == "discrimination":
                continue

            flat = values.ravel()
            n_params = len(flat)

            structure[name] = {
                "start_idx": idx,
                "end_idx": idx + n_params,
                "shape": values.shape,
            }

            params_list.append(flat)

            bound = bounds_map.get(name, (-10.0, 10.0))
            bounds_list.extend([bound] * n_params)

            idx += n_params

        return np.concatenate(params_list), bounds_list, structure

    def _unflatten_parameters(
        self,
        model: BaseItemModel,
        params: NDArray[np.float64],
        structure: dict,
    ) -> None:
        """Set model parameters from flat optimization vector."""
        for name, info in structure.items():
            flat_params = params[info["start_idx"] : info["end_idx"]]
            model._parameters[name] = flat_params.reshape(info["shape"])

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        structure: dict,
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors using numerical Hessian."""
        h = 1e-5
        n_params = len(params)

        def neg_ll(p):
            self._unflatten_parameters(model, p, structure)
            return -self._compute_marginal_log_likelihood(model, responses)

        hessian_diag = np.zeros(n_params)
        ll_center = neg_ll(params)

        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += h
            params_minus = params.copy()
            params_minus[i] -= h

            ll_plus = neg_ll(params_plus)
            ll_minus = neg_ll(params_minus)

            hessian_diag[i] = (ll_plus - 2 * ll_center + ll_minus) / (h**2)

        self._unflatten_parameters(model, params, structure)

        se_flat = np.where(
            hessian_diag > 0,
            np.sqrt(1.0 / hessian_diag),
            np.nan,
        )

        se_dict = {}
        for name, info in structure.items():
            se_values = se_flat[info["start_idx"] : info["end_idx"]]
            se_dict[name] = se_values.reshape(info["shape"])

        if model.model_name == "1PL":
            se_dict["discrimination"] = np.zeros(model.n_items)

        return se_dict
