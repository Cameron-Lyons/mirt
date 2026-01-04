"""Expectation-Maximization algorithm for IRT parameter estimation."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class EMEstimator(BaseEstimator):
    """Expectation-Maximization algorithm with Gauss-Hermite quadrature.

    Uses marginal maximum likelihood (MML) estimation via the Bock-Aitkin
    EM algorithm. The E-step computes expected sufficient statistics using
    numerical integration, and the M-step updates item parameters.

    Parameters
    ----------
    n_quadpts : int, default=21
        Number of quadrature points per dimension.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance (change in log-likelihood).
    accelerate : bool, default=True
        Whether to use Ramsay acceleration to speed convergence.
    verbose : bool, default=False
        Whether to print progress information.

    Attributes
    ----------
    n_quadpts : int
        Number of quadrature points.
    accelerate : bool
        Whether acceleration is enabled.

    References
    ----------
    Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
    of item parameters: Application of an EM algorithm. Psychometrika, 46(4),
    443-459.

    Examples
    --------
    >>> from mirt.models import TwoParameterLogistic
    >>> from mirt.estimation import EMEstimator
    >>> import numpy as np
    >>>
    >>> # Create model and estimator
    >>> model = TwoParameterLogistic(n_items=20)
    >>> estimator = EMEstimator(n_quadpts=21, verbose=True)
    >>>
    >>> # Generate some response data (in practice, use real data)
    >>> np.random.seed(42)
    >>> responses = np.random.binomial(1, 0.5, size=(500, 20))
    >>>
    >>> # Fit the model
    >>> result = estimator.fit(model, responses)
    >>> print(result.summary())
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        accelerate: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self.accelerate = accelerate
        self._quadrature: Optional[GaussHermiteQuadrature] = None

    def fit(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        prior_mean: Optional[NDArray[np.float64]] = None,
        prior_cov: Optional[NDArray[np.float64]] = None,
    ) -> "FitResult":
        """Fit model using EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix. Missing values coded as -1.
        prior_mean : ndarray, optional
            Mean of prior distribution for theta. Default: zeros.
        prior_cov : ndarray, optional
            Covariance of prior distribution. Default: identity.

        Returns
        -------
        FitResult
            Fitted model results.
        """
        from mirt.results.fit_result import FitResult

        # Validate input
        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]

        # Initialize quadrature
        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        # Set up prior
        if prior_mean is None:
            prior_mean = np.zeros(model.n_factors)
        if prior_cov is None:
            prior_cov = np.eye(model.n_factors)

        # Initialize parameters if needed
        if not model._is_fitted:
            model._initialize_parameters()

        # Store for convergence check
        self._convergence_history = []
        prev_ll = -np.inf

        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: compute posterior weights
            posterior_weights, marginal_ll = self._e_step(
                model, responses, prior_mean, prior_cov
            )

            # Compute log-likelihood
            current_ll = np.sum(np.log(marginal_ll + 1e-300))
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            # Check convergence
            if self._check_convergence(prev_ll, current_ll):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            # M-step: update item parameters
            self._m_step(model, responses, posterior_weights)

        model._is_fitted = True

        # Compute standard errors
        standard_errors = self._compute_standard_errors(
            model, responses, posterior_weights
        )

        # Compute fit statistics
        n_params = model.n_parameters
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=iteration < self.max_iter - 1,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _e_step(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step: Compute posterior probabilities at quadrature points.

        Parameters
        ----------
        model : BaseItemModel
            The IRT model.
        responses : ndarray
            Response matrix.
        prior_mean : ndarray
            Prior mean.
        prior_cov : ndarray
            Prior covariance.

        Returns
        -------
        posterior_weights : ndarray of shape (n_persons, n_quadpts)
            Posterior probability of each quadrature point for each person.
        marginal_ll : ndarray of shape (n_persons,)
            Marginal likelihood for each person.
        """
        quad_points = self._quadrature.nodes  # (n_quad, n_factors)
        quad_weights = self._quadrature.weights  # (n_quad,)
        n_persons = responses.shape[0]
        n_quad = len(quad_weights)

        # Compute likelihood at each quadrature point for each person
        # log_likelihoods shape: (n_persons, n_quad)
        log_likelihoods = np.zeros((n_persons, n_quad))

        for q in range(n_quad):
            theta_q = np.tile(quad_points[q], (n_persons, 1))
            log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)

        # Log prior at quadrature points (already incorporated in quadrature)
        # For standard normal prior with Gauss-Hermite, prior is built into weights
        log_prior = self._log_multivariate_normal(quad_points, prior_mean, prior_cov)

        # Log joint = log likelihood + log prior + log(weight)
        log_joint = log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]

        # Normalize to get posterior using log-sum-exp for numerical stability
        log_marginal = self._logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _m_step(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> None:
        """M-step: Update item parameters.

        Parameters
        ----------
        model : BaseItemModel
            The IRT model to update.
        responses : ndarray
            Response matrix.
        posterior_weights : ndarray
            Posterior weights from E-step.
        """
        quad_points = self._quadrature.nodes
        n_items = model.n_items

        # Expected counts at each quadrature point
        # n_k = sum over persons of posterior weight for point k
        n_k = posterior_weights.sum(axis=0)  # (n_quad,)

        # For each item, optimize parameters
        for item_idx in range(n_items):
            self._optimize_item(
                model, item_idx, responses, posterior_weights, quad_points, n_k
            )

    def _optimize_item(
        self,
        model: "BaseItemModel",
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
    ) -> None:
        """Optimize parameters for a single item.

        Parameters
        ----------
        model : BaseItemModel
            The IRT model.
        item_idx : int
            Index of item to optimize.
        responses : ndarray
            Response matrix.
        posterior_weights : ndarray
            Posterior weights.
        quad_points : ndarray
            Quadrature points.
        n_k : ndarray
            Expected counts at quadrature points.
        """
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        # Expected number correct at each quadrature point
        # r_k = sum over valid persons of response * posterior weight
        r_k = np.zeros(len(n_k))
        for k in range(len(n_k)):
            r_k[k] = np.sum(item_responses[valid_mask] * posterior_weights[valid_mask, k])

        # Adjust n_k for valid responses
        n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)

        # Get current parameters and bounds based on model type
        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)

        def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
            """Negative expected complete-data log-likelihood for item."""
            # Temporarily set parameters
            self._set_item_params(model, item_idx, params)

            # Compute probabilities at quadrature points
            probs = np.array([
                model.probability(theta.reshape(1, -1), item_idx)[0]
                for theta in quad_points
            ])

            # Clip for numerical stability
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            # Expected log-likelihood
            ll = np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))

            return -ll

        # Optimize
        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6},
        )

        # Set optimized parameters
        self._set_item_params(model, item_idx, result.x)

    def _get_item_params_and_bounds(
        self,
        model: "BaseItemModel",
        item_idx: int,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current item parameters and their bounds.

        Returns
        -------
        params : ndarray
            Current parameter values.
        bounds : list of tuple
            (lower, upper) bounds for each parameter.
        """
        params_list = []
        bounds = []

        # Handle different model types
        model_name = model.model_name

        if model_name in ("1PL", "2PL", "3PL", "4PL"):
            # Discrimination
            if model_name != "1PL":
                a = model._parameters["discrimination"]
                if a.ndim == 1:
                    params_list.append(a[item_idx])
                    bounds.append((0.1, 5.0))
                else:
                    params_list.extend(a[item_idx])
                    bounds.extend([(0.1, 5.0)] * model.n_factors)

            # Difficulty
            b = model._parameters["difficulty"][item_idx]
            params_list.append(b)
            bounds.append((-6.0, 6.0))

            # Guessing (3PL, 4PL)
            if model_name in ("3PL", "4PL"):
                c = model._parameters["guessing"][item_idx]
                params_list.append(c)
                bounds.append((0.0, 0.5))

            # Upper asymptote (4PL)
            if model_name == "4PL":
                d = model._parameters["upper"][item_idx]
                params_list.append(d)
                bounds.append((0.5, 1.0))

        else:
            # Generic handling for polytomous models
            for name, values in model._parameters.items():
                if values.ndim == 1 and len(values) == model.n_items:
                    params_list.append(values[item_idx])
                    if "discrimination" in name or "slope" in name:
                        bounds.append((0.1, 5.0))
                    else:
                        bounds.append((-6.0, 6.0))
                elif values.ndim == 2 and values.shape[0] == model.n_items:
                    params_list.extend(values[item_idx])
                    if "discrimination" in name or "slope" in name:
                        bounds.extend([(0.1, 5.0)] * values.shape[1])
                    else:
                        bounds.extend([(-6.0, 6.0)] * values.shape[1])

        return np.array(params_list), bounds

    def _set_item_params(
        self,
        model: "BaseItemModel",
        item_idx: int,
        params: NDArray[np.float64],
    ) -> None:
        """Set item parameters from flat array."""
        model_name = model.model_name
        idx = 0

        if model_name in ("1PL", "2PL", "3PL", "4PL"):
            # Discrimination
            if model_name != "1PL":
                a = model._parameters["discrimination"]
                if a.ndim == 1:
                    a[item_idx] = params[idx]
                    idx += 1
                else:
                    n_factors = a.shape[1]
                    a[item_idx] = params[idx:idx + n_factors]
                    idx += n_factors

            # Difficulty
            model._parameters["difficulty"][item_idx] = params[idx]
            idx += 1

            # Guessing
            if model_name in ("3PL", "4PL"):
                model._parameters["guessing"][item_idx] = params[idx]
                idx += 1

            # Upper
            if model_name == "4PL":
                model._parameters["upper"][item_idx] = params[idx]
                idx += 1

        else:
            # Generic handling
            for name, values in model._parameters.items():
                if values.ndim == 1 and len(values) == model.n_items:
                    values[item_idx] = params[idx]
                    idx += 1
                elif values.ndim == 2 and values.shape[0] == model.n_items:
                    n_vals = values.shape[1]
                    values[item_idx] = params[idx:idx + n_vals]
                    idx += n_vals

    def _compute_standard_errors(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors using observed information.

        Uses numerical differentiation to approximate the Hessian of the
        log-likelihood, then inverts to get the covariance matrix.
        """
        # For simplicity, use numerical Hessian approximation
        # A more efficient approach would use the Louis (1982) SEM method

        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model._parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                # Fixed parameter, no SE
                standard_errors[name] = np.zeros_like(values)
                continue

            # Compute SE for each item's parameter
            se = np.zeros_like(values)

            for item_idx in range(model.n_items):
                item_se = self._compute_item_se(
                    model, item_idx, name, responses, posterior_weights
                )
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se

            standard_errors[name] = se

        return standard_errors

    def _compute_item_se(
        self,
        model: "BaseItemModel",
        item_idx: int,
        param_name: str,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """Compute SE for a specific item parameter using numerical Hessian."""
        quad_points = self._quadrature.nodes
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        # Get current parameter value
        values = model._parameters[param_name]
        if values.ndim == 1:
            current = values[item_idx]
            is_scalar = True
        else:
            current = values[item_idx].copy()
            is_scalar = False

        # Expected counts
        r_k = np.zeros(len(quad_points))
        n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)
        for k in range(len(quad_points)):
            r_k[k] = np.sum(item_responses[valid_mask] * posterior_weights[valid_mask, k])

        def log_likelihood(param_val):
            """Log-likelihood as function of parameter."""
            if is_scalar:
                model._parameters[param_name][item_idx] = param_val
            else:
                model._parameters[param_name][item_idx] = param_val

            probs = np.array([
                model.probability(theta.reshape(1, -1), item_idx)[0]
                for theta in quad_points
            ])
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            ll = np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))

            # Restore original
            if is_scalar:
                model._parameters[param_name][item_idx] = current
            else:
                model._parameters[param_name][item_idx] = current.copy()

            return ll

        # Numerical second derivative
        h = 1e-5
        if is_scalar:
            ll_plus = log_likelihood(current + h)
            ll_minus = log_likelihood(current - h)
            ll_center = log_likelihood(current)

            hessian = (ll_plus - 2 * ll_center + ll_minus) / (h ** 2)

            if hessian < 0:
                se = np.sqrt(-1.0 / hessian)
            else:
                se = np.nan

            return se
        else:
            # Multidimensional case - return array of SEs
            n_params = len(current)
            se = np.zeros(n_params)

            for i in range(n_params):
                param_plus = current.copy()
                param_plus[i] += h
                param_minus = current.copy()
                param_minus[i] -= h

                ll_plus = log_likelihood(param_plus)
                ll_minus = log_likelihood(param_minus)
                ll_center = log_likelihood(current)

                hessian = (ll_plus - 2 * ll_center + ll_minus) / (h ** 2)

                if hessian < 0:
                    se[i] = np.sqrt(-1.0 / hessian)
                else:
                    se[i] = np.nan

            return se

    @staticmethod
    def _log_multivariate_normal(
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log of multivariate normal density.

        Parameters
        ----------
        x : ndarray of shape (n, d)
            Points to evaluate.
        mean : ndarray of shape (d,)
            Mean vector.
        cov : ndarray of shape (d, d)
            Covariance matrix.

        Returns
        -------
        ndarray of shape (n,)
            Log density values.
        """
        n, d = x.shape
        diff = x - mean

        # Use Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solve = np.linalg.solve(L, diff.T)
            maha = np.sum(solve ** 2, axis=0)
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue decomposition
            sign, log_det = np.linalg.slogdet(cov)
            cov_inv = np.linalg.pinv(cov)
            maha = np.sum(diff @ cov_inv * diff, axis=1)

        log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det)
        return log_norm - 0.5 * maha

    @staticmethod
    def _logsumexp(
        a: NDArray[np.float64],
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> NDArray[np.float64]:
        """Compute log(sum(exp(a))) in a numerically stable way."""
        a_max = np.max(a, axis=axis, keepdims=True)
        result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))

        if not keepdims:
            result = np.squeeze(result, axis=axis)

        return result
