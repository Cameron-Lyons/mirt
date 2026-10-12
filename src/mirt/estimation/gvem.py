from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._backend_config import gpu_dispatch_allowed, validate_use_gpu
from mirt._gpu_backend import (
    gvem_compute_elbo_gpu,
    gvem_e_step_gpu,
    gvem_m_step_gpu,
    is_gpu_available,
)
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation._variational import jaakkola_lambda, variational_e_step
from mirt.estimation._variational_objective import (
    affine_marginal_log_likelihood,
    variational_elbo,
)
from mirt.estimation._variational_statistics import variational_item_statistics
from mirt.estimation.base import (
    BaseEstimator,
    StartValues,
    _apply_starting_values,
    _reject_parameter_restrictions,
    _validate_start,
)
from mirt.exceptions import MirtValidationError
from mirt.results.fit_result import FitResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class GVEMResult(FitResult):
    """Result of :class:`GVEMEstimator`, a ``FitResult`` that keeps the ELBO.

    ``log_likelihood``, AIC and BIC use the marginal log-likelihood at the
    GVEM estimates, comparable with an EM fit, and ``elbo`` is the final
    evidence lower bound that GVEM maximizes, which lies below it.
    """

    elbo: float = float("nan")

    def __post_init__(self) -> None:
        super().__post_init__()
        self.elbo = float(self.elbo)


class GVEMEstimator(BaseEstimator):
    """Gaussian Variational EM estimator for multidimensional IRT models.

    Uses variational inference with the Jaakkola-Jordan bound on the logistic
    function to achieve closed-form E-step updates. This makes GVEM highly
    efficient for high-dimensional models where quadrature-based EM becomes
    computationally infeasible.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for ELBO change.
    verbose : bool
        Whether to print progress during fitting.
    n_inner_iter : int
        Number of inner iterations for variational parameter updates per EM step.
    se_step_size : float
        Step size for numerical differentiation when computing standard errors.
        At the default step size, built-in 1PL/2PL models use exact diagonal
        curvature. Custom objectives and other step sizes use numerical differences.
    use_gpu : bool or "auto", default="auto"
        Whether to run the variational steps on a GPU when one is available.
        ``"auto"`` follows :func:`~mirt.set_backend`: the ``"auto"`` and
        ``"gpu"`` preferences use an available GPU, ``"rust"`` and
        ``"numpy"`` do not. ``True`` and ``False`` override the preference.

    Notes
    -----
    GVEM maintains a Gaussian variational distribution q_i(theta_i) = N(mu_i, Sigma_i)
    for each person's latent abilities. The Jaakkola-Jordan bound provides a
    quadratic lower bound on the logistic log-likelihood, enabling closed-form
    updates for the variational parameters.

    Currently supports 2PL models (unidimensional and multidimensional).

    The estimates maximize the lower bound, not the likelihood, and are
    biased toward smaller slopes (Cho et al., 2021). On a simulated 2PL with
    1000 persons and 20 items, GVEM slopes were 0.79 to 1.01 times the EM
    estimates and the difficulties of the easiest and hardest items moved by
    up to 0.34; on a two-factor test with 16 items, the communalities
    ``diag(A A')`` averaged 0.73 of EM's. Tighter ``tol`` or more
    ``n_inner_iter`` leave this unchanged, because it is the bias of the
    approximation rather than early stopping. Use GVEM for fast approximate
    fits of many factors, and EM, QMC-EM or MH-RM when unbiased estimates
    matter; importance-weighted GVEM (Ma et al., 2024) corrects the bias.

    ``log_likelihood``, AIC and BIC use the marginal log-likelihood at the
    estimates, integrated over the prior on EM's default grid for up to three
    factors and by importance sampling from the variational posteriors for
    more (see :class:`GVEMResult`); ``GVEMResult.elbo`` holds the lower
    bound. Unidimensional fits report observed-information standard errors
    of the marginal likelihood at the estimates (``se_method="oakes"``) and
    their covariance in ``vcov``. Multidimensional fits report the diagonal
    curvature of the bound with the variational posteriors held fixed
    (``se_method="variational_curvature"``), which omits the latent
    uncertainty and understates the standard errors severalfold; refit with
    :class:`~mirt.estimation.em.EMEstimator` for usable uncertainty.

    References
    ----------
    Cho, A. E., Wang, C., Zhang, X., & Xu, G. (2021). Gaussian variational
        estimation for multidimensional item response theory. British Journal
        of Mathematical and Statistical Psychology, 74, 52-85.

    Jaakkola, T. S., & Jordan, M. I. (2000). Bayesian parameter estimation
        via variational methods. Statistics and Computing, 10(1), 25-37.

    Ma, C., Ouyang, J., Wang, C., & Xu, G. (2024). A note on improving
        variational estimation for multidimensional item response theory.
        Psychometrika, 89, 172-204.
    """

    # The closed-form M-step moves every coordinate and gives every item its
    # own slope, so it neither holds fixed values nor ties a common slope.
    _holds_fixed_parameters = False

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        n_inner_iter: int = 3,
        se_step_size: float = 1e-5,
        use_gpu: bool | Literal["auto"] = "auto",
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if (
            not isinstance(n_inner_iter, Integral)
            or isinstance(n_inner_iter, (bool, np.bool_))
            or n_inner_iter < 1
        ):
            raise MirtValidationError(
                "n_inner_iter must be at least 1 and an integer",
                parameter="n_inner_iter",
                value=n_inner_iter,
                expected="positive integer",
            )
        if (
            not isinstance(se_step_size, Real)
            or isinstance(se_step_size, (bool, np.bool_))
            or not np.isfinite(se_step_size)
            or se_step_size <= 0
        ):
            raise MirtValidationError(
                "se_step_size must be finite and positive",
                parameter="se_step_size",
                value=se_step_size,
                expected="finite value > 0",
            )
        self.use_gpu: bool | Literal["auto"] = validate_use_gpu(use_gpu)
        self.n_inner_iter: int = int(n_inner_iter)
        self.se_step_size: float = float(se_step_size)

        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._xi: NDArray[np.float64] | None = None
        self._elbo_history: list[float] = []

        self._slopes: NDArray[np.float64] | None = None
        self._intercepts: NDArray[np.float64] | None = None

    @property
    def _should_use_gpu(self) -> bool:
        """Whether ``use_gpu``, the backend preference and a GPU allow GPU use."""
        return gpu_dispatch_allowed(self.use_gpu) and is_gpu_available()

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        *,
        start: StartValues = "default",
    ) -> GVEMResult:
        """Fit model using Gaussian Variational EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit. Currently supports 2PL models.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix with -1 indicating missing responses.
        prior_mean : ndarray of shape (n_factors,), optional
            Prior mean for latent abilities. Defaults to zeros.
        prior_cov : ndarray of shape (n_factors, n_factors), optional
            Prior covariance for latent abilities. Defaults to identity.
        start : {"default", "model"} or mapping, default="default"
            Starting values, as for :meth:`EMEstimator.fit`.

        Returns
        -------
        GVEMResult
            Fitted model with estimates, the marginal log-likelihood, the
            ELBO and diagnostics.

        Raises
        ------
        ValueError
            If model type is not supported.
        MirtValidationError
            If ``set_free_parameter_masks`` fixes parameters, which the
            closed-form M-step cannot hold.
        MirtModelError
            If the model keeps a free parameter equal across items, such as
            the common slope of ``OneParameterLogistic(...,
            estimate_discrimination=True)``.
        """
        if model.model_name not in ("2PL", "1PL"):
            raise ValueError(
                f"GVEMEstimator currently only supports 2PL models, got {model.model_name}"
            )
        self._check_shared_parameters(model)
        _reject_parameter_restrictions(model, "GVEMEstimator")
        start = _validate_start(start)

        responses = self._validate_responses(responses, model.n_items, model)
        n_persons = responses.shape[0]
        n_items = model.n_items
        n_factors = model.n_factors

        prior_mean, prior_cov = self._validate_prior(
            prior_mean,
            prior_cov,
            n_factors,
        )

        _apply_starting_values(model, start)

        self._convert_to_slope_intercept(model)

        self._initialize_variational_params(
            n_persons,
            n_factors,
            n_items,
            prior_mean,
            prior_cov,
        )

        self._convergence_history = []
        self._elbo_history = []
        prev_elbo = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            self._e_step(model, responses, prior_mean, prior_cov)

            current_elbo = self._compute_elbo(model, responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)

            self._log_iteration(iteration, current_elbo, elbo=current_elbo)

            if self._check_convergence(prev_elbo, current_elbo):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = current_elbo

            self._m_step(model, responses)
        else:
            self._e_step(model, responses, prior_mean, prior_cov)
            current_elbo = self._compute_elbo(model, responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)
            converged = self._check_convergence(prev_elbo, current_elbo)

        self._convert_from_slope_intercept(model)

        model._is_fitted = True

        standard_errors, se_method, vcov = self._standard_errors(
            model,
            responses,
            prior_mean,
            prior_cov,
        )
        log_likelihood = self._marginal_log_likelihood(
            model, responses, prior_mean, prior_cov
        )

        n_params = model.n_parameters
        aic = self._compute_aic(log_likelihood, n_params)
        bic = self._compute_bic(log_likelihood, n_params, n_persons)

        return GVEMResult(
            model=model,
            log_likelihood=log_likelihood,
            n_iterations=iteration + 1,
            converged=converged,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
            se_method=se_method,
            vcov=vcov,
            latent_covariance=(
                None if np.array_equal(prior_cov, np.eye(n_factors)) else prior_cov
            ),
            latent_mean=prior_mean if np.any(prior_mean) else None,
            elbo=current_elbo,
        )

    def _marginal_log_likelihood(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Integrate the likelihood of the reported parameters over the prior."""
        self._convert_to_slope_intercept(model)
        value, _ = affine_marginal_log_likelihood(
            responses,
            self._slopes,
            self._intercepts,
            prior_mean,
            prior_cov,
            mu=self._mu,
            sigma=self._sigma,
        )
        return value

    def _standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[np.float64]], str, NDArray[np.float64] | None]:
        """Return standard errors, their method and the parameter covariance.

        Unidimensional built-in models use the observed information of the
        marginal likelihood at the estimates, integrated on EM's default
        grid over the prior. Others fall back to the variational curvature
        of :meth:`_compute_standard_errors`.
        """
        from mirt.estimation._louis_information import has_analytic_item_derivatives
        from mirt.estimation._patterns import (
            compress_responses,
            supports_pattern_compression,
        )
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.estimation.quadrature import build_quadrature, default_quadpts
        from mirt.estimation.standard_errors import estimate_covariance

        if not has_analytic_item_derivatives(model):
            errors = self._compute_standard_errors(
                model, responses, prior_mean, prior_cov
            )
            return errors, "variational_curvature", None
        frequencies = None
        if supports_pattern_compression(model):
            responses, frequencies = compress_responses(responses)
        quadrature = build_quadrature(default_quadpts(1), 1)
        density = GaussianDensity(mean=prior_mean, cov=prior_cov, n_dimensions=1)
        mass = np.exp(density.log_quadrature_mass(quadrature.nodes, quadrature.weights))
        estimate = estimate_covariance(
            model,
            responses,
            quadrature,
            mass,
            "oakes",
            frequencies=frequencies,
        )
        return estimate.standard_errors, "oakes", estimate.covariance

    @staticmethod
    def _validate_prior(
        prior_mean: NDArray[np.float64] | None,
        prior_cov: NDArray[np.float64] | None,
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return finite prior arrays with a positive-definite covariance."""
        try:
            mean = (
                np.zeros(n_factors, dtype=np.float64)
                if prior_mean is None
                else np.asarray(prior_mean, dtype=np.float64)
            )
            cov = (
                np.eye(n_factors, dtype=np.float64)
                if prior_cov is None
                else np.asarray(prior_cov, dtype=np.float64)
            )
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "prior_mean and prior_cov must be numeric arrays"
            ) from exc

        if mean.shape != (n_factors,):
            raise MirtValidationError(
                "prior_mean shape must match the model dimensions",
                parameter="prior_mean",
                value=mean.shape,
                expected=f"({n_factors},)",
            )
        if not np.all(np.isfinite(mean)):
            raise MirtValidationError(
                "prior_mean must contain only finite values",
                parameter="prior_mean",
                expected="finite values",
            )
        if cov.shape != (n_factors, n_factors):
            raise MirtValidationError(
                "prior_cov shape must match the model dimensions",
                parameter="prior_cov",
                value=cov.shape,
                expected=f"({n_factors}, {n_factors})",
            )
        if not np.all(np.isfinite(cov)):
            raise MirtValidationError(
                "prior_cov must contain only finite values",
                parameter="prior_cov",
                expected="finite values",
            )
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise MirtValidationError(
                "prior_cov must be symmetric",
                parameter="prior_cov",
                expected="symmetric positive-definite matrix",
            )
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise MirtValidationError(
                "prior_cov must be positive definite",
                parameter="prior_cov",
                expected="symmetric positive-definite matrix",
            ) from exc

        return mean.copy(), cov.copy()

    def _convert_to_slope_intercept(self, model: BaseItemModel) -> None:
        """Convert model parameters from discrimination-difficulty to slope-intercept form."""
        a = model.parameters["discrimination"]
        b = model.parameters["difficulty"]

        if model.n_factors == 1:
            self._slopes = a.reshape(-1, 1).copy()
            self._intercepts = (-a * b).copy()
        else:
            self._slopes = a.copy()
            self._intercepts = (-np.sum(a, axis=1) * b).copy()

    def _convert_from_slope_intercept(self, model: BaseItemModel) -> None:
        """Convert parameters from slope-intercept back to discrimination-difficulty form."""
        if model.n_factors == 1:
            a = self._slopes.ravel()
            d = self._intercepts
            b = -d / (a + PROB_EPSILON)

            model._parameters["discrimination"] = a
            model._parameters["difficulty"] = b
        else:
            a = self._slopes
            d = self._intercepts
            a_sum = np.sum(a, axis=1)
            b = -d / (a_sum + PROB_EPSILON)

            model._parameters["discrimination"] = a
            model._parameters["difficulty"] = b

    def _initialize_variational_params(
        self,
        n_persons: int,
        n_factors: int,
        n_items: int,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize variational parameters."""
        if prior_mean is None:
            prior_mean = np.zeros(n_factors, dtype=np.float64)
        if prior_cov is None:
            prior_cov = np.eye(n_factors, dtype=np.float64)
        self._mu = np.broadcast_to(prior_mean, (n_persons, n_factors)).copy()
        self._sigma = np.broadcast_to(
            prior_cov,
            (n_persons, n_factors, n_factors),
        ).copy()
        self._xi = np.ones((n_persons, n_items))

    @staticmethod
    def _lambda(xi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the Jaakkola-Jordan lambda function.

        lambda(xi) = tanh(xi/2) / (4*xi)

        This is numerically stable for xi near 0 where the limit is 1/8.
        """
        return jaakkola_lambda(xi)

    def _e_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> None:
        """E-step: update variational parameters with closed-form updates."""
        prior_cov_inv = np.linalg.solve(prior_cov, np.eye(model.n_factors))
        centered_mu = self._mu - prior_mean
        centered_intercepts = self._intercepts + self._slopes @ prior_mean

        if self._should_use_gpu:
            gpu_result = gvem_e_step_gpu(
                responses,
                self._slopes,
                centered_intercepts,
                prior_cov_inv,
                centered_mu,
                self._sigma,
                self._xi,
                self.n_inner_iter,
            )
            if gpu_result is not None:
                centered_mu, self._sigma, self._xi = gpu_result
                self._mu = centered_mu + prior_mean
                return

        self._e_step_python(responses, prior_mean, prior_cov_inv)

    def _e_step_python(
        self,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov_inv: NDArray[np.float64],
    ) -> None:
        """CPU E-step."""
        self._mu, self._sigma, self._xi = variational_e_step(
            responses,
            self._slopes,
            self._intercepts,
            prior_mean,
            prior_cov_inv,
            self._xi,
            self.n_inner_iter,
            xi_floor=0.0,
            lambda_function=self._lambda,
        )

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> None:
        """M-step: update item parameters with closed-form updates.

        The 1PL holds its slopes at 1, and its intercepts are updated with
        those unit slopes.
        """
        fixed_loadings = model.model_name == "1PL"
        if fixed_loadings:
            self._slopes[np.any(responses >= 0, axis=0)] = 1.0
        if self._should_use_gpu:
            gpu_result = gvem_m_step_gpu(
                responses,
                self._mu,
                self._sigma,
                self._xi,
                self._slopes,
                self._intercepts,
                estimate_loadings=not fixed_loadings,
            )
            if gpu_result is not None:
                self._slopes, self._intercepts = gpu_result
                return

        self._m_step_python(model, responses)

    def _m_step_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> None:
        """CPU M-step."""
        fixed_loadings = model.model_name == "1PL"
        stats = variational_item_statistics(
            responses,
            self._mu,
            self._sigma,
            self._xi,
            self._intercepts,
            self._lambda,
            estimate_loadings=not fixed_loadings,
        )
        observed = stats.observed
        if fixed_loadings:
            self._slopes[observed] = 1.0
        else:
            assert stats.curvature is not None and stats.score is not None
            curvature = stats.curvature[observed]
            score = stats.score[observed]
            curvature += REGULARIZATION_EPSILON * np.eye(model.n_factors)
            try:
                loadings = np.linalg.solve(curvature, score[..., None])[..., 0]
            except np.linalg.LinAlgError:
                loadings = np.empty_like(score)
                for item, (matrix, rhs) in enumerate(
                    zip(curvature, score, strict=True)
                ):
                    try:
                        loadings[item] = np.linalg.solve(matrix, rhs)
                    except np.linalg.LinAlgError:
                        loadings[item] = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            self._slopes[observed] = loadings

        numerator = stats.response_sum - np.einsum(
            "jf,jf->j", self._slopes, stats.weighted_mean
        )
        updated = np.divide(
            numerator,
            stats.weight_sum,
            out=np.zeros_like(self._intercepts),
            where=stats.weight_sum > PROB_EPSILON,
        )
        self._intercepts[observed] = np.clip(updated[observed], -10.0, 10.0)

    def _compute_elbo(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Compute the Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(y|theta)] + E_q[log p(theta)] - E_q[log q(theta)]
        """
        if self._should_use_gpu:
            gpu_result = gvem_compute_elbo_gpu(
                responses,
                self._slopes,
                self._intercepts,
                prior_mean,
                prior_cov,
                self._mu,
                self._sigma,
                self._xi,
            )
            if gpu_result is not None:
                return gpu_result

        return self._compute_elbo_python(model, responses, prior_mean, prior_cov)

    def _compute_elbo_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """CPU ELBO computation."""
        return variational_elbo(
            responses,
            self._slopes,
            self._intercepts,
            self._mu,
            self._sigma,
            self._xi,
            prior_mean,
            prior_cov,
            self._lambda,
        )

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute diagonal ELBO curvature with the variational state held fixed.

        Holding the variational posteriors fixed omits the latent
        uncertainty, so these errors understate the sampling variability.
        """
        from mirt.estimation._gvem_information import gvem_standard_errors
        from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

        self._convert_to_slope_intercept(model)
        if (
            type(self) is GVEMEstimator
            and type(model) in (OneParameterLogistic, TwoParameterLogistic)
            and self.se_step_size == 1e-5
            and not any(
                name in vars(self)
                for name in (
                    "_compute_elbo",
                    "_compute_elbo_python",
                    "_convert_to_slope_intercept",
                    "_lambda",
                )
            )
        ):
            return gvem_standard_errors(
                model, responses, self._mu, self._sigma, self._xi, self._lambda
            )

        standard_errors: dict[str, NDArray[np.float64]] = {}
        center_elbo = self._compute_elbo(model, responses, prior_mean, prior_cov)
        h = self.se_step_size
        for name, values in model.parameters.items():
            free = model.free_parameter_masks[name]
            se = np.zeros_like(values, dtype=np.float64)
            for index in np.ndindex(values.shape):
                if not free[index]:
                    continue
                original = float(values[index])
                try:
                    model._parameters[name][index] = original + h
                    self._convert_to_slope_intercept(model)
                    elbo_plus = self._compute_elbo(
                        model, responses, prior_mean, prior_cov
                    )

                    model._parameters[name][index] = original - h
                    self._convert_to_slope_intercept(model)
                    elbo_minus = self._compute_elbo(
                        model, responses, prior_mean, prior_cov
                    )
                finally:
                    model._parameters[name][index] = original
                    self._convert_to_slope_intercept(model)

                hessian = (elbo_plus - 2 * center_elbo + elbo_minus) / h**2
                se[index] = np.sqrt(-1.0 / hessian) if hessian < 0.0 else np.nan
            standard_errors[name] = model._expand_parameter_standard_errors(name, se)
        return standard_errors

    @property
    def elbo_history(self) -> list[float]:
        """Return history of ELBO values during fitting."""
        return self._elbo_history.copy()

    @property
    def variational_means(self) -> NDArray[np.float64] | None:
        """Return variational means (person ability estimates)."""
        return self._mu.copy() if self._mu is not None else None

    @property
    def variational_covariances(self) -> NDArray[np.float64] | None:
        """Return variational covariances (uncertainty in ability estimates)."""
        return self._sigma.copy() if self._sigma is not None else None
