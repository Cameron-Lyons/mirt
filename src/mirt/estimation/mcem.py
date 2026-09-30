"""Monte Carlo EM (MCEM) and Quasi-Monte Carlo EM (QMCEM) estimators.

These estimation methods are useful for high-dimensional IRT models where
Gauss-Hermite quadrature becomes computationally infeasible. They use
Monte Carlo integration in the E-step instead of numerical quadrature.

MCEM uses standard pseudo-random sampling, while QMCEM uses low-discrepancy
sequences (Quasi-Monte Carlo) for more uniform coverage of the integration
space and faster convergence.

References:
    Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the
        EM algorithm and the poor man's data augmentation algorithms.
        Journal of the American Statistical Association, 85(411), 699-704.

    Cagnone, S., & Monari, P. (2013). Latent variable models for ordinal
        data by using the adaptive quadrature approximation.
        Computational Statistics, 28(2), 597-619.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import xlog1py, xlogy
from scipy.stats import qmc

from mirt.constants import PROB_EPSILON
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import BaseEstimator
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_MAX_LIKELIHOOD_ELEMENTS = 2_000_000
_MAX_QMC_COUNT_ELEMENTS = 1_000_000


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    """Return a validated integer count."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _boolean(value: bool, name: str) -> bool:
    """Return a validated Boolean option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _seed_value(seed: int | None) -> int | None:
    """Return a backend-safe non-negative random seed."""
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, Integral)
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    return int(seed)


def _validated_prior(
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a finite Gaussian prior and return its Cholesky factor."""
    mean = (
        np.zeros(n_factors, dtype=np.float64)
        if prior_mean is None
        else np.asarray(prior_mean, dtype=np.float64)
    )
    if mean.shape != (n_factors,):
        raise ValueError(f"prior_mean must have shape ({n_factors},)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("prior_mean must contain only finite values")

    covariance = (
        np.eye(n_factors, dtype=np.float64)
        if prior_cov is None
        else np.asarray(prior_cov, dtype=np.float64)
    )
    expected_shape = (n_factors, n_factors)
    if covariance.shape != expected_shape:
        raise ValueError(f"prior_cov must have shape {expected_shape}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("prior_cov must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("prior_cov must be symmetric")
    covariance = (covariance + covariance.T) * 0.5
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_cov must be positive definite") from exc
    return mean.copy(), cholesky


class MCEMEstimator(BaseEstimator):
    """Monte Carlo EM estimator for IRT models.

    Uses Monte Carlo integration in the E-step, making it suitable for
    models with many latent dimensions where quadrature is infeasible.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples per person per iteration.
        More samples give more accurate E-step but slower computation.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood change.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for reproducibility.
    importance_sampling : bool
        Whether to use importance sampling from the prior.
        Improves efficiency when posterior differs from prior.

    Notes
    -----
    MCEM is particularly useful when:
    - The model has more than 3-4 latent dimensions
    - Quadrature-based EM is too slow
    - Exact integration is not required

    The number of samples should increase as iterations progress to
    ensure convergence. This implementation uses a fixed number for
    simplicity.

    Built-in logistic, affine, and polytomous items use shared analytic
    gradients. Small item sample blocks are prepared once; larger draws
    stream bounded blocks without copying every observed person's samples.
    Custom models and item objectives retain numerical optimization.
    """

    _minimum_samples = 50

    def __init__(
        self,
        n_samples: int = 500,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        importance_sampling: bool = True,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        self.n_samples = _positive_integer(
            n_samples, "n_samples", minimum=self._minimum_samples
        )
        self.seed = _seed_value(seed)
        self.importance_sampling = _boolean(importance_sampling, "importance_sampling")
        self._rng: np.random.Generator | None = None

    def _random_generator(self) -> np.random.Generator:
        """Return the initialized fit-local random generator."""
        if self._rng is None:
            raise RuntimeError("the estimator random generator is not initialized")
        return self._rng

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Fit model using Monte Carlo EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit
        responses : ndarray of shape (n_persons, n_items)
            Response matrix
        prior_mean : ndarray of shape (n_factors,), optional
            Prior mean for latent abilities
        prior_cov : ndarray of shape (n_factors, n_factors), optional
            Prior covariance for latent abilities

        Returns
        -------
        FitResult
            Fitted model with estimates and diagnostics
        """
        from mirt.results.fit_result import FitResult

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        self._rng = np.random.default_rng(self.seed)

        prior_mean, cholesky = _validated_prior(prior_mean, prior_cov, n_factors)

        if not model._is_fitted:
            model._initialize_parameters()

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            theta_samples, weights = self._e_step_mc(
                model, responses, prior_mean, cholesky, n_factors
            )

            current_ll = self._estimate_marginal_ll(
                model, responses, theta_samples, weights
            )
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step_mc(model, responses, theta_samples, weights)
        else:
            current_ll, weights = self._refresh_mc_state(
                model, responses, theta_samples, weights
            )
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)

        model._is_fitted = True

        standard_errors = self._compute_standard_errors_mc(
            model, responses, theta_samples, weights
        )

        n_params = model.n_parameters
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=converged,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    @staticmethod
    def _validated_log_likelihoods(
        values: NDArray[np.float64],
        expected_shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Validate a person-by-sample log-likelihood matrix."""
        log_likelihoods = np.asarray(values, dtype=np.float64)
        if log_likelihoods.shape != expected_shape:
            raise ValueError(
                "model log-likelihood output has shape "
                f"{log_likelihoods.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(log_likelihoods)):
            raise ValueError("model log-likelihood output must be finite")
        return log_likelihoods

    def _sample_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate person-specific samples in memory-bounded batches."""
        n_persons = responses.shape[0]
        expected_shape = (n_persons, self.n_samples, model.n_factors)
        samples = np.asarray(theta_samples, dtype=np.float64)
        if samples.shape != expected_shape:
            raise ValueError(f"theta_samples must have shape {expected_shape}")
        if not np.all(np.isfinite(samples)):
            raise ValueError("theta_samples must contain only finite values")

        log_likelihoods = np.empty((n_persons, self.n_samples), dtype=np.float64)
        elements_per_sample = max(1, n_persons * model.n_items)
        chunk_size = max(
            1,
            min(self.n_samples, _MAX_LIKELIHOOD_ELEMENTS // elements_per_sample),
        )
        for start in range(0, self.n_samples, chunk_size):
            stop = min(start + chunk_size, self.n_samples)
            width = stop - start
            theta_chunk = samples[:, start:stop, :].reshape(
                n_persons * width, model.n_factors
            )
            response_chunk = np.repeat(responses, width, axis=0)
            values = np.asarray(
                model.log_likelihood(response_chunk, theta_chunk), dtype=np.float64
            )
            if values.shape != (n_persons * width,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "model.log_likelihood() returned invalid sampled values"
                )
            log_likelihoods[:, start:stop] = values.reshape(n_persons, width)
        return log_likelihoods

    @staticmethod
    def _normalized_importance_weights(
        log_likelihoods: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize prior-sample likelihood ratios by person."""
        # Custom callers may supply views or cache a read-only likelihood array.
        owned = np.array(log_likelihoods, dtype=np.float64, copy=True)
        weights, normalizer = normalize_log_posterior(owned)
        return weights, normalizer[:, None]

    @staticmethod
    def _gaussian_log_kernel(
        theta_samples: NDArray[np.float64],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate a Gaussian prior up to its shared normalizing constant."""
        centered = theta_samples - prior_mean
        original_shape = centered.shape[:-1]
        flattened = centered.reshape(-1, centered.shape[-1])
        standardized = np.linalg.solve(cholesky, flattened.T).T
        return (-0.5 * np.sum(standardized**2, axis=1)).reshape(original_shape)

    def _draw_posterior_samples(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Draw parallel Metropolis samples from every person posterior."""
        n_persons = responses.shape[0]
        rng = self._random_generator()
        standard_normal = rng.standard_normal((n_persons, self.n_samples, n_factors))
        current = prior_mean + np.einsum("ij,...j->...i", cholesky, standard_normal)
        current_ll = self._sample_log_likelihoods(model, responses, current)
        current_lp = self._gaussian_log_kernel(current, prior_mean, cholesky)

        proposal_scale = 0.5
        for _ in range(20):
            standard_proposal = rng.standard_normal(current.shape)
            proposal = current + proposal_scale * np.einsum(
                "ij,...j->...i", cholesky, standard_proposal
            )
            proposal_ll = self._sample_log_likelihoods(model, responses, proposal)
            proposal_lp = self._gaussian_log_kernel(proposal, prior_mean, cholesky)
            log_acceptance = (proposal_ll + proposal_lp) - (current_ll + current_lp)
            uniforms = np.maximum(
                rng.random((n_persons, self.n_samples)),
                np.nextafter(0.0, 1.0),
            )
            accepted = np.log(uniforms) < log_acceptance
            current[accepted] = proposal[accepted]
            current_ll[accepted] = proposal_ll[accepted]
            current_lp[accepted] = proposal_lp[accepted]

        weights = np.full(
            (n_persons, self.n_samples),
            1.0 / self.n_samples,
            dtype=np.float64,
        )
        return current, weights

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Monte Carlo sampling.

        Returns theta samples and their importance weights for each person.
        """
        n_persons = responses.shape[0]
        if not self.importance_sampling:
            return self._draw_posterior_samples(
                model, responses, prior_mean, L, n_factors
            )
        rng = self._random_generator()
        z = rng.standard_normal((n_persons, self.n_samples, n_factors))
        theta_samples = prior_mean + np.einsum("ij,...j->...i", L, z)
        log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        weights, _ = self._normalized_importance_weights(log_likes)
        return theta_samples, weights

    def _estimate_marginal_ll(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Estimate the marginal log-likelihood on the current samples."""
        marginal_ll, _ = self._refresh_mc_state(
            model, responses, theta_samples, weights
        )
        return marginal_ll

    def _refresh_mc_state(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate current parameters on an existing Monte Carlo draw."""
        log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        if self.importance_sampling:
            weights, log_normalizer = self._normalized_importance_weights(log_likes)
            log_marginal = log_normalizer.ravel() - np.log(self.n_samples)
        else:
            # Posterior draws satisfy E[1 / p(y | theta)] = 1 / p(y).
            inverse_normalizer = logsumexp(-log_likes, axis=1, keepdims=False) - np.log(
                self.n_samples
            )
            log_marginal = -inverse_normalizer

        return float(np.sum(log_marginal)), weights

    def _m_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """M-step: optimize item parameters using weighted samples."""
        n_items = model.n_items

        for item_idx in range(n_items):
            self._optimize_item_mc(model, item_idx, responses, theta_samples, weights)

    def _item_expected_log_likelihood(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return a chunked weighted log-likelihood for one observed item."""
        n_persons = len(responses)
        if theta_samples.shape != (
            n_persons,
            self.n_samples,
            model.n_factors,
        ):
            raise ValueError("theta_samples has an incompatible shape")
        if weights.shape != (n_persons, self.n_samples):
            raise ValueError("weights has an incompatible shape")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")

        elements_per_sample = max(1, n_persons * model.n_items)
        chunk_size = max(
            1,
            min(self.n_samples, _MAX_LIKELIHOOD_ELEMENTS // elements_per_sample),
        )
        total = 0.0
        for start in range(0, self.n_samples, chunk_size):
            stop = min(start + chunk_size, self.n_samples)
            width = stop - start
            theta_chunk = theta_samples[:, start:stop, :].reshape(
                n_persons * width, model.n_factors
            )
            response_chunk = np.repeat(responses, width).astype(np.intp, copy=False)
            weight_chunk = weights[:, start:stop].reshape(-1)
            probabilities = np.asarray(
                model.probability(theta_chunk, item_idx), dtype=np.float64
            )

            if model.is_polytomous:
                if (
                    probabilities.ndim != 2
                    or probabilities.shape[0] != n_persons * width
                    or np.any(response_chunk >= probabilities.shape[1])
                ):
                    raise ValueError(
                        "model returned invalid item category probabilities"
                    )
                selected = probabilities[np.arange(n_persons * width), response_chunk]
                if not np.all(np.isfinite(selected)):
                    raise ValueError("model returned non-finite item probabilities")
                log_probability = np.log(np.clip(selected, PROB_EPSILON, 1.0))
            else:
                probabilities = probabilities.reshape(-1)
                if probabilities.shape != (n_persons * width,) or not np.all(
                    np.isfinite(probabilities)
                ):
                    raise ValueError("model returned invalid item probabilities")
                probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
                log_probability = response_chunk * np.log(probabilities) + (
                    1 - response_chunk
                ) * np.log1p(-probabilities)
            total += float(weight_chunk @ log_probability)
        return total

    def _optimize_item_mc(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Optimize parameters for a single item using MC samples."""
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        if not valid_mask.any():
            return

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)
        if current_params.size == 0:
            return

        if self._uses_default_item_methods():
            from mirt.estimation._mc_objective import prepare_mc_objective

            prepared = prepare_mc_objective(
                model,
                item_idx,
                item_responses,
                theta_samples,
                weights,
                self.n_samples,
                bounds,
            )
            if prepared is not None:
                self._minimize_item_mc(
                    model, item_idx, current_params, bounds, prepared, analytic=True
                )
                return

        valid_responses = item_responses[valid_mask]
        valid_theta = theta_samples[valid_mask]
        valid_weights = weights[valid_mask]

        def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
            self._set_item_params(model, item_idx, params)
            return -self._item_expected_log_likelihood(
                model,
                item_idx,
                valid_responses,
                valid_theta,
                valid_weights,
            )

        self._minimize_item_mc(
            model, item_idx, current_params, bounds, neg_expected_log_likelihood
        )

    def _uses_default_item_methods(self) -> bool:
        return type(self) in (
            MCEMEstimator,
            QMCEMEstimator,
            StochasticEMEstimator,
        ) and all(
            name not in vars(self) and getattr(type(self), name) is method
            for name, method in _DEFAULT_MC_ITEM_METHODS.items()
        )

    def _minimize_item_mc(
        self,
        model: BaseItemModel,
        item_idx: int,
        current_params: NDArray[np.float64],
        bounds: list[tuple[float, float]],
        objective: Callable[
            [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
        ],
        *,
        analytic: bool = False,
    ) -> None:
        """Install a finite optimizer result, restoring the item after failures."""
        installed = False
        try:
            result = minimize(
                objective,
                x0=current_params,
                method="L-BFGS-B",
                jac=analytic,
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-6},
            )
            candidate = np.asarray(result.x, dtype=np.float64)
            if (
                candidate.shape != current_params.shape
                or not np.all(np.isfinite(candidate))
                or not np.isfinite(result.fun)
            ):
                raise RuntimeError("item optimization returned invalid parameters")
            self._set_item_params(model, item_idx, candidate)
            installed = True
        finally:
            if not installed:
                self._set_item_params(model, item_idx, current_params)

    def _compute_standard_errors_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Return placeholders for standard errors not yet estimated by MCEM."""
        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model.parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.full_like(values, np.nan)
            standard_errors[name] = se

        return standard_errors


class QMCEMEstimator(MCEMEstimator):
    """Quasi-Monte Carlo EM estimator for IRT models.

    Uses low-discrepancy sequences (Sobol, Halton) instead of pseudo-random
    numbers for more uniform coverage of the integration space. This typically
    leads to faster convergence than standard MCEM.

    Parameters
    ----------
    n_samples : int
        Number of QMC samples per person per iteration.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for scrambling.
    sequence : str
        Type of low-discrepancy sequence: "sobol" or "halton".

    Notes
    -----
    QMCEM typically requires fewer samples than MCEM for the same accuracy
    because the quasi-random points fill the space more uniformly.
    Its shared ability grid also allows the M-step to aggregate expected
    response counts once, instead of evaluating each respondent's samples
    during every optimizer trial. Built-in logistic, affine, and polytomous
    models use analytic gradients.

    References
    ----------
    Niederreiter, H. (1992). Random number generation and quasi-Monte Carlo
        methods. Society for Industrial and Applied Mathematics.
    """

    def __init__(
        self,
        n_samples: int = 256,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        sequence: Literal["sobol", "halton"] = "sobol",
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=True,
        )

        if sequence not in ("sobol", "halton"):
            raise ValueError("sequence must be 'sobol' or 'halton'")

        self.sequence = sequence

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Quasi-Monte Carlo sampling."""
        n_persons = responses.shape[0]

        if self.sequence == "sobol":
            sampler = qmc.Sobol(d=n_factors, scramble=True, seed=self.seed)
            exponent = int(np.ceil(np.log2(self.n_samples)))
            uniform_samples = sampler.random_base2(exponent)[: self.n_samples]
        else:
            sampler = qmc.Halton(d=n_factors, scramble=True, seed=self.seed)
            uniform_samples = sampler.random(self.n_samples)

        from scipy.stats import norm

        lower = np.nextafter(0.0, 1.0)
        upper = np.nextafter(1.0, 0.0)
        uniform_samples = np.clip(uniform_samples, lower, upper)
        z_base = norm.ppf(uniform_samples)

        theta_base = prior_mean + z_base @ L.T
        theta_samples = np.broadcast_to(
            theta_base[None, :, :],
            (n_persons, self.n_samples, n_factors),
        )

        if hasattr(model, "log_likelihood_batch"):
            log_likes = self._validated_log_likelihoods(
                model.log_likelihood_batch(responses, theta_base),
                (n_persons, self.n_samples),
            )
        else:
            log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        weights, _ = self._normalized_importance_weights(log_likes)

        return theta_samples, weights

    def _sample_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Refresh shared-grid likelihoods without respondent/sample expansion."""
        if MCEMEstimator._sample_log_likelihoods is not _DEFAULT_MC_SAMPLE_LIKELIHOODS:
            return super()._sample_log_likelihoods(model, responses, theta_samples)
        n_persons = len(responses)
        samples = np.asarray(theta_samples)
        expected = (n_persons, self.n_samples, model.n_factors)
        if samples.shape != expected:
            raise ValueError(f"theta_samples must have shape {expected}")
        if (
            n_persons == 0
            or samples.strides[0] != 0
            or not hasattr(model, "log_likelihood_batch")
        ):
            return super()._sample_log_likelihoods(model, responses, theta_samples)
        grid = np.asarray(samples[0], dtype=np.float64)
        if not np.all(np.isfinite(grid)):
            raise ValueError("theta_samples must contain only finite values")
        values = self._validated_log_likelihoods(
            model.log_likelihood_batch(responses, grid), (n_persons, self.n_samples)
        )
        # The MC sampler owns its output and may update accepted cells in place.
        return np.array(values, dtype=np.float64, copy=True)

    def _m_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Optimize shared-grid items from expected counts without expanding theta."""
        from mirt.estimation._em_context import EMFitContext

        n_persons = responses.shape[0]
        if theta_samples.shape != (n_persons, self.n_samples, model.n_factors):
            raise ValueError("theta_samples has an incompatible shape")
        if weights.shape != (n_persons, self.n_samples):
            raise ValueError("weights has an incompatible shape")
        if theta_samples.strides[0] != 0 or not self._uses_default_item_methods():
            # Independent draws and custom item callbacks use the MC item path.
            super()._m_step_mc(model, responses, theta_samples, weights)
            return

        row_block = max(1, _MAX_QMC_COUNT_ELEMENTS // self.n_samples)
        for start in range(0, n_persons, row_block):
            block = weights[start : start + row_block]
            if not np.all(np.isfinite(block)) or np.any(block < 0.0):
                raise ValueError("weights must be finite and non-negative")

        if not any(np.any(mask) for mask in model.free_parameter_masks.values()):
            return

        theta = theta_samples[0]
        context = EMFitContext(responses)
        if model.is_polytomous:
            correct = observed = None
        else:
            correct, observed = context.expected_counts(weights)

        for item_idx in range(model.n_items):
            item_responses = responses[:, item_idx]
            if not np.any(item_responses >= 0):
                continue
            params, bounds = self._get_item_params_and_bounds(model, item_idx)
            if params.size == 0:
                continue

            objective: Callable[
                [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
            ]
            analytic = False
            if model.is_polytomous:
                from mirt.estimation._polytomous_objective import (
                    prepare_polytomous_objective,
                )

                n_categories = model.n_categories[item_idx]
                if np.any(item_responses >= n_categories):
                    raise ValueError(
                        "model returned invalid item category probabilities"
                    )
                counts = context.expected_category_counts(
                    item_idx, n_categories, weights
                )
                prepared = prepare_polytomous_objective(
                    model, item_idx, theta, counts, PROB_EPSILON, max_probability=1.0
                )
                analytic = prepared is not None
                if prepared is not None:
                    objective = prepared
                else:

                    def objective(trial):
                        self._set_item_params(model, item_idx, trial)
                        probabilities = np.asarray(
                            model.probability(theta, item_idx), dtype=np.float64
                        )
                        if probabilities.shape != counts.shape or not np.all(
                            np.isfinite(probabilities)
                        ):
                            raise ValueError(
                                "model returned invalid item category probabilities"
                            )
                        return -float(
                            np.sum(
                                xlogy(counts, np.clip(probabilities, PROB_EPSILON, 1.0))
                            )
                        )

            else:
                from mirt.estimation._affine_objective import prepare_affine_objective
                from mirt.estimation._dichotomous_objective import (
                    prepare_dichotomous_objective,
                )

                n_k, r_k = observed[item_idx], correct[item_idx]
                prepared = prepare_dichotomous_objective(
                    model, item_idx, theta, n_k, r_k, PROB_EPSILON, bounds
                )
                if prepared is None:
                    prepared = prepare_affine_objective(
                        model, item_idx, theta, n_k, r_k, PROB_EPSILON, bounds
                    )
                analytic = prepared is not None
                if prepared is not None:
                    objective = prepared
                else:

                    def objective(trial):
                        self._set_item_params(model, item_idx, trial)
                        probabilities = np.asarray(
                            model.probability(theta, item_idx), dtype=np.float64
                        ).reshape(-1)
                        if probabilities.shape != (self.n_samples,) or not np.all(
                            np.isfinite(probabilities)
                        ):
                            raise ValueError(
                                "model returned invalid item probabilities"
                            )
                        probabilities = np.clip(
                            probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON
                        )
                        return -float(
                            np.sum(
                                xlogy(r_k, probabilities)
                                + xlog1py(n_k - r_k, -probabilities)
                            )
                        )

            self._minimize_item_mc(
                model, item_idx, params, bounds, objective, analytic=analytic
            )


class StochasticEMEstimator(MCEMEstimator):
    """Stochastic EM (SEM) estimator for IRT models.

    SEM draws a single sample from the posterior in the E-step instead
    of computing expectations. This makes each iteration faster but
    noisier, requiring more iterations to converge.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed.
    n_chains : int
        Number of independent chains to average over.

    Notes
    -----
    SEM can be useful for very large datasets where computing full
    expectations is too expensive. It converges to a neighborhood of
    the MLE rather than exactly to it.

    References
    ----------
    Celeux, G., & Diebolt, J. (1985). The SEM algorithm: a probabilistic
        teacher algorithm derived from the EM algorithm for the mixture
        problem. Computational Statistics Quarterly, 2(1), 73-82.
    """

    _minimum_samples = 1

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        n_chains: int = 5,
    ) -> None:
        super().__init__(
            n_samples=n_chains,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=False,
        )
        self.n_chains = self.n_samples

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step: sample from posterior using Metropolis-Hastings."""
        return self._draw_posterior_samples(
            model,
            responses,
            prior_mean,
            L,
            n_factors,
        )


_DEFAULT_MC_ITEM_METHODS = {
    name: getattr(MCEMEstimator, name)
    for name in (
        "_item_expected_log_likelihood",
        "_get_item_params_and_bounds",
        "_set_item_params",
        "_optimize_item_mc",
    )
}

_DEFAULT_MC_SAMPLE_LIKELIHOODS = MCEMEstimator._sample_log_likelihoods
