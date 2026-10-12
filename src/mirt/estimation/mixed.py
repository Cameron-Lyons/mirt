"""Mixed Effects IRT Models.

This module implements explanatory IRT models that incorporate
person and/or item covariates into the IRT framework.

Also known as:
- Latent Regression IRT
- Explanatory IRT
- Linear Logistic Test Model (LLTM) extensions
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import expit, logsumexp

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_LATENT_REGRESSION_MODELS = frozenset({"1PL", "2PL", "3PL"})
# Rectangular ability grid of the latent regression, on the scale where the
# sample's latent distribution has mean 0 and variance 1.
_LATENT_GRID_LIMIT = 6.0
_MIN_RESIDUAL_VARIANCE = 1e-4


@dataclass
class MixedEffectsFitResult:
    """Result from mixed effects IRT estimation.

    Attributes
    ----------
    model : BaseItemModel
        Base IRT model with estimated parameters
    person_effects : NDArray or None
        Regression coefficients for person covariates
    item_effects : NDArray or None
        Regression coefficients for item covariates
    log_likelihood : float
        Marginal log-likelihood of the fitted IRT model, including the
        latent regression when person covariates are given
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    converged : bool
        Whether estimation converged
    person_effect_se : NDArray or None
        Standard errors for person effects
    item_effect_se : NDArray or None
        Standard errors for item effects
    residual_variance : float
        Variance of ability not explained by the person covariates
    theta : NDArray or None
        Estimated person abilities retained for prediction and extraction
    theta_se : NDArray or None
        Standard errors for the estimated person abilities
    person_intercept, item_intercept : float
        Intercepts for the person and item covariate regressions
    person_covariate_names, item_covariate_names : tuple[str, ...]
        Stable names corresponding to the fitted covariate effects
    """

    model: Any
    person_effects: NDArray[np.float64] | None
    item_effects: NDArray[np.float64] | None
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    person_effect_se: NDArray[np.float64] | None = None
    item_effect_se: NDArray[np.float64] | None = None
    residual_variance: float = 1.0
    theta: NDArray[np.float64] | None = None
    theta_se: NDArray[np.float64] | None = None
    person_intercept: float = 0.0
    item_intercept: float = 0.0
    person_intercept_se: float = np.nan
    item_intercept_se: float = np.nan
    person_covariate_names: tuple[str, ...] = ()
    item_covariate_names: tuple[str, ...] = ()

    def summary(self) -> str:
        """Generate summary of mixed effects results."""
        lines = [
            "Mixed Effects IRT Results",
            "=" * 50,
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            f"Converged: {self.converged}",
            "",
        ]

        if self.person_effects is not None:
            lines.append("Person Covariate Effects:")
            lines.append(
                f"  Intercept: {self.person_intercept:.4f} "
                f"(SE: {self.person_intercept_se:.4f})"
            )
            standard_errors = (
                self.person_effect_se
                if self.person_effect_se is not None
                else np.full(self.person_effects.shape, np.nan)
            )
            names = self.person_covariate_names or tuple(
                f"Covariate {i + 1}" for i in range(len(self.person_effects))
            )
            if len(names) != len(self.person_effects):
                names = tuple(
                    f"Covariate {i + 1}" for i in range(len(self.person_effects))
                )
            for i, (est, se) in enumerate(
                zip(self.person_effects, standard_errors, strict=True)
            ):
                lines.append(f"  {names[i]}: {est:.4f} (SE: {se:.4f})")
            lines.append("")

        if self.item_effects is not None:
            lines.append("Item Covariate Effects:")
            lines.append(
                f"  Intercept: {self.item_intercept:.4f} "
                f"(SE: {self.item_intercept_se:.4f})"
            )
            standard_errors = (
                self.item_effect_se
                if self.item_effect_se is not None
                else np.full(self.item_effects.shape, np.nan)
            )
            names = self.item_covariate_names or tuple(
                f"Covariate {i + 1}" for i in range(len(self.item_effects))
            )
            if len(names) != len(self.item_effects):
                names = tuple(
                    f"Covariate {i + 1}" for i in range(len(self.item_effects))
                )
            for i, (est, se) in enumerate(
                zip(self.item_effects, standard_errors, strict=True)
            ):
                lines.append(f"  {names[i]}: {est:.4f} (SE: {se:.4f})")

        return "\n".join(lines)


class MixedEffectsIRT:
    """Mixed effects (explanatory) IRT model.

    This model allows person and item covariates to explain variance:

    For person covariates (latent regression):
        theta_i = beta_0 + beta_1 * Z_i1 + ... + beta_p * Z_ip + epsilon_i,
        epsilon_i ~ N(0, sigma^2)

    For item covariates (LLTM-style):
        b_j = gamma_0 + gamma_1 * W_j1 + ... + gamma_q * W_jq

    where:
    - Z: Person covariates (n_persons x p)
    - W: Item covariates (n_items x q)
    - beta: Person covariate effects
    - gamma: Item covariate effects

    The latent regression is estimated jointly with the item parameters by
    marginal maximum likelihood, as ``mirt(covdata=, formula=)`` does in R.
    Item covariate effects are a second-stage least-squares regression of
    the fitted difficulties and do not enter the likelihood.

    This is useful for:
    - Understanding sources of individual differences
    - Item parameter modeling (e.g., cognitive complexity)
    - DIF analysis with continuous moderators
    """

    def __init__(
        self,
        base_model: Literal["1PL", "2PL", "3PL"] = "2PL",
        person_covariates: NDArray[np.float64] | None = None,
        item_covariates: NDArray[np.float64] | None = None,
        person_covariate_names: Sequence[str] | None = None,
        item_covariate_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize mixed effects IRT model.

        Parameters
        ----------
        base_model : str
            Base IRT model type
        person_covariates : NDArray, optional
            Person-level covariates (n_persons x n_person_cov)
        item_covariates : NDArray, optional
            Item-level covariates (n_items x n_item_cov)
        person_covariate_names : sequence of str, optional
            Names for the columns of ``person_covariates``
        item_covariate_names : sequence of str, optional
            Names for the columns of ``item_covariates``
        """
        self.base_model = base_model
        self._person_covariates = self._prepare_covariates(
            person_covariates, "person_covariates"
        )
        self._item_covariates = self._prepare_covariates(
            item_covariates, "item_covariates"
        )
        self._person_covariate_names = self._prepare_covariate_names(
            person_covariate_names,
            self.n_person_covariates,
            "person",
        )
        self._item_covariate_names = self._prepare_covariate_names(
            item_covariate_names,
            self.n_item_covariates,
            "item",
        )

        self._person_effects: NDArray[np.float64] | None = None
        self._item_effects: NDArray[np.float64] | None = None
        self._person_intercept: float = 0.0
        self._item_intercept: float = 0.0
        self._residual_variance: float = 1.0

    @staticmethod
    def _prepare_covariates(
        covariates: NDArray[np.float64] | None,
        name: str,
    ) -> NDArray[np.float64] | None:
        if covariates is None:
            return None
        values = np.asarray(covariates, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(f"{name} must be one- or two-dimensional")
        if values.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one column")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values

    @staticmethod
    def _prepare_covariate_names(
        names: Sequence[str] | None,
        n_covariates: int,
        prefix: str,
    ) -> tuple[str, ...]:
        if names is None:
            return tuple(f"{prefix}_{i}" for i in range(n_covariates))
        if len(names) != n_covariates:
            raise ValueError(
                f"{prefix}_covariate_names has {len(names)} entries, "
                f"expected {n_covariates}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"{prefix}_covariate_names must be unique")
        if any(not isinstance(name, str) or not name or ":" in name for name in names):
            raise ValueError(
                f"{prefix}_covariate_names must be non-empty strings without ':'"
            )
        return tuple(names)

    @property
    def n_person_covariates(self) -> int:
        """Number of person covariates."""
        if self._person_covariates is None:
            return 0
        return self._person_covariates.shape[1]

    @property
    def n_item_covariates(self) -> int:
        """Number of item covariates."""
        if self._item_covariates is None:
            return 0
        return self._item_covariates.shape[1]

    @property
    def person_covariate_names(self) -> tuple[str, ...]:
        """Names corresponding to person covariate columns."""
        return self._person_covariate_names

    @property
    def item_covariate_names(self) -> tuple[str, ...]:
        """Names corresponding to item covariate columns."""
        return self._item_covariate_names

    def fit(
        self,
        responses: NDArray[np.int_],
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        *,
        n_quadpts: int = 61,
    ) -> MixedEffectsFitResult:
        """Fit mixed effects IRT model.

        The base IRT model is fitted first. With person covariates, the item
        parameters and the latent regression are then estimated jointly by
        marginal maximum-likelihood EM: every respondent's ability has the
        prior ``N(beta_0 + Z_i beta, sigma^2)`` in the E-step, and the M-step
        updates the item parameters from expected response counts and the
        regression from the posterior ability moments. Item covariate
        effects are a least-squares regression of the fitted difficulties.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        max_iter : int, default=500
            Maximum EM iterations for the base fit and for the latent
            regression
        tol : float, default=1e-4
            Convergence tolerance for the absolute change in the marginal
            log-likelihood
        verbose : bool
            Whether to print progress
        n_quadpts : int, default=61
            Number of equally spaced ability points on ``[-6, 6]`` for the
            latent regression, as R mirt uses for one dimension

        Returns
        -------
        MixedEffectsFitResult
            Estimation results. ``log_likelihood``, ``aic`` and ``bic``
            describe the marginal likelihood of the fitted model, and
            ``theta`` holds EAP abilities under the fitted latent regression.

        Notes
        -----
        Item slopes, including the common slope of ``fit_mirt``'s 1PL, leave
        the ability scale free, so the latent regression is reported on the
        scale on which the sample's ability distribution has mean 0 and
        variance 1, as in an ordinary fit: ``person_intercept`` is
        ``-mean(Z) @ person_effects`` and ``residual_variance`` equals
        ``1 - R^2``. R mirt instead fixes the intercept at 0 and the
        residual variance at 1; its coefficients are
        ``person_effects / sqrt(residual_variance)``. A 1PL whose slope is
        fixed at 1 estimates the residual variance on that scale instead.
        Standard errors of the person effects come from the observed
        information of all item and regression parameters.
        """
        from mirt import fit_mirt
        from mirt.scoring import fscores

        responses = np.asarray(responses)
        if responses.ndim != 2 or responses.shape[0] == 0 or responses.shape[1] == 0:
            raise ValueError("responses must be a non-empty two-dimensional matrix")
        n_persons, n_items = responses.shape

        if self._person_covariates is not None:
            if self._person_covariates.shape[0] != n_persons:
                raise ValueError(
                    f"Person covariates shape {self._person_covariates.shape[0]} "
                    f"does not match n_persons {n_persons}"
                )
            if self.base_model not in _LATENT_REGRESSION_MODELS:
                raise MirtValidationError(
                    "Person covariates require a 1PL, 2PL or 3PL base model",
                    parameter="base_model",
                    value=self.base_model,
                    expected="'1PL', '2PL' or '3PL'",
                )
            if (
                isinstance(n_quadpts, (bool, np.bool_))
                or not isinstance(n_quadpts, (int, np.integer))
                or n_quadpts < 3
            ):
                raise MirtValidationError(
                    "n_quadpts must be an integer of at least 3",
                    parameter="n_quadpts",
                    value=n_quadpts,
                    expected=">= 3",
                )

        if self._item_covariates is not None:
            if self._item_covariates.shape[0] != n_items:
                raise ValueError(
                    f"Item covariates shape {self._item_covariates.shape[0]} "
                    f"does not match n_items {n_items}"
                )

        if verbose:
            print("Stage 1: Fitting base IRT model...")

        base_result = fit_mirt(
            responses,
            model=self.base_model,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            compute_standard_errors=False,
        )
        n_params = int(
            getattr(base_result, "n_parameters", base_result.model.n_parameters)
        )

        person_effects = None
        person_se = None
        item_effects = None
        item_se = None
        person_intercept_se = np.nan
        item_intercept_se = np.nan

        if self._person_covariates is not None:
            if verbose:
                print("Stage 2: Estimating the latent regression...")
            regression = _LatentRegressionEM(
                self.base_model,
                responses,
                self._person_covariates,
                n_quadpts=int(n_quadpts),
            )
            fit = regression.fit(
                base_result.model, max_iter=max_iter, tol=tol, verbose=verbose
            )
            fitted_model = fit.model
            ll = fit.log_likelihood
            converged = fit.converged
            theta_estimates = fit.theta
            theta_se = fit.theta_se
            self._person_intercept = fit.intercept
            person_effects = fit.effects
            person_intercept_se = fit.intercept_se
            person_se = fit.effect_se
            self._residual_variance = fit.residual_variance
            self._person_effects = person_effects
            n_params += fit.n_parameters
        else:
            fitted_model = base_result.model
            ll = float(base_result.log_likelihood)
            converged = bool(base_result.converged)
            score_result = fscores(fitted_model, responses, method="EAP")
            theta_estimates = score_result.theta
            theta_se = score_result.standard_error
            if theta_estimates.ndim == 2:
                theta_estimates = theta_estimates[:, 0]
            if theta_se.ndim == 2:
                theta_se = theta_se[:, 0]

        if self._item_covariates is not None:
            if verbose:
                print("Stage 3: Estimating item covariate effects...")
            (
                self._item_intercept,
                item_effects,
                item_intercept_se,
                item_se,
                _,
            ) = self._estimate_item_effects(fitted_model)
            self._item_effects = item_effects

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + np.log(n_persons) * n_params

        return MixedEffectsFitResult(
            model=fitted_model,
            person_effects=person_effects,
            item_effects=item_effects,
            log_likelihood=ll,
            aic=float(aic),
            bic=float(bic),
            converged=converged,
            person_effect_se=person_se,
            item_effect_se=item_se,
            residual_variance=self._residual_variance,
            theta=np.asarray(theta_estimates, dtype=np.float64),
            theta_se=np.asarray(theta_se, dtype=np.float64),
            person_intercept=self._person_intercept,
            item_intercept=self._item_intercept,
            person_intercept_se=float(person_intercept_se),
            item_intercept_se=float(item_intercept_se),
            person_covariate_names=self._person_covariate_names,
            item_covariate_names=self._item_covariate_names,
        )

    @staticmethod
    def _fit_linear_effects(
        covariates: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], float, NDArray[np.float64], float]:
        """Fit a stable linear effect model with intercept and standard errors."""
        outcomes = np.asarray(outcomes, dtype=np.float64).reshape(-1)
        if covariates.shape[0] != outcomes.size:
            raise ValueError("covariates and outcomes must have the same row count")
        if not np.all(np.isfinite(outcomes)):
            raise ValueError("outcomes must contain only finite values")
        design = np.column_stack([np.ones(len(outcomes)), covariates])
        coefficients, _, rank, _ = np.linalg.lstsq(design, outcomes, rcond=None)
        residuals = outcomes - design @ coefficients
        degrees_of_freedom = max(len(outcomes) - rank, 1)
        residual_variance = max(
            float(np.dot(residuals, residuals) / degrees_of_freedom),
            np.finfo(np.float64).eps,
        )
        covariance = residual_variance * np.linalg.pinv(design.T @ design)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return (
            float(coefficients[0]),
            coefficients[1:],
            float(standard_errors[0]),
            standard_errors[1:],
            residual_variance,
        )

    def _estimate_item_effects(
        self,
        model: BaseItemModel,
    ) -> tuple[float, NDArray[np.float64], float, NDArray[np.float64], float]:
        """Estimate effects of item covariates on difficulty."""
        b = model.parameters.get("difficulty", np.zeros(model.n_items))
        return self._fit_linear_effects(self._item_covariates, b)

    def predict_theta(
        self,
        person_covariates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict theta from person covariates.

        Parameters
        ----------
        person_covariates : NDArray
            Covariate values (n_new_persons x n_covariates)

        Returns
        -------
        NDArray
            Predicted theta values
        """
        if self._person_effects is None:
            raise ValueError("Model has not been fit with person covariates")

        covariates = np.asarray(person_covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(1, -1)
        if covariates.ndim != 2 or covariates.shape[1] != len(self._person_effects):
            raise ValueError(
                f"person_covariates must have {len(self._person_effects)} columns"
            )
        if not np.all(np.isfinite(covariates)):
            raise ValueError("person_covariates must contain only finite values")

        return self._person_intercept + covariates @ self._person_effects

    def predict_difficulty(
        self,
        item_covariates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict item difficulty from item covariates.

        Parameters
        ----------
        item_covariates : NDArray
            Covariate values (n_new_items x n_covariates)

        Returns
        -------
        NDArray
            Predicted difficulty values
        """
        if self._item_effects is None:
            raise ValueError("Model has not been fit with item covariates")

        covariates = np.asarray(item_covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(1, -1)
        if covariates.ndim != 2 or covariates.shape[1] != len(self._item_effects):
            raise ValueError(
                f"item_covariates must have {len(self._item_effects)} columns"
            )
        if not np.all(np.isfinite(covariates)):
            raise ValueError("item_covariates must contain only finite values")

        return self._item_intercept + covariates @ self._item_effects


class LLTM:
    """Linear Logistic Test Model.

    The LLTM decomposes item difficulty into a linear combination
    of basic parameters representing cognitive operations:

    b_j = sum_k q_jk * eta_k

    where:
    - q_jk: Weight matrix (like Q-matrix) indicating which cognitive
            operations are required for item j
    - eta_k: Basic parameter for cognitive operation k

    This provides a more parsimonious model and allows prediction
    of difficulty for new items.

    Both the LLTM and the unrestricted Rasch model it is tested against have
    one common slope ``a`` and a standard-normal ability, as
    ``fit_mirt(model="1PL")`` does, which is R mirt's Rasch model with a free
    latent variance. Both are fitted by marginal maximum likelihood on the
    same quadrature, so their log-likelihoods, AIC and BIC are comparable.
    """

    def __init__(
        self,
        q_matrix: NDArray[np.float64],
    ) -> None:
        """Initialize LLTM.

        Parameters
        ----------
        q_matrix : NDArray
            Weight matrix (n_items x n_operations) specifying the
            contribution of each operation to each item
        """
        self._q_matrix = np.asarray(q_matrix)
        self._eta: NDArray[np.float64] | None = None

    def fit(
        self,
        responses: NDArray[np.int_],
        verbose: bool = False,
        *,
        n_quadpts: int = 21,
    ) -> dict[str, Any]:
        """Fit LLTM model.

        The Rasch model is fitted with ``fit_mirt(model="1PL")`` and its
        slope and difficulties are refined by direct maximization of the
        marginal log-likelihood. The common slope and the basic parameters
        ``eta`` then maximize the same marginal log-likelihood with
        ``b = Q @ eta``, starting from the least-squares projection of the
        Rasch difficulties.

        Parameters
        ----------
        responses : NDArray
            Response matrix
        verbose : bool
            Whether to print progress
        n_quadpts : int, default=21
            Number of Gauss-Hermite points for the standard-normal ability,
            as in ``fit_mirt``

        Returns
        -------
        dict
            Fit results: ``eta``, ``difficulty_rasch``, ``difficulty_lltm``,
            the marginal ``log_likelihood_rasch`` and ``log_likelihood_lltm``,
            their ``aic_*`` and ``bic_*`` (counting the common slope), and
            the likelihood-ratio test of the LLTM against the Rasch model:
            ``chi_sq``, ``df = n_items - rank(Q)`` and ``p_value`` (``NaN``
            when ``df == 0``). ``converged`` reports whether the LLTM
            optimizer met its tolerance.
        """
        from scipy import stats

        from mirt import fit_mirt
        from mirt.estimation.quadrature import GaussHermiteQuadrature
        from mirt.utils.data import validate_responses

        responses = np.asarray(responses)
        n_persons, n_items = responses.shape

        if self._q_matrix.shape[0] != n_items:
            raise ValueError(
                f"Q-matrix has {self._q_matrix.shape[0]} rows but data has {n_items} items"
            )
        q_matrix = np.asarray(self._q_matrix, dtype=np.float64)
        if q_matrix.ndim != 2 or not np.all(np.isfinite(q_matrix)):
            raise ValueError("Q-matrix must be a finite two-dimensional matrix")

        rasch_result = fit_mirt(
            responses, model="1PL", verbose=False, compute_standard_errors=False
        )
        values = validate_responses(responses, n_items=n_items)
        quadrature = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=1)
        objective = _RaschMarginalObjective(
            values, quadrature.nodes[:, 0], quadrature.weights
        )

        rasch_parameters = rasch_result.model.parameters
        log_slope = float(np.log(np.mean(rasch_parameters["discrimination"])))
        rasch_start = np.concatenate(
            [[log_slope], np.asarray(rasch_parameters["difficulty"], dtype=float)]
        )
        rasch, ll_rasch, _ = objective.maximize(np.eye(n_items), rasch_start)
        b_rasch = rasch[1:]
        eta_start, _, _, _ = np.linalg.lstsq(q_matrix, b_rasch, rcond=None)
        lltm, ll, converged = objective.maximize(
            q_matrix, np.concatenate([rasch[:1], eta_start])
        )
        self._eta = lltm[1:]
        if verbose:
            print(f"Rasch LL = {ll_rasch:.4f}, LLTM LL = {ll:.4f}")

        b_lltm = q_matrix @ self._eta

        # Both models also estimate the common slope.
        n_params_rasch = n_items + 1
        n_params_lltm = int(np.linalg.matrix_rank(q_matrix)) + 1

        aic_rasch = -2 * ll_rasch + 2 * n_params_rasch
        aic_lltm = -2 * ll + 2 * n_params_lltm

        bic_rasch = -2 * ll_rasch + np.log(n_persons) * n_params_rasch
        bic_lltm = -2 * ll + np.log(n_persons) * n_params_lltm

        # The Rasch model nests the LLTM, so a negative statistic is
        # optimizer noise.
        chi_sq = max(2 * (ll_rasch - ll), 0.0)
        df = n_params_rasch - n_params_lltm

        p_value = float(stats.chi2.sf(chi_sq, df)) if df > 0 else np.nan

        return {
            "eta": self._eta,
            "difficulty_rasch": b_rasch,
            "difficulty_lltm": b_lltm,
            "log_likelihood_rasch": float(ll_rasch),
            "log_likelihood_lltm": float(ll),
            "aic_rasch": float(aic_rasch),
            "aic_lltm": float(aic_lltm),
            "bic_rasch": float(bic_rasch),
            "bic_lltm": float(bic_lltm),
            "chi_sq": float(chi_sq),
            "df": df,
            "p_value": float(p_value),
            "converged": bool(converged),
        }

    def predict_difficulty(
        self,
        q_new: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict difficulty for new items.

        Parameters
        ----------
        q_new : NDArray
            Q-matrix for new items (n_new_items x n_operations)

        Returns
        -------
        NDArray
            Predicted difficulties
        """
        if self._eta is None:
            raise ValueError("Model has not been fit")

        return np.asarray(q_new) @ self._eta


@dataclass
class _LatentRegressionFit:
    """Estimates of a marginal maximum-likelihood latent regression."""

    model: Any
    intercept: float
    effects: NDArray[np.float64]
    intercept_se: float
    effect_se: NDArray[np.float64]
    residual_variance: float
    log_likelihood: float
    converged: bool
    n_iterations: int
    n_parameters: int
    theta: NDArray[np.float64]
    theta_se: NDArray[np.float64]


class _LatentRegressionEM:
    """Joint EM for 1PL, 2PL or 3PL items and a normal latent regression.

    The ability of respondent ``i`` has the prior
    ``N(beta_0 + x_i' beta, sigma^2)``, discretized on an equally spaced grid
    and normalized per respondent, as R mirt does for ``covdata``. Each
    M-step maximizes the expected item log-likelihoods and regresses the
    posterior ability means on the covariates, with ``sigma^2`` the mean
    posterior variance plus the mean squared residual.

    Item locations leave the location of ability free, and free slopes
    (per item, or the common 1PL slope) leave its scale free. After each
    M-step the scale is moved back to sample mean 0 and, with free slopes,
    total variance ``var(x' beta) + sigma^2 = 1``, with the item parameters
    transformed to keep the likelihood: the reduction step of
    parameter-expanded EM (Liu, Rubin and Wu, 1998). A 1PL with its slope
    fixed at 1 estimates ``sigma^2`` on that scale instead.

    Items are stored as optimizer coordinates ``(log a, b[, c])``.
    """

    def __init__(
        self,
        base_model: str,
        responses: NDArray[np.int_],
        covariates: NDArray[np.float64],
        *,
        n_quadpts: int,
    ) -> None:
        from mirt.utils.data import validate_responses

        values = validate_responses(responses)
        if np.any(values > 1):
            raise ValueError("Latent regression requires binary responses")
        observed = values >= 0
        self.base_model = base_model
        # Item gradients use the (log a, b[, c]) layout of these families.
        self.gradient_family = "3PL" if base_model == "3PL" else "2PL"
        self.correct = np.where(observed, values, 0).astype(np.float64)
        self.observed = observed.astype(np.float64)
        self.incorrect = self.observed - self.correct
        self.covariates = np.asarray(covariates, dtype=np.float64)
        self.covariate_mean = self.covariates.mean(axis=0)
        self.centered = self.covariates - self.covariate_mean
        self.covariate_covariance = (
            self.centered.T @ self.centered / self.centered.shape[0]
        )
        self.design = np.column_stack([np.ones(values.shape[0]), self.covariates])
        self.nodes = np.linspace(-_LATENT_GRID_LIMIT, _LATENT_GRID_LIMIT, n_quadpts)
        self.slopes = "item" if base_model != "1PL" else "fixed"

    @property
    def free_scale(self) -> bool:
        """Whether free slopes leave the ability scale to be standardized."""
        return self.slopes != "fixed"

    def _item_values(self, model: BaseItemModel) -> NDArray[np.float64]:
        """Return the model's items as ``(log a, b[, c])`` coordinates."""
        from mirt.models.mixture import (
            _DIFFICULTY_BOUNDS,
            _GUESSING_BOUNDS,
            _LOG_DISCRIMINATION_BOUNDS,
        )

        parameters = model.parameters
        log_slope = np.log(np.maximum(parameters["discrimination"], 1e-300))
        columns = [
            np.clip(log_slope, *_LOG_DISCRIMINATION_BOUNDS),
            np.clip(parameters["difficulty"], *_DIFFICULTY_BOUNDS),
        ]
        if self.slopes == "fixed":
            columns[0] = np.zeros_like(columns[0])
        if self.base_model == "3PL":
            columns.append(np.clip(parameters["guessing"], *_GUESSING_BOUNDS))
        return np.column_stack(columns).astype(np.float64)

    @staticmethod
    def _bounds(family: str) -> list[tuple[float, float]]:
        from mirt.models.mixture import (
            _DIFFICULTY_BOUNDS,
            _GUESSING_BOUNDS,
            _LOG_DISCRIMINATION_BOUNDS,
        )

        if family == "1PL":
            return [_DIFFICULTY_BOUNDS]
        bounds = [_LOG_DISCRIMINATION_BOUNDS, _DIFFICULTY_BOUNDS]
        if family == "3PL":
            bounds.append(_GUESSING_BOUNDS)
        return bounds

    def _posterior(
        self,
        items: NDArray[np.float64],
        prior_mean: NDArray[np.float64],
        residual_variance: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior grid weights and per-person log marginals."""
        slopes = np.exp(items[:, 0])
        logits = slopes[None, :] * (self.nodes[:, None] - items[None, :, 1])
        if self.base_model == "3PL":
            guessing = items[:, 2]
            probability = guessing + (1.0 - guessing) * expit(logits)
            probability = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)
            log_correct = np.log(probability)
            log_incorrect = np.log1p(-probability)
        else:
            log_correct = -np.logaddexp(0.0, -logits)
            log_incorrect = -np.logaddexp(0.0, logits)
        log_prior = (
            -0.5 * (self.nodes[None, :] - prior_mean[:, None]) ** 2 / residual_variance
        )
        log_prior -= logsumexp(log_prior, axis=1, keepdims=True)
        joint = self.correct @ log_correct.T + self.incorrect @ log_incorrect.T
        joint += log_prior
        log_marginal = logsumexp(joint, axis=1)
        posterior = np.exp(joint - log_marginal[:, None])
        return posterior, log_marginal

    def _optimize_items(
        self,
        values: NDArray[np.float64],
        family: str,
        nodes: NDArray[np.float64],
        posterior: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximize every item's expected complete-data log-likelihood."""
        from mirt.models.mixture import _item_expected_objective

        expected_correct = posterior.T @ self.correct
        expected_total = posterior.T @ self.observed
        bounds = self._bounds(family)
        updated = values.copy()
        for item in range(values.shape[0]):
            arguments = (
                family,
                nodes,
                np.clip(expected_correct[:, item], 0.0, expected_total[:, item]),
                expected_total[:, item],
            )
            initial_value, _ = _item_expected_objective(values[item], *arguments)
            result = optimize.minimize(
                _item_expected_objective,
                values[item],
                args=arguments,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-7},
            )
            if (
                np.all(np.isfinite(result.x))
                and np.isfinite(result.fun)
                and float(result.fun) <= initial_value + 1e-8
            ):
                updated[item] = result.x
        return updated

    def _m_step(
        self,
        items: NDArray[np.float64],
        posterior: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Run the expanded M-step and return the standardized estimates."""
        if self.slopes == "item":
            items = self._optimize_items(items, self.base_model, self.nodes, posterior)
            scale = 1.0
        else:
            # Unit slopes on the ability scale ``theta * a``; the reduction
            # below returns the expanded scale to the common slope.
            scale = float(np.exp(items[0, 0]))
            locations = self._optimize_items(
                scale * items[:, 1:2], "1PL", scale * self.nodes, posterior
            )
            items = np.column_stack([np.zeros(items.shape[0]), locations])

        mean = scale * (posterior @ self.nodes)
        variance = np.maximum(scale**2 * (posterior @ self.nodes**2) - mean**2, 0.0)
        coefficients, _, _, _ = np.linalg.lstsq(self.design, mean, rcond=None)
        residual = mean - self.design @ coefficients
        residual_variance = float(np.mean(variance + residual**2))
        residual_variance = max(residual_variance, _MIN_RESIDUAL_VARIANCE)
        return self._standardize(items, coefficients, residual_variance)

    def _standardize(
        self,
        items: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        residual_variance: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Move the ability scale back to mean 0 (and total variance 1)."""
        log_slope_bounds, difficulty_bounds = self._bounds("2PL")
        predicted = self.design @ coefficients
        location = float(predicted.mean())
        scale = 1.0
        if self.free_scale:
            scale = float(np.sqrt(predicted.var() + residual_variance))
        coefficients = coefficients / scale
        coefficients[0] -= location / scale
        items = items.copy()
        items[:, 1] = np.clip((items[:, 1] - location) / scale, *difficulty_bounds)
        if self.free_scale:
            items[:, 0] = np.clip(items[:, 0] + np.log(scale), *log_slope_bounds)
        return items, coefficients, residual_variance / scale**2

    def fit(
        self,
        start_model: BaseItemModel,
        *,
        max_iter: int,
        tol: float,
        verbose: bool,
    ) -> _LatentRegressionFit:
        """Run EM from the base-model items with zero covariate effects."""
        if self.base_model == "1PL" and getattr(
            start_model, "estimate_discrimination", False
        ):
            self.slopes = "common"
        items = self._item_values(start_model)
        coefficients = np.zeros(self.design.shape[1])
        residual_variance = 1.0
        previous = -np.inf
        converged = False
        iteration = 0
        for iteration in range(max_iter):
            posterior, log_marginal = self._posterior(
                items, self.design @ coefficients, residual_variance
            )
            log_likelihood = float(np.sum(log_marginal))
            if verbose:
                print(f"Iteration {iteration:4d}: LL = {log_likelihood:.4f}")
            if abs(log_likelihood - previous) < tol:
                converged = True
                break
            previous = log_likelihood
            items, coefficients, residual_variance = self._m_step(items, posterior)
        else:
            posterior, log_marginal = self._posterior(
                items, self.design @ coefficients, residual_variance
            )
            log_likelihood = float(np.sum(log_marginal))
            converged = abs(log_likelihood - previous) < tol

        effects = coefficients[1:].copy()
        covariance = self._effect_covariance(items, effects, residual_variance)
        effect_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        intercept_variance = float(
            self.covariate_mean @ covariance @ self.covariate_mean
        )
        theta = posterior @ self.nodes
        theta_se = np.sqrt(np.maximum(posterior @ self.nodes**2 - theta**2, 0.0))
        return _LatentRegressionFit(
            model=self._fitted_model(start_model, items),
            intercept=float(coefficients[0]),
            effects=effects,
            intercept_se=float(np.sqrt(max(intercept_variance, 0.0))),
            effect_se=effect_se,
            residual_variance=float(residual_variance),
            log_likelihood=log_likelihood,
            converged=converged,
            n_iterations=iteration + 1,
            # The regression adds its slopes; the residual variance is
            # determined by the unit total variance unless the slopes are
            # fixed. The intercept centers the predicted abilities.
            n_parameters=effects.size + int(self.slopes == "fixed"),
            theta=theta,
            theta_se=theta_se,
        )

    def _fitted_model(
        self, start_model: BaseItemModel, items: NDArray[np.float64]
    ) -> BaseItemModel:
        parameters = {"difficulty": items[:, 1].copy()}
        if self.free_scale:
            parameters["discrimination"] = np.exp(items[:, 0])
        if self.base_model == "3PL":
            parameters["guessing"] = items[:, 2].copy()
        model = start_model.copy()
        model.set_parameters(**parameters)
        model._is_fitted = True
        return model

    def _unpack(
        self, vector: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return items, effects and residual variance from SE coordinates.

        Free per-item coordinates come first: every ``(log a, b[, c])`` for
        the 2PL and 3PL, the common log-slope and the difficulties for a 1PL
        with a free slope, and only the difficulties otherwise. The effects
        follow; a 1PL with a fixed slope ends with the log residual
        variance, which is otherwise ``1 - var(x' beta)``. The intercept
        centers the predicted abilities.
        """
        n_items = self.correct.shape[1]
        n_effects = self.covariates.shape[1]
        if self.slopes == "item":
            width = 3 if self.base_model == "3PL" else 2
            n_item_values = n_items * width
            items = vector[:n_item_values].reshape(n_items, width)
        else:
            n_item_values = n_items + int(self.slopes == "common")
            log_slope = vector[0] if self.slopes == "common" else 0.0
            items = np.column_stack(
                [
                    np.full(n_items, log_slope),
                    vector[n_item_values - n_items : n_item_values],
                ]
            )
        effects = vector[n_item_values : n_item_values + n_effects]
        if self.free_scale:
            residual_variance = 1.0 - float(
                effects @ self.covariate_covariance @ effects
            )
        else:
            residual_variance = float(np.exp(vector[-1]))
        return items, effects, residual_variance

    def _pack(
        self,
        items: NDArray[np.float64],
        effects: NDArray[np.float64],
        residual_variance: float,
    ) -> NDArray[np.float64]:
        if self.slopes == "item":
            parts = [items.ravel()]
        elif self.slopes == "common":
            parts = [items[:1, 0], items[:, 1]]
        else:
            parts = [items[:, 1]]
        parts.append(effects)
        if not self.free_scale:
            parts.append(np.array([np.log(residual_variance)]))
        return np.concatenate(parts)

    def _score(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the marginal score by Fisher's identity."""
        from mirt.models.mixture import _item_expected_objective

        items, effects, residual_variance = self._unpack(vector)
        if residual_variance <= 0.0:
            return np.full(vector.size, np.nan)
        prior_mean = self.centered @ effects
        posterior, _ = self._posterior(items, prior_mean, residual_variance)

        expected_correct = posterior.T @ self.correct
        expected_total = posterior.T @ self.observed
        item_scores = np.empty_like(items)
        for item in range(items.shape[0]):
            _, gradient = _item_expected_objective(
                items[item],
                self.gradient_family,
                self.nodes,
                expected_correct[:, item],
                expected_total[:, item],
            )
            item_scores[item] = -gradient

        mean = posterior @ self.nodes
        residual = mean - prior_mean
        squared = np.maximum(posterior @ self.nodes**2 - mean**2, 0.0) + residual**2
        mean_score = self.centered.T @ residual / residual_variance
        variance_score = float(
            np.sum(-0.5 / residual_variance + 0.5 * squared / residual_variance**2)
        )
        if self.slopes == "item":
            parts = [item_scores.ravel()]
        elif self.slopes == "common":
            parts = [[item_scores[:, 0].sum()], item_scores[:, 1]]
        else:
            parts = [item_scores[:, 1]]
        if self.free_scale:
            # d sigma^2 / d beta = -2 S beta under the unit total variance.
            explained = self.covariate_covariance @ effects
            parts.append(mean_score - 2.0 * variance_score * explained)
        else:
            parts.extend([mean_score, [variance_score * residual_variance]])
        return np.concatenate([np.asarray(part, dtype=np.float64) for part in parts])

    def _effect_covariance(
        self,
        items: NDArray[np.float64],
        effects: NDArray[np.float64],
        residual_variance: float,
    ) -> NDArray[np.float64]:
        """Invert the observed information from central score differences."""
        vector = self._pack(items, effects, residual_variance)
        hessian = np.empty((vector.size, vector.size))
        for index in range(vector.size):
            step = 1e-5 * max(1.0, abs(vector[index]))
            upper = vector.copy()
            lower = vector.copy()
            upper[index] += step
            lower[index] -= step
            hessian[:, index] = (self._score(upper) - self._score(lower)) / (2.0 * step)
        information = -(hessian + hessian.T) * 0.5
        n_effects = effects.size
        offset = vector.size - n_effects - int(not self.free_scale)
        effect_slice = slice(offset, offset + n_effects)
        if not np.all(np.isfinite(information)):
            return np.full((n_effects, n_effects), np.nan)
        try:
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(information)
        block = covariance[effect_slice, effect_slice]
        if np.any(np.diag(block) <= 0.0):
            return np.full((n_effects, n_effects), np.nan)
        return block


class _RaschMarginalObjective:
    """Marginal 1PL log-likelihood with one slope and ``b = design @ eta``.

    Coefficients are ``(log a, eta)``. The common slope ``a`` of a
    standard-normal ability is equivalent to a free latent variance with unit
    slopes, the Rasch model of R mirt.
    """

    def __init__(
        self,
        responses: NDArray[np.int_],
        nodes: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        observed = responses >= 0
        self.correct = np.where(observed, responses, 0).astype(np.float64)
        self.observed = observed.astype(np.float64)
        self.incorrect = self.observed - self.correct
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.log_weights = np.log(np.asarray(weights, dtype=np.float64))

    def negative_log_likelihood(
        self, coefficients: NDArray[np.float64], design: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative marginal log-likelihood and its gradient."""
        slope = float(np.exp(coefficients[0]))
        difficulty = design @ coefficients[1:]
        centered = self.nodes[:, None] - difficulty[None, :]
        logits = slope * centered
        joint = (
            self.correct @ -np.logaddexp(0.0, -logits).T
            + self.incorrect @ -np.logaddexp(0.0, logits).T
            + self.log_weights[None, :]
        )
        log_marginal = logsumexp(joint, axis=1)
        posterior = np.exp(joint - log_marginal[:, None])
        # Fisher's identity: the marginal score is the posterior-expected
        # complete-data score, built from the expected counts r_qj and n_qj.
        expected_correct = posterior.T @ self.correct
        expected_total = posterior.T @ self.observed
        logit_score = expected_correct - expected_total * expit(logits)
        difficulty_score = -slope * np.sum(logit_score, axis=0)
        log_slope_score = float(np.sum(logit_score * logits))
        gradient = np.concatenate([[log_slope_score], design.T @ difficulty_score])
        return -float(np.sum(log_marginal)), -gradient

    def maximize(
        self, design: NDArray[np.float64], start: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool]:
        """Return the maximizing coefficients, log-likelihood and status."""
        start = np.asarray(start, dtype=np.float64)
        result = optimize.minimize(
            self.negative_log_likelihood,
            start,
            args=(design,),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-6},
        )
        start_value, _ = self.negative_log_likelihood(start, design)
        if not np.isfinite(result.fun) or result.fun > start_value:
            return start, -start_value, False
        return result.x, -float(result.fun), bool(result.success)
