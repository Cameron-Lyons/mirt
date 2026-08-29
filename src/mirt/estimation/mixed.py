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

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


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
        Log-likelihood at estimates
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
        theta_i = beta_0 + beta_1 * Z_i1 + ... + beta_p * Z_ip + epsilon_i

    For item covariates (LLTM-style):
        b_j = gamma_0 + gamma_1 * W_j1 + ... + gamma_q * W_jq

    where:
    - Z: Person covariates (n_persons x p)
    - W: Item covariates (n_items x q)
    - beta: Person covariate effects
    - gamma: Item covariate effects

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
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> MixedEffectsFitResult:
        """Fit mixed effects IRT model.

        Uses a two-stage approach:
        1. Fit base IRT model
        2. Estimate covariate effects

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        max_iter : int
            Maximum iterations for EM
        tol : float
            Convergence tolerance
        verbose : bool
            Whether to print progress

        Returns
        -------
        MixedEffectsFitResult
            Estimation results
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
        )
        score_result = fscores(base_result.model, responses, method="EAP")
        theta_estimates = score_result.theta
        theta_se = score_result.standard_error

        if theta_estimates.ndim == 2:
            theta_estimates = theta_estimates[:, 0]
        if theta_se.ndim == 2:
            theta_se = theta_se[:, 0]

        if verbose:
            print("Stage 2: Estimating covariate effects...")

        person_effects = None
        person_se = None
        item_effects = None
        item_se = None
        person_intercept_se = np.nan
        item_intercept_se = np.nan

        if self._person_covariates is not None:
            (
                self._person_intercept,
                person_effects,
                person_intercept_se,
                person_se,
                self._residual_variance,
            ) = self._estimate_person_effects(theta_estimates)
            self._person_effects = person_effects

        if self._item_covariates is not None:
            (
                self._item_intercept,
                item_effects,
                item_intercept_se,
                item_se,
                _,
            ) = self._estimate_item_effects(base_result.model)
            self._item_effects = item_effects

        ll = base_result.log_likelihood

        if person_effects is not None:
            predicted_theta = (
                self._person_intercept + self._person_covariates @ person_effects
            )
            residuals = theta_estimates - predicted_theta
            ll += float(
                np.sum(
                    -0.5 * np.log(2 * np.pi * self._residual_variance)
                    - 0.5 * residuals**2 / self._residual_variance
                )
            )

        n_params = int(
            getattr(base_result, "n_parameters", base_result.model.n_parameters)
        )
        if person_effects is not None:
            n_params += len(person_effects) + 2
        if item_effects is not None:
            n_params += len(item_effects) + 1

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + np.log(n_persons) * n_params

        return MixedEffectsFitResult(
            model=base_result.model,
            person_effects=person_effects,
            item_effects=item_effects,
            log_likelihood=ll,
            aic=float(aic),
            bic=float(bic),
            converged=base_result.converged,
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

    def _estimate_person_effects(
        self,
        theta: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], float, NDArray[np.float64], float]:
        """Estimate effects of person covariates on theta."""
        return self._fit_linear_effects(self._person_covariates, theta)

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
    ) -> dict[str, Any]:
        """Fit LLTM model.

        Parameters
        ----------
        responses : NDArray
            Response matrix
        verbose : bool
            Whether to print progress

        Returns
        -------
        dict
            Fit results including eta parameters
        """
        from mirt import fit_mirt

        responses = np.asarray(responses)
        n_persons, n_items = responses.shape

        if self._q_matrix.shape[0] != n_items:
            raise ValueError(
                f"Q-matrix has {self._q_matrix.shape[0]} rows but data has {n_items} items"
            )

        rasch_result = fit_mirt(responses, model="1PL", verbose=False)
        b_rasch = rasch_result.model.parameters["difficulty"]

        self._eta, _, _, _ = np.linalg.lstsq(self._q_matrix, b_rasch, rcond=None)

        b_lltm = self._q_matrix @ self._eta

        from mirt.models.dichotomous import OneParameterLogistic

        lltm_model = OneParameterLogistic(n_items=n_items)
        lltm_model._initialize_parameters()
        lltm_model._parameters["difficulty"] = b_lltm
        lltm_model._is_fitted = True

        from mirt.scoring import fscores

        scores = fscores(lltm_model, responses, method="EAP")
        ll = float(
            np.sum(lltm_model.log_likelihood(responses, scores.theta.reshape(-1, 1)))
        )

        n_params_rasch = n_items
        n_params_lltm = self._q_matrix.shape[1]

        aic_rasch = -2 * rasch_result.log_likelihood + 2 * n_params_rasch
        aic_lltm = -2 * ll + 2 * n_params_lltm

        bic_rasch = (
            -2 * rasch_result.log_likelihood + np.log(n_persons) * n_params_rasch
        )
        bic_lltm = -2 * ll + np.log(n_persons) * n_params_lltm

        chi_sq = 2 * (rasch_result.log_likelihood - ll)
        df = n_params_rasch - n_params_lltm
        from scipy import stats

        p_value = 1 - stats.chi2.cdf(chi_sq, df) if df > 0 else np.nan

        return {
            "eta": self._eta,
            "difficulty_rasch": b_rasch,
            "difficulty_lltm": b_lltm,
            "log_likelihood_rasch": rasch_result.log_likelihood,
            "log_likelihood_lltm": ll,
            "aic_rasch": float(aic_rasch),
            "aic_lltm": float(aic_lltm),
            "bic_rasch": float(bic_rasch),
            "bic_lltm": float(bic_lltm),
            "chi_sq": float(chi_sq),
            "df": df,
            "p_value": float(p_value),
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
