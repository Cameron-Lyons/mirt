from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.exceptions import MirtValidationError
from mirt.models.base import DichotomousItemModel

_MIN_EXP_INPUT = -745.0
_MAX_DOUBLE_EXP_INPUT = 50.0


def _fisher_information(
    probability: NDArray[np.float64],
    derivative: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Bernoulli Fisher information without unstable tail division."""
    denominator = probability * (1.0 - probability)
    return np.divide(
        derivative**2,
        denominator,
        out=np.zeros_like(probability, dtype=np.float64),
        where=denominator > 0,
    )


def _bounded_exponential(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate safely for links containing a second exponential."""
    return np.exp(np.clip(value, _MIN_EXP_INPUT, _MAX_DOUBLE_EXP_INPUT))


class _ParameterizedDichotomousModel(DichotomousItemModel):
    """Shared parameter-domain validation for dichotomous response curves."""

    _requires_positive_discrimination = False

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        for name, values in parameters.items():
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                    expected="finite values",
                )

        discrimination = parameters["discrimination"]
        if self._requires_positive_discrimination and np.any(discrimination <= 0.0):
            raise MirtValidationError(
                "discrimination must be strictly positive",
                parameter="discrimination",
                value=discrimination,
                expected="> 0",
            )

        guessing = parameters.get("guessing")
        if guessing is not None and np.any((guessing < 0.0) | (guessing >= 1.0)):
            raise MirtValidationError(
                "guessing must be in [0, 1)",
                parameter="guessing",
                value=guessing,
                expected="[0, 1)",
            )

        upper = parameters.get("upper")
        if upper is not None:
            if np.any((upper < 0.0) | (upper > 1.0)):
                raise MirtValidationError(
                    "upper must be in [0, 1]",
                    parameter="upper",
                    value=upper,
                    expected="[0, 1]",
                )
            if guessing is not None and np.any(guessing > upper):
                raise MirtValidationError(
                    "guessing cannot exceed upper",
                    parameter="guessing",
                    value=guessing,
                    expected="guessing <= upper",
                )

        asymmetry = parameters.get("asymmetry")
        if asymmetry is not None and np.any(asymmetry <= 0.0):
            raise MirtValidationError(
                "asymmetry must be strictly positive",
                parameter="asymmetry",
                value=asymmetry,
                expected="> 0",
            )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters atomically after validating the complete model state."""
        candidate = {name: values.copy() for name, values in self._parameters.items()}
        for name, value in params.items():
            if name not in candidate:
                valid_params = ", ".join(candidate)
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}",
                    parameter=name,
                    expected=valid_params,
                )

            value_array = np.asarray(value, dtype=np.float64)
            expected_shape = candidate[name].shape
            if value_array.shape != expected_shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {expected_shape}, "
                    f"got {value_array.shape}",
                    parameter=name,
                    value=value_array.shape,
                    expected=str(expected_shape),
                )
            candidate[name] = value_array.copy()

        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item parameter while preserving the model's domain."""
        item_idx = self._validate_item_idx(item_idx)
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        current = self._parameters[param_name]
        value_array = np.asarray(value, dtype=np.float64)
        expected_shape = current.shape[1:]
        if value_array.shape != expected_shape:
            expected = "scalar" if not expected_shape else str(expected_shape)
            raise MirtValidationError(
                f"{param_name} for one item must have shape {expected}",
                parameter=param_name,
                value=value_array.shape,
                expected=expected,
            )

        updated = current.copy()
        updated[item_idx] = value_array
        self.set_parameters(**{param_name: updated})

    def _validate_item_idx(self, item_idx: int) -> int:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
            item_idx, (int, np.integer)
        ):
            raise IndexError("item_idx must be an integer")
        item_idx = int(item_idx)
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return item_idx

    def _unidimensional_pair_logits(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Gather parameters and evaluate aligned unidimensional logits."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        discrimination = self._parameters["discrimination"][indices]
        difficulty = self._parameters["difficulty"][indices]
        return indices, discrimination * (theta_2d[:, 0] - difficulty)


class TwoParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "2PL"
    n_params_per_item = 2
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if self.n_factors == 1:
            theta_1d = theta.ravel()

            if item_idx is not None:
                z = a[item_idx] * (theta_1d - b[item_idx])
                return sigmoid(z)

            z = a[None, :] * (theta_1d[:, None] - b[None, :])
            return sigmoid(z)

        else:
            if item_idx is not None:
                z = np.dot(theta, a[item_idx]) - a[item_idx].sum() * b[item_idx]
                return sigmoid(z)

            z = np.dot(theta, a.T) - np.sum(a, axis=1) * b
            return sigmoid(z)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        discrimination = self._parameters["discrimination"][indices]
        difficulty = self._parameters["difficulty"][indices]
        if self.n_factors == 1:
            logits = discrimination * (theta_2d[:, 0] - difficulty)
        else:
            logits = np.einsum("ij,ij->i", theta_2d, discrimination)
            logits -= np.sum(discrimination, axis=1) * difficulty
        return sigmoid(logits)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            if self.n_factors == 1:
                a_val = a[item_idx]
            else:
                a_val = np.sqrt(np.sum(a[item_idx] ** 2))
            return (a_val**2) * p * q

        if self.n_factors == 1:
            return (a[None, :] ** 2) * p * q
        else:
            a_sq = np.sum(a**2, axis=1)
            return a_sq[None, :] * p * q


class OneParameterLogistic(TwoParameterLogistic):
    model_name = "1PL"
    n_params_per_item = 1
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("1PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["discrimination"] = np.zeros_like(self.discrimination, dtype=np.bool_)
        return masks

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "discrimination":
            canonical.fill(1.0)
        return canonical

    def set_parameters(self, **params: NDArray[np.float64]) -> "OneParameterLogistic":
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in 1PL model (fixed to 1)")
        return super().set_parameters(**params)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        if param_name == "discrimination":
            self._validate_item_idx(item_idx)
            value_array = np.asarray(value, dtype=np.float64)
            if value_array.ndim == 0 and float(value_array) == 1.0:
                return
            raise ValueError("Cannot set discrimination in 1PL model (fixed to 1)")
        super().set_item_parameter(item_idx, param_name, value)


class ThreeParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "3PL"
    n_params_per_item = 3
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("3PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_star = sigmoid(z)
            return c[item_idx] + (1.0 - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = sigmoid(z)
        return c[None, :] + (1.0 - c[None, :]) * p_star

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        indices, logits = self._unidimensional_pair_logits(theta, item_indices)
        guessing = self._parameters["guessing"][indices]
        logistic = sigmoid(logits)
        return guessing + (1.0 - guessing) * logistic

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            logistic = sigmoid(a_val * (theta_1d - b[item_idx]))
            probability = c_val + (1.0 - c_val) * logistic
            derivative = a_val * (1.0 - c_val) * logistic * (1.0 - logistic)
            return _fisher_information(probability, derivative)

        logistic = sigmoid(a[None, :] * (theta_1d[:, None] - b[None, :]))
        probability = c[None, :] + (1.0 - c[None, :]) * logistic
        derivative = a[None, :] * (1.0 - c[None, :]) * logistic * (1.0 - logistic)
        return _fisher_information(probability, derivative)


class FourParameterLogistic(_ParameterizedDichotomousModel):
    model_name = "4PL"
    n_params_per_item = 4
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("4PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)
        self._parameters["upper"] = np.ones(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_star = sigmoid(z)
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = sigmoid(z)
        return c[None, :] + (d[None, :] - c[None, :]) * p_star

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        indices, logits = self._unidimensional_pair_logits(theta, item_indices)
        guessing = self._parameters["guessing"][indices]
        upper = self._parameters["upper"][indices]
        return guessing + (upper - guessing) * sigmoid(logits)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            d_val = d[item_idx]
            logistic = sigmoid(a_val * (theta_1d - b[item_idx]))
            probability = c_val + (d_val - c_val) * logistic
            derivative = a_val * (d_val - c_val) * logistic * (1.0 - logistic)
            return _fisher_information(probability, derivative)

        logistic = sigmoid(a[None, :] * (theta_1d[:, None] - b[None, :]))
        probability = c[None, :] + (d[None, :] - c[None, :]) * logistic
        derivative = (
            a[None, :] * (d[None, :] - c[None, :]) * logistic * (1.0 - logistic)
        )
        return _fisher_information(probability, derivative)


Rasch = OneParameterLogistic

ThreeParameterLogisticUpper = FourParameterLogistic


class UnipolarLogLogistic(_ParameterizedDichotomousModel):
    """Unipolar Log-Logistic (ULL) model for dichotomous items.

    The ULL model is designed for items where only positive trait levels
    are expected to endorse the item. It has a lower asymptote at 0 and
    approaches 1 more slowly than the standard logistic.

    This is useful for clinical or personality assessment where items
    measure presence/absence of a trait that only manifests at higher
    trait levels.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters (must be positive)
    difficulty : ndarray
        Item difficulty/location parameters

    Notes
    -----
    The ULL probability function is:

        P(X=1|θ) = exp(a(θ - b)) / (1 + exp(a(θ - b)))^2

    which is the derivative of the logistic function, giving a
    bell-shaped response function peaking near b.

    Alternatively, using the log-logistic formulation:

        P(X=1|θ) = 1 / (1 + exp(-a(θ - b)))  for θ >= b
        P(X=1|θ) ≈ 0                          for θ << b

    References
    ----------
    Samejima, F. (1995). Acceleration model in the heterogeneous case
        of the general graded response model. Psychometrika, 60, 549-572.
    """

    model_name = "ULL"
    n_params_per_item = 2
    supports_multidimensional = False
    _requires_positive_discrimination = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("ULL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            logistic = sigmoid(z)
            return logistic * (1.0 - logistic)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        logistic = sigmoid(z)
        return logistic * (1.0 - logistic)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        _, logits = self._unidimensional_pair_logits(theta, item_indices)
        logistic = sigmoid(logits)
        return logistic * (1.0 - logistic)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            logistic = sigmoid(a[item_idx] * (theta_1d - b[item_idx]))
            probability = logistic * (1.0 - logistic)
            derivative = a[item_idx] * probability * (1.0 - 2.0 * logistic)
            return _fisher_information(probability, derivative)

        logistic = sigmoid(a[None, :] * (theta_1d[:, None] - b[None, :]))
        probability = logistic * (1.0 - logistic)
        derivative = a[None, :] * probability * (1.0 - 2.0 * logistic)
        return _fisher_information(probability, derivative)


class FiveParameterLogistic(_ParameterizedDichotomousModel):
    """Five-Parameter Logistic (5PL) model with asymmetric curves.

    The 5PL model extends the 4PL with an asymmetry parameter that allows
    the IRF to have different slopes in the lower and upper regions.
    This is useful when item characteristics vary across the ability range.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination (slope) parameters
    difficulty : ndarray
        Item difficulty (location) parameters
    guessing : ndarray
        Lower asymptote (guessing) parameters
    upper : ndarray
        Upper asymptote parameters
    asymmetry : ndarray
        Asymmetry parameters (> 1 steeper on right, < 1 steeper on left)

    Notes
    -----
    The 5PL probability function is:

        P(X=1|θ) = c + (d - c) / (1 + exp(-a(θ - b)))^e

    where e is the asymmetry parameter.

    References
    ----------
    Reise, S. P., & Waller, N. G. (2003). How many IRT parameters does it
        take to model psychopathology items? Psychological Methods.
    """

    model_name = "5PL"
    n_params_per_item = 5
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("5PL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)
        self._parameters["guessing"] = np.full(self.n_items, 0.2)
        self._parameters["upper"] = np.ones(self.n_items)
        self._parameters["asymmetry"] = np.ones(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    @property
    def asymmetry(self) -> NDArray[np.float64]:
        return self._parameters["asymmetry"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]
        e = self._parameters["asymmetry"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            logistic = sigmoid(z)
            p_star = np.power(logistic, e[item_idx])
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        logistic = sigmoid(z)
        p_star = np.power(logistic, e[None, :])
        return c[None, :] + (d[None, :] - c[None, :]) * p_star

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        indices, logits = self._unidimensional_pair_logits(theta, item_indices)
        guessing = self._parameters["guessing"][indices]
        upper = self._parameters["upper"][indices]
        asymmetry = self._parameters["asymmetry"][indices]
        powered = np.power(sigmoid(logits), asymmetry)
        return guessing + (upper - guessing) * powered

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]
        e = self._parameters["asymmetry"]

        if item_idx is not None:
            logistic = sigmoid(a[item_idx] * (theta_1d - b[item_idx]))
            powered = np.power(logistic, e[item_idx])
            probability = c[item_idx] + (d[item_idx] - c[item_idx]) * powered
            derivative = (
                a[item_idx]
                * e[item_idx]
                * (d[item_idx] - c[item_idx])
                * powered
                * (1.0 - logistic)
            )
            return _fisher_information(probability, derivative)

        logistic = sigmoid(a[None, :] * (theta_1d[:, None] - b[None, :]))
        powered = np.power(logistic, e[None, :])
        probability = c[None, :] + (d[None, :] - c[None, :]) * powered
        derivative = (
            a[None, :]
            * e[None, :]
            * (d[None, :] - c[None, :])
            * powered
            * (1.0 - logistic)
        )
        return _fisher_information(probability, derivative)


class ComplementaryLogLog(_ParameterizedDichotomousModel):
    """Complementary Log-Log (CLL) model for dichotomous items.

    The CLL model uses an asymmetric link function instead of the
    symmetric logistic. This is useful when the probability curve
    should approach 0 and 1 at different rates.

    Parameters
    ----------
    n_items : int
        Number of items
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray
        Item discrimination parameters
    difficulty : ndarray
        Item difficulty parameters

    Notes
    -----
    The CLL probability function is:

        P(X=1|θ) = 1 - exp(-exp(a(θ - b)))

    The CLL function approaches 0 slowly and 1 quickly.

    For slow approach to 1 and fast to 0, use the negative-log-log
    (NLL) variant: P = exp(-exp(-a(θ - b)))
    """

    model_name = "CLL"
    n_params_per_item = 2
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("CLL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            exp_z = _bounded_exponential(z)
            return -np.expm1(-exp_z)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        exp_z = _bounded_exponential(z)
        return -np.expm1(-exp_z)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        _, logits = self._unidimensional_pair_logits(theta, item_indices)
        return -np.expm1(-_bounded_exponential(logits))

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            exp_z = _bounded_exponential(z)
            exp_neg_exp_z = np.exp(-exp_z)
            probability = -np.expm1(-exp_z)
            derivative = a[item_idx] * exp_z * exp_neg_exp_z
            return _fisher_information(probability, derivative)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        exp_z = _bounded_exponential(z)
        exp_neg_exp_z = np.exp(-exp_z)
        probability = -np.expm1(-exp_z)
        derivative = a[None, :] * exp_z * exp_neg_exp_z
        return _fisher_information(probability, derivative)


class NegativeLogLog(_ParameterizedDichotomousModel):
    """Negative Log-Log (NLL) model for dichotomous items.

    The NLL model is the mirror image of CLL, approaching 1 slowly
    and 0 quickly.

    Notes
    -----
    The NLL probability function is:

        P(X=1|θ) = exp(-exp(-a(θ - b)))
    """

    model_name = "NLL"
    n_params_per_item = 2
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("NLL model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = -a[item_idx] * (theta_1d - b[item_idx])
            return np.exp(-_bounded_exponential(z))

        z = -a[None, :] * (theta_1d[:, None] - b[None, :])
        return np.exp(-_bounded_exponential(z))

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in one vectorized pass."""
        _, logits = self._unidimensional_pair_logits(theta, item_indices)
        return np.exp(-_bounded_exponential(-logits))

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = -a[item_idx] * (theta_1d - b[item_idx])
            exp_z = _bounded_exponential(z)
            probability = np.exp(-exp_z)
            derivative = a[item_idx] * exp_z * probability
            return _fisher_information(probability, derivative)

        z = -a[None, :] * (theta_1d[:, None] - b[None, :])
        exp_z = _bounded_exponential(z)
        probability = np.exp(-exp_z)
        derivative = a[None, :] * exp_z * probability
        return _fisher_information(probability, derivative)
