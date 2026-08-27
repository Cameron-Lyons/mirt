import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
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


class TwoParameterLogistic(DichotomousItemModel):
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


class ThreeParameterLogistic(DichotomousItemModel):
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


class FourParameterLogistic(DichotomousItemModel):
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


class UnipolarLogLogistic(DichotomousItemModel):
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


class FiveParameterLogistic(DichotomousItemModel):
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


class ComplementaryLogLog(DichotomousItemModel):
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


class NegativeLogLog(DichotomousItemModel):
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
