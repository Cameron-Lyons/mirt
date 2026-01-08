import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


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
                return 1.0 / (1.0 + np.exp(-z))

            z = a[None, :] * (theta_1d[:, None] - b[None, :])
            return 1.0 / (1.0 + np.exp(-z))

        else:
            if item_idx is not None:
                z = np.dot(theta, a[item_idx]) - a[item_idx].sum() * b[item_idx]
                return 1.0 / (1.0 + np.exp(-z))

            z = np.dot(theta, a.T) - np.sum(a, axis=1) * b
            return 1.0 / (1.0 + np.exp(-z))

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
            p_star = 1.0 / (1.0 + np.exp(-z))
            return c[item_idx] + (1.0 - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = 1.0 / (1.0 + np.exp(-z))
        return c[None, :] + (1.0 - c[None, :]) * p_star

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        a = self._parameters["discrimination"]
        c = self._parameters["guessing"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            numerator = (a_val**2) * ((p - c_val) ** 2)
            denominator = ((1 - c_val) ** 2) * p * (1 - p) + 1e-10
            return numerator / denominator

        numerator = (a[None, :] ** 2) * ((p - c[None, :]) ** 2)
        denominator = ((1 - c[None, :]) ** 2) * p * (1 - p) + 1e-10
        return numerator / denominator


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
            p_star = 1.0 / (1.0 + np.exp(-z))
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_star = 1.0 / (1.0 + np.exp(-z))
        return c[None, :] + (d[None, :] - c[None, :]) * p_star

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        a = self._parameters["discrimination"]
        c = self._parameters["guessing"]
        d = self._parameters["upper"]

        if item_idx is not None:
            a_val = a[item_idx]
            c_val = c[item_idx]
            d_val = d[item_idx]
            numerator = (a_val**2) * ((p - c_val) ** 2) * ((d_val - p) ** 2)
            denominator = ((d_val - c_val) ** 2) * p * (1 - p) + 1e-10
            return numerator / denominator

        numerator = (
            (a[None, :] ** 2) * ((p - c[None, :]) ** 2) * ((d[None, :] - p) ** 2)
        )
        denominator = ((d[None, :] - c[None, :]) ** 2) * p * (1 - p) + 1e-10
        return numerator / denominator


Rasch = OneParameterLogistic


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
            logistic = 1.0 / (1.0 + np.exp(-z))
            p_star = np.power(logistic, e[item_idx])
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        logistic = 1.0 / (1.0 + np.exp(-z))
        p_star = np.power(logistic, e[None, :])
        return c[None, :] + (d[None, :] - c[None, :]) * p_star

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        # Numerical derivative for information
        h = 1e-5
        theta_plus = theta + h
        theta_minus = theta - h

        p_plus = self.probability(theta_plus, item_idx)
        p_minus = self.probability(theta_minus, item_idx)

        dp = (p_plus - p_minus) / (2 * h)

        # Information = (dp/dtheta)^2 / (p * q)
        return (dp**2) / (p * (1 - p) + 1e-10)


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
            return 1.0 - np.exp(-np.exp(z))

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return 1.0 - np.exp(-np.exp(z))

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
            exp_z = np.exp(z)
            exp_neg_exp_z = np.exp(-exp_z)
            p = 1.0 - exp_neg_exp_z
            q = exp_neg_exp_z

            # Derivative of CLL: dp/dtheta = a * exp(z) * exp(-exp(z))
            dp = a[item_idx] * exp_z * exp_neg_exp_z

            return (dp**2) / (p * q + 1e-10)

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        exp_z = np.exp(z)
        exp_neg_exp_z = np.exp(-exp_z)
        p = 1.0 - exp_neg_exp_z
        q = exp_neg_exp_z

        dp = a[None, :] * exp_z * exp_neg_exp_z

        return (dp**2) / (p * q + 1e-10)


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
            return np.exp(-np.exp(z))

        z = -a[None, :] * (theta_1d[:, None] - b[None, :])
        return np.exp(-np.exp(z))

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
            exp_z = np.exp(z)
            p = np.exp(-exp_z)
            q = 1.0 - p

            dp = a[item_idx] * exp_z * p

            return (dp**2) / (p * q + 1e-10)

        z = -a[None, :] * (theta_1d[:, None] - b[None, :])
        exp_z = np.exp(z)
        p = np.exp(-exp_z)
        q = 1.0 - p

        dp = a[None, :] * exp_z * p

        return (dp**2) / (p * q + 1e-10)
