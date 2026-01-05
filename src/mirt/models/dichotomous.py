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

    def set_parameters(self, **params: NDArray[np.float64]) -> OneParameterLogistic:
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
