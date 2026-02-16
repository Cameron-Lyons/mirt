"""Custom item type creation for IRT models.

This module provides a flexible API for creating user-defined item types
with custom Item Characteristic Curves (ICC), information functions, and
parameter specifications.

This is equivalent to R's mirt::createItem() function.

Examples
--------
>>> # Create a custom 2PL-like item
>>> def my_icc(theta, a, b):
...     return 1 / (1 + np.exp(-a * (theta - b)))
>>>
>>> def my_info(theta, a, b):
...     p = my_icc(theta, a, b)
...     return a**2 * p * (1 - p)
>>>
>>> MyItem = create_item_type(
...     name="My2PL",
...     icc_function=my_icc,
...     info_function=my_info,
...     par_names=["a", "b"],
...     par_bounds={"a": (0.01, 5.0), "b": (-5.0, 5.0)},
...     par_defaults={"a": 1.0, "b": 0.0},
... )
>>>
>>> # Use in a model
>>> model = CustomItemModel(n_items=10, item_type=MyItem)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    pass


@dataclass
class ItemTypeSpec:
    """Specification for a custom item type.

    Parameters
    ----------
    name : str
        Name of the item type
    icc_function : callable
        Function computing P(X=1|theta) for dichotomous or
        P(X=k|theta) for polytomous items.
        Signature: f(theta, **params) -> probabilities
    info_function : callable, optional
        Function computing item information.
        Signature: f(theta, **params) -> information
        If not provided, numerical differentiation is used.
    par_names : list[str]
        Names of item parameters
    par_bounds : dict[str, tuple]
        Bounds for each parameter as (lower, upper)
    par_defaults : dict[str, float]
        Default values for each parameter
    n_categories : int
        Number of response categories (2 for dichotomous)
    gradient_function : callable, optional
        Function computing gradients for optimization.
        Signature: f(theta, **params) -> dict of gradients
    """

    name: str
    icc_function: Callable[..., NDArray[np.float64]]
    info_function: Callable[..., NDArray[np.float64]] | None = None
    par_names: list[str] = field(default_factory=list)
    par_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    par_defaults: dict[str, float] = field(default_factory=dict)
    n_categories: int = 2
    gradient_function: Callable[..., dict[str, NDArray[np.float64]]] | None = None

    def __post_init__(self):
        for name in self.par_names:
            if name not in self.par_bounds:
                self.par_bounds[name] = (-np.inf, np.inf)
            if name not in self.par_defaults:
                self.par_defaults[name] = 0.0


def create_item_type(
    name: str,
    icc_function: Callable[..., NDArray[np.float64]],
    info_function: Callable[..., NDArray[np.float64]] | None = None,
    par_names: list[str] | None = None,
    par_bounds: dict[str, tuple[float, float]] | None = None,
    par_defaults: dict[str, float] | None = None,
    n_categories: int = 2,
    gradient_function: Callable[..., dict[str, NDArray[np.float64]]] | None = None,
) -> ItemTypeSpec:
    """Create a custom item type specification.

    This is the main API for defining new item types, equivalent to
    R's mirt::createItem().

    Parameters
    ----------
    name : str
        Name for the item type
    icc_function : callable
        Function that computes item response probabilities.
        For dichotomous items: P(X=1|theta)
        For polytomous items: P(X=k|theta) for all k

        The function signature should be:
            f(theta, param1, param2, ...) -> ndarray

        Where theta is shape (n,) or (n, 1) and returns probabilities
        of shape (n,) for dichotomous or (n, n_categories) for polytomous.

    info_function : callable, optional
        Function that computes item information.
        Signature: f(theta, param1, param2, ...) -> ndarray

        If not provided, information is computed numerically from ICC.

    par_names : list[str], optional
        Names of the item parameters.
        If not provided, inferred from icc_function signature.

    par_bounds : dict, optional
        Bounds for each parameter as {"param_name": (lower, upper)}.
        Parameters not specified get (-inf, inf).

    par_defaults : dict, optional
        Default initial values for each parameter.
        Parameters not specified get 0.0.

    n_categories : int
        Number of response categories. Default 2 (dichotomous).

    gradient_function : callable, optional
        Function computing parameter gradients for optimization.
        If not provided, numerical gradients are used.

    Returns
    -------
    ItemTypeSpec
        Item type specification that can be used with CustomItemModel

    Examples
    --------
    >>> # Standard 2PL
    >>> def icc_2pl(theta, discrimination, difficulty):
    ...     return 1 / (1 + np.exp(-discrimination * (theta - difficulty)))
    >>>
    >>> TwoPL = create_item_type(
    ...     name="2PL",
    ...     icc_function=icc_2pl,
    ...     par_names=["discrimination", "difficulty"],
    ...     par_bounds={"discrimination": (0.01, 5.0), "difficulty": (-5.0, 5.0)},
    ...     par_defaults={"discrimination": 1.0, "difficulty": 0.0},
    ... )

    >>> # Custom 4PL with different parameterization
    >>> def icc_4pl(theta, a, b, c, d):
    ...     exp_term = np.exp(-a * (theta - b))
    ...     return c + (d - c) / (1 + exp_term)
    >>>
    >>> FourPL = create_item_type(
    ...     name="4PL",
    ...     icc_function=icc_4pl,
    ...     par_names=["a", "b", "c", "d"],
    ...     par_bounds={"a": (0.01, 5), "b": (-5, 5), "c": (0, 0.5), "d": (0.5, 1)},
    ...     par_defaults={"a": 1, "b": 0, "c": 0, "d": 1},
    ... )
    """
    import inspect

    if par_names is None:
        sig = inspect.signature(icc_function)
        par_names = [
            p.name
            for p in sig.parameters.values()
            if p.name != "theta" and p.name != "self"
        ]

    return ItemTypeSpec(
        name=name,
        icc_function=icc_function,
        info_function=info_function,
        par_names=par_names,
        par_bounds=par_bounds or {},
        par_defaults=par_defaults or {},
        n_categories=n_categories,
        gradient_function=gradient_function,
    )


class CustomItemModel:
    """IRT model using custom item types.

    This class allows using user-defined item types in IRT estimation.

    Parameters
    ----------
    n_items : int
        Number of items
    item_type : ItemTypeSpec or callable
        Item type specification or ICC function
    n_factors : int
        Number of latent factors (default 1)

    Examples
    --------
    >>> # Define a custom item type
    >>> def my_icc(theta, a, b, c):
    ...     return c + (1 - c) / (1 + np.exp(-a * (theta - b)))
    >>>
    >>> MyItem = create_item_type(
    ...     name="3PL",
    ...     icc_function=my_icc,
    ...     par_names=["a", "b", "c"],
    ...     par_bounds={"a": (0.01, 5), "b": (-5, 5), "c": (0, 0.4)},
    ... )
    >>>
    >>> model = CustomItemModel(n_items=10, item_type=MyItem)
    """

    def __init__(
        self,
        n_items: int,
        item_type: ItemTypeSpec | Callable,
        n_factors: int = 1,
    ) -> None:
        self.n_items = n_items
        self.n_factors = n_factors
        self._is_fitted = False

        if callable(item_type) and not isinstance(item_type, ItemTypeSpec):
            item_type = create_item_type(
                name="Custom",
                icc_function=item_type,
            )

        self.item_type = item_type
        self.model_name = item_type.name

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        for name in self.item_type.par_names:
            default = self.item_type.par_defaults.get(name, 0.0)
            self._parameters[name] = np.full(self.n_items, default)

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        """Get model parameters."""
        return self._parameters.copy()

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def is_polytomous(self) -> bool:
        """Whether this is a polytomous model."""
        return self.item_type.n_categories > 2

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return self.n_items * len(self.item_type.par_names)

    def set_parameters(self, **kwargs: Any) -> None:
        """Set model parameters."""
        for name, values in kwargs.items():
            if name in self._parameters:
                self._parameters[name] = np.asarray(values)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set a single item parameter."""
        if param_name in self._parameters:
            self._parameters[param_name][item_idx] = value

    def get_item_parameters(self, item_idx: int) -> dict[str, float]:
        """Get parameters for a specific item."""
        return {
            name: float(values[item_idx]) for name, values in self._parameters.items()
        }

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        Parameters
        ----------
        theta : ndarray
            Ability values, shape (n_persons,) or (n_persons, n_factors)
        item_idx : int, optional
            If provided, compute for single item. Otherwise all items.

        Returns
        -------
        ndarray
            Probabilities. Shape depends on model type and item_idx.
        """
        theta = np.atleast_1d(theta)
        if theta.ndim == 2:
            theta = theta[:, 0]

        if item_idx is not None:
            params = self.get_item_parameters(item_idx)
            return self.item_type.icc_function(theta, **params)
        else:
            probs = np.zeros((len(theta), self.n_items))
            for j in range(self.n_items):
                params = self.get_item_parameters(j)
                probs[:, j] = self.item_type.icc_function(theta, **params)
            return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute item information.

        Parameters
        ----------
        theta : ndarray
            Ability values
        item_idx : int, optional
            If provided, compute for single item

        Returns
        -------
        ndarray
            Information values
        """
        theta = np.atleast_1d(theta)
        if theta.ndim == 2:
            theta = theta[:, 0]

        if self.item_type.info_function is not None:
            if item_idx is not None:
                params = self.get_item_parameters(item_idx)
                return self.item_type.info_function(theta, **params)
            else:
                info = np.zeros((len(theta), self.n_items))
                for j in range(self.n_items):
                    params = self.get_item_parameters(j)
                    info[:, j] = self.item_type.info_function(theta, **params)
                return info
        else:
            return self._numerical_information(theta, item_idx)

    def _numerical_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute information numerically from ICC."""
        h = 1e-5

        if item_idx is not None:
            p = self.probability(theta, item_idx)
            p_plus = self.probability(theta + h, item_idx)
            p_minus = self.probability(theta - h, item_idx)

            dp = (p_plus - p_minus) / (2 * h)

            p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
            info = dp**2 / (p * (1 - p))

            return info
        else:
            info = np.zeros((len(theta), self.n_items))
            for j in range(self.n_items):
                info[:, j] = self._numerical_information(theta, j)
            return info

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood of responses given theta.

        Parameters
        ----------
        responses : ndarray
            Response matrix, shape (n_persons, n_items)
        theta : ndarray
            Ability values, shape (n_persons,) or (n_persons, n_factors)

        Returns
        -------
        ndarray
            Log-likelihood for each person, shape (n_persons,)
        """
        probs = self.probability(theta)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(probs) + (1 - responses) * np.log(1 - probs),
            0.0,
        )

        return np.sum(ll, axis=1)

    def copy(self) -> CustomItemModel:
        """Create a copy of the model."""
        new_model = CustomItemModel(
            n_items=self.n_items,
            item_type=self.item_type,
            n_factors=self.n_factors,
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._is_fitted = self._is_fitted
        return new_model


STANDARD_2PL = create_item_type(
    name="Standard2PL",
    icc_function=lambda theta, a, b: 1 / (1 + np.exp(-a * (theta - b))),
    info_function=lambda theta, a, b: (
        a**2
        * (1 / (1 + np.exp(-a * (theta - b))))
        * (1 - 1 / (1 + np.exp(-a * (theta - b))))
    ),
    par_names=["a", "b"],
    par_bounds={"a": (0.01, 5.0), "b": (-5.0, 5.0)},
    par_defaults={"a": 1.0, "b": 0.0},
)

STANDARD_3PL = create_item_type(
    name="Standard3PL",
    icc_function=lambda theta, a, b, c: c + (1 - c) / (1 + np.exp(-a * (theta - b))),
    par_names=["a", "b", "c"],
    par_bounds={"a": (0.01, 5.0), "b": (-5.0, 5.0), "c": (0.0, 0.5)},
    par_defaults={"a": 1.0, "b": 0.0, "c": 0.0},
)

LOGISTIC_DEVIATION = create_item_type(
    name="LogisticDeviation",
    icc_function=lambda theta, alpha, delta: 1 / (1 + np.exp(-(alpha + delta * theta))),
    par_names=["alpha", "delta"],
    par_bounds={"alpha": (-5.0, 5.0), "delta": (0.01, 5.0)},
    par_defaults={"alpha": 0.0, "delta": 1.0},
)


def list_standard_item_types() -> list[str]:
    """List available standard item type specifications."""
    return ["STANDARD_2PL", "STANDARD_3PL", "LOGISTIC_DEVIATION"]


def get_standard_item_type(name: str) -> ItemTypeSpec:
    """Get a standard item type specification by name."""
    types = {
        "STANDARD_2PL": STANDARD_2PL,
        "STANDARD_3PL": STANDARD_3PL,
        "LOGISTIC_DEVIATION": LOGISTIC_DEVIATION,
    }
    if name not in types:
        raise ValueError(f"Unknown item type: {name}. Available: {list(types.keys())}")
    return types[name]


@dataclass
class GroupSpec:
    """Specification for a custom group-level object.

    Used in multiple group IRT analysis to define group-specific
    characteristics like latent distribution parameters.

    Parameters
    ----------
    name : str
        Name of the group.
    mean_function : callable, optional
        Function computing group mean of latent trait.
        Signature: f(**params) -> float or ndarray
    cov_function : callable, optional
        Function computing group covariance of latent trait.
        Signature: f(**params) -> ndarray
    par_names : list[str]
        Names of group-level parameters.
    par_bounds : dict[str, tuple]
        Bounds for each parameter as (lower, upper).
    par_defaults : dict[str, float]
        Default values for each parameter.
    n_factors : int
        Number of latent factors. Default 1.
    """

    name: str
    mean_function: Callable[..., float | NDArray[np.float64]] | None = None
    cov_function: Callable[..., NDArray[np.float64]] | None = None
    par_names: list[str] = field(default_factory=list)
    par_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    par_defaults: dict[str, float] = field(default_factory=dict)
    n_factors: int = 1

    def __post_init__(self):
        for name in self.par_names:
            if name not in self.par_bounds:
                self.par_bounds[name] = (-np.inf, np.inf)
            if name not in self.par_defaults:
                self.par_defaults[name] = 0.0


def createGroup(
    name: str,
    mean_function: Callable[..., float | NDArray[np.float64]] | None = None,
    cov_function: Callable[..., NDArray[np.float64]] | None = None,
    par_names: list[str] | None = None,
    par_bounds: dict[str, tuple[float, float]] | None = None,
    par_defaults: dict[str, float] | None = None,
    n_factors: int = 1,
) -> GroupSpec:
    """Create a user-defined group-level object for multiple group analysis.

    This is equivalent to R's mirt::createGroup() function. It allows
    specification of custom latent distribution models for groups in
    multiple group IRT analysis.

    Parameters
    ----------
    name : str
        Name for the group specification.
    mean_function : callable, optional
        Function that computes the group mean of the latent trait.
        Signature: f(param1, param2, ...) -> float or ndarray

        If None, uses zero mean.

    cov_function : callable, optional
        Function that computes the group covariance matrix.
        Signature: f(param1, param2, ...) -> ndarray

        If None, uses identity covariance.

    par_names : list[str], optional
        Names of the group-level parameters.
        If None, inferred from function signatures.

    par_bounds : dict, optional
        Bounds for each parameter as {"param_name": (lower, upper)}.

    par_defaults : dict, optional
        Default initial values for each parameter.

    n_factors : int
        Number of latent factors. Default 1.

    Returns
    -------
    GroupSpec
        Group specification for use in multiple group models.

    Examples
    --------
    >>> # Define a group with estimable mean and variance
    >>> def group_mean(mu):
    ...     return mu
    >>>
    >>> def group_cov(sigma):
    ...     return np.array([[sigma**2]])
    >>>
    >>> FreeGroup = createGroup(
    ...     name="FreeGroup",
    ...     mean_function=group_mean,
    ...     cov_function=group_cov,
    ...     par_names=["mu", "sigma"],
    ...     par_bounds={"mu": (-5, 5), "sigma": (0.1, 5)},
    ...     par_defaults={"mu": 0, "sigma": 1},
    ... )

    >>> # Use in multiple group analysis
    >>> # result = fit_multigroup(data, groups, group_spec=FreeGroup)

    Notes
    -----
    Common use cases include:
    - Freely estimating group means and variances
    - Specifying non-normal latent distributions
    - Implementing latent regression models at the group level
    """
    import inspect

    inferred_names: list[str] = []

    if par_names is None:
        if mean_function is not None:
            sig = inspect.signature(mean_function)
            inferred_names.extend(
                p.name
                for p in sig.parameters.values()
                if p.name not in ("self", "theta")
            )

        if cov_function is not None:
            sig = inspect.signature(cov_function)
            for p in sig.parameters.values():
                if p.name not in ("self", "theta") and p.name not in inferred_names:
                    inferred_names.append(p.name)

        par_names = inferred_names if inferred_names else []

    return GroupSpec(
        name=name,
        mean_function=mean_function,
        cov_function=cov_function,
        par_names=par_names,
        par_bounds=par_bounds or {},
        par_defaults=par_defaults or {},
        n_factors=n_factors,
    )


class CustomGroupModel:
    """Model for custom group-level specifications.

    Wraps a GroupSpec for use in multiple group IRT models.

    Parameters
    ----------
    group_spec : GroupSpec
        Group specification from createGroup().
    """

    def __init__(self, group_spec: GroupSpec) -> None:
        self.spec = group_spec
        self.name = group_spec.name
        self.n_factors = group_spec.n_factors
        self._parameters: dict[str, float] = {}
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        for name in self.spec.par_names:
            self._parameters[name] = self.spec.par_defaults.get(name, 0.0)

    @property
    def parameters(self) -> dict[str, float]:
        """Get group parameters."""
        return self._parameters.copy()

    def set_parameters(self, **kwargs: float) -> None:
        """Set group parameters."""
        for name, value in kwargs.items():
            if name in self._parameters:
                self._parameters[name] = value

    def get_mean(self) -> NDArray[np.float64]:
        """Get the group mean vector."""
        if self.spec.mean_function is None:
            return np.zeros(self.n_factors)
        result = self.spec.mean_function(**self._parameters)
        return np.atleast_1d(result)

    def get_cov(self) -> NDArray[np.float64]:
        """Get the group covariance matrix."""
        if self.spec.cov_function is None:
            return np.eye(self.n_factors)
        return np.atleast_2d(self.spec.cov_function(**self._parameters))

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample from the group latent distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        ndarray
            Samples of shape (n, n_factors).
        """
        if rng is None:
            rng = np.random.default_rng()

        mean = self.get_mean()
        cov = self.get_cov()

        return rng.multivariate_normal(mean, cov, size=n)
