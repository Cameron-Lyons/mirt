"""Custom prior distributions for Bayesian IRT estimation.

This module provides flexible prior specifications for item parameters,
supporting both informative and weakly informative priors.

Supported distributions:
- Normal / Truncated Normal
- Log-normal
- Beta
- Uniform
- Gamma
- Custom (user-defined log-pdf)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class Prior(ABC):
    """Abstract base class for prior distributions."""

    @abstractmethod
    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log probability density at x."""
        pass

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw samples from the prior."""
        pass

    @property
    @abstractmethod
    def mean(self) -> float:
        """Prior mean."""
        pass

    @property
    @abstractmethod
    def variance(self) -> float:
        """Prior variance."""
        pass


class NormalPrior(Prior):
    """Normal (Gaussian) prior distribution.

    Parameters
    ----------
    mu : float
        Mean of the distribution
    sigma : float
        Standard deviation

    Examples
    --------
    >>> prior = NormalPrior(mu=0, sigma=1)
    >>> prior.log_pdf(np.array([0.0]))
    array([-0.9189385...])
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2

    def __repr__(self) -> str:
        return f"NormalPrior(mu={self.mu}, sigma={self.sigma})"


class TruncatedNormalPrior(Prior):
    """Truncated normal prior distribution.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal
    sigma : float
        Standard deviation of the underlying normal
    lower : float
        Lower truncation point
    upper : float
        Upper truncation point
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if lower >= upper:
            raise ValueError("lower must be less than upper")

        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        self._dist = stats.truncnorm(a, b, loc=mu, scale=sigma)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        return self._dist.rvs(size, random_state=rng)

    @property
    def mean(self) -> float:
        return float(self._dist.mean())

    @property
    def variance(self) -> float:
        return float(self._dist.var())

    def __repr__(self) -> str:
        return (
            f"TruncatedNormalPrior(mu={self.mu}, sigma={self.sigma}, "
            f"lower={self.lower}, upper={self.upper})"
        )


class LogNormalPrior(Prior):
    """Log-normal prior distribution.

    Useful for parameters that must be positive (e.g., discrimination).

    Parameters
    ----------
    mu : float
        Mean of the log of the variable
    sigma : float
        Standard deviation of the log of the variable
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.5) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return np.exp(rng.normal(self.mu, self.sigma, size))

    @property
    def mean(self) -> float:
        return float(np.exp(self.mu + self.sigma**2 / 2))

    @property
    def variance(self) -> float:
        return float((np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2))

    def __repr__(self) -> str:
        return f"LogNormalPrior(mu={self.mu}, sigma={self.sigma})"


class BetaPrior(Prior):
    """Beta prior distribution.

    Useful for parameters bounded between 0 and 1 (e.g., guessing).

    Parameters
    ----------
    alpha : float
        Alpha (shape) parameter
    beta : float
        Beta (shape) parameter
    """

    def __init__(self, alpha: float = 2.0, beta: float = 8.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.beta(alpha, beta)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        return self._dist.rvs(size, random_state=rng)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab**2 * (ab + 1))

    def __repr__(self) -> str:
        return f"BetaPrior(alpha={self.alpha}, beta={self.beta})"


class UniformPrior(Prior):
    """Uniform prior distribution.

    Parameters
    ----------
    lower : float
        Lower bound
    upper : float
        Upper bound
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0) -> None:
        if lower >= upper:
            raise ValueError("lower must be less than upper")
        self.lower = lower
        self.upper = upper
        self._dist = stats.uniform(loc=lower, scale=upper - lower)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lower, self.upper, size)

    @property
    def mean(self) -> float:
        return (self.lower + self.upper) / 2

    @property
    def variance(self) -> float:
        return (self.upper - self.lower) ** 2 / 12

    def __repr__(self) -> str:
        return f"UniformPrior(lower={self.lower}, upper={self.upper})"


class GammaPrior(Prior):
    """Gamma prior distribution.

    Useful for positive parameters like variance components.

    Parameters
    ----------
    shape : float
        Shape parameter (k or alpha)
    rate : float
        Rate parameter (1/scale or beta)
    """

    def __init__(self, shape: float = 1.0, rate: float = 1.0) -> None:
        if shape <= 0 or rate <= 0:
            raise ValueError("shape and rate must be positive")
        self.shape = shape
        self.rate = rate
        self._dist = stats.gamma(a=shape, scale=1 / rate)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._dist.logpdf(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        return self._dist.rvs(size, random_state=rng)

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        return self.shape / (self.rate**2)

    def __repr__(self) -> str:
        return f"GammaPrior(shape={self.shape}, rate={self.rate})"


class CustomPrior(Prior):
    """Custom prior distribution with user-defined log-pdf.

    Parameters
    ----------
    log_pdf_fn : callable
        Function that takes array and returns log-pdf values
    sample_fn : callable
        Function that takes size and rng, returns samples
    mean_value : float
        Prior mean
    variance_value : float
        Prior variance
    """

    def __init__(
        self,
        log_pdf_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        sample_fn: Callable[
            [int | tuple[int, ...], np.random.Generator | None], NDArray[np.float64]
        ],
        mean_value: float = 0.0,
        variance_value: float = 1.0,
    ) -> None:
        self._log_pdf_fn = log_pdf_fn
        self._sample_fn = sample_fn
        self._mean = mean_value
        self._variance = variance_value

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._log_pdf_fn(x)

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        return self._sample_fn(size, rng)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._variance

    def __repr__(self) -> str:
        return f"CustomPrior(mean={self._mean}, variance={self._variance})"


@dataclass
class PriorSpecification:
    """Complete prior specification for IRT model parameters.

    Attributes
    ----------
    discrimination : Prior
        Prior for discrimination parameters
    difficulty : Prior
        Prior for difficulty parameters
    guessing : Prior or None
        Prior for guessing parameters (3PL, 4PL)
    upper : Prior or None
        Prior for upper asymptote (4PL)
    theta : Prior
        Prior for latent abilities
    """

    discrimination: Prior | None = None
    difficulty: Prior | None = None
    guessing: Prior | None = None
    upper: Prior | None = None
    theta: Prior | None = None

    def __post_init__(self) -> None:
        if self.discrimination is None:
            self.discrimination = LogNormalPrior(mu=0, sigma=0.5)
        if self.difficulty is None:
            self.difficulty = NormalPrior(mu=0, sigma=2)
        if self.theta is None:
            self.theta = NormalPrior(mu=0, sigma=1)


def default_priors(model_name: str) -> PriorSpecification:
    """Get default priors for a given model type.

    Parameters
    ----------
    model_name : str
        Model type (1PL, 2PL, 3PL, 4PL)

    Returns
    -------
    PriorSpecification
        Default prior specification
    """
    spec = PriorSpecification(
        discrimination=LogNormalPrior(mu=0, sigma=0.5),
        difficulty=NormalPrior(mu=0, sigma=2),
        theta=NormalPrior(mu=0, sigma=1),
    )

    if model_name in ("3PL", "4PL"):
        spec.guessing = BetaPrior(alpha=2, beta=8)

    if model_name == "4PL":
        spec.upper = BetaPrior(alpha=8, beta=2)

    return spec


def weakly_informative_priors() -> PriorSpecification:
    """Get weakly informative priors.

    Returns more diffuse priors that constrain parameters
    to reasonable ranges without strongly influencing estimation.
    """
    return PriorSpecification(
        discrimination=LogNormalPrior(mu=0, sigma=1.0),
        difficulty=NormalPrior(mu=0, sigma=5),
        guessing=BetaPrior(alpha=1, beta=4),
        upper=BetaPrior(alpha=4, beta=1),
        theta=NormalPrior(mu=0, sigma=1),
    )


def compute_prior_log_pdf(
    priors: PriorSpecification,
    discrimination: NDArray[np.float64] | None = None,
    difficulty: NDArray[np.float64] | None = None,
    guessing: NDArray[np.float64] | None = None,
    upper: NDArray[np.float64] | None = None,
) -> float:
    """Compute total log-prior for model parameters.

    Parameters
    ----------
    priors : PriorSpecification
        Prior distributions
    discrimination, difficulty, guessing, upper : ndarray, optional
        Parameter values

    Returns
    -------
    float
        Total log-prior probability
    """
    log_prior = 0.0

    if discrimination is not None:
        log_prior += np.sum(priors.discrimination.log_pdf(discrimination.ravel()))

    if difficulty is not None:
        log_prior += np.sum(priors.difficulty.log_pdf(difficulty.ravel()))

    if guessing is not None and priors.guessing is not None:
        log_prior += np.sum(priors.guessing.log_pdf(guessing.ravel()))

    if upper is not None and priors.upper is not None:
        log_prior += np.sum(priors.upper.log_pdf(upper.ravel()))

    return float(log_prior)
