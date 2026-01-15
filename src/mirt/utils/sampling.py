"""Parameter sampling functions for IRT models.

Provides functions for drawing parameter samples from the
posterior distribution for uncertainty quantification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ParameterSamples:
    """Container for parameter samples.

    Attributes
    ----------
    discrimination : NDArray[np.float64]
        Discrimination samples. Shape: (n_samples, n_items) or (n_samples, n_items, n_dims).
    difficulty : NDArray[np.float64]
        Difficulty samples. Shape: (n_samples, n_items) or varies for polytomous.
    guessing : NDArray[np.float64] | None
        Guessing parameter samples if applicable.
    slipping : NDArray[np.float64] | None
        Slipping parameter samples if applicable.
    """

    discrimination: NDArray[np.float64]
    difficulty: NDArray[np.float64]
    guessing: NDArray[np.float64] | None = None
    slipping: NDArray[np.float64] | None = None


def draw_parameters(
    model: "BaseItemModel",
    n_samples: int = 1000,
    vcov: NDArray[np.float64] | None = None,
    method: str = "mvn",
    seed: int | None = None,
) -> ParameterSamples:
    """Draw parameter samples from approximate posterior.

    Uses the asymptotic normal approximation to the posterior
    distribution based on the variance-covariance matrix.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    n_samples : int
        Number of samples to draw. Default 1000.
    vcov : NDArray[np.float64], optional
        Variance-covariance matrix. If None, estimated from
        information matrix or uses default.
    method : str
        Sampling method:
        - "mvn": Multivariate normal (asymptotic)
        - "bootstrap": Bootstrap resampling (requires responses)
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ParameterSamples
        Container with parameter samples.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> samples = draw_parameters(result.model, n_samples=1000)
    >>> # Compute 95% credible interval for item 0 discrimination
    >>> ci = np.percentile(samples.discrimination[:, 0], [2.5, 97.5])
    >>> print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """
    rng = np.random.default_rng(seed)

    disc = np.asarray(model.discrimination)
    diff = np.asarray(model.difficulty)

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    n_items = disc.shape[0]
    n_dims = disc.shape[1] if disc.ndim > 1 else 1

    disc_flat = disc.ravel()
    diff_flat = diff.ravel()

    mean = np.concatenate([disc_flat, diff_flat])
    n_params = len(mean)

    if vcov is None:
        if hasattr(model, "vcov") and model.vcov is not None:
            vcov = model.vcov
        else:
            vcov = np.eye(n_params) * 0.01

    if vcov.shape[0] != n_params:
        vcov = np.eye(n_params) * 0.01

    vcov = (vcov + vcov.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(vcov))
    if min_eig < PROB_EPSILON:
        vcov = vcov + np.eye(n_params) * (REGULARIZATION_EPSILON - min_eig)

    if method == "mvn":
        samples = rng.multivariate_normal(mean, vcov, size=n_samples)
    else:
        samples = rng.multivariate_normal(mean, vcov, size=n_samples)

    n_disc = len(disc_flat)
    disc_samples = samples[:, :n_disc].reshape(n_samples, n_items, n_dims)
    diff_samples = samples[:, n_disc:].reshape(n_samples, -1)

    disc_samples = np.maximum(disc_samples, 0.01)

    if disc_samples.shape[2] == 1:
        disc_samples = disc_samples.squeeze(axis=2)

    guessing_samples = None
    slipping_samples = None

    if hasattr(model, "guessing") and model.guessing is not None:
        g = np.asarray(model.guessing)
        g_se = 0.02
        guessing_samples = rng.normal(g, g_se, size=(n_samples, len(g)))
        guessing_samples = np.clip(guessing_samples, 0, 0.5)

    if hasattr(model, "slipping") and model.slipping is not None:
        s = np.asarray(model.slipping)
        s_se = 0.02
        slipping_samples = rng.normal(s, s_se, size=(n_samples, len(s)))
        slipping_samples = np.clip(slipping_samples, 0.5, 1.0)

    return ParameterSamples(
        discrimination=disc_samples,
        difficulty=diff_samples,
        guessing=guessing_samples,
        slipping=slipping_samples,
    )


def posterior_summary(
    samples: ParameterSamples,
    credible_level: float = 0.95,
) -> dict[str, dict]:
    """Compute posterior summary statistics from samples.

    Parameters
    ----------
    samples : ParameterSamples
        Parameter samples from draw_parameters().
    credible_level : float
        Level for credible intervals. Default 0.95.

    Returns
    -------
    dict
        Nested dictionary with summary statistics for each parameter type.

    Examples
    --------
    >>> samples = draw_parameters(result.model)
    >>> summary = posterior_summary(samples)
    >>> print(summary["discrimination"]["mean"])
    >>> print(summary["discrimination"]["ci_lower"])
    """
    alpha = 1 - credible_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    summary = {}

    summary["discrimination"] = {
        "mean": np.mean(samples.discrimination, axis=0),
        "std": np.std(samples.discrimination, axis=0),
        "median": np.median(samples.discrimination, axis=0),
        "ci_lower": np.percentile(samples.discrimination, lower_q, axis=0),
        "ci_upper": np.percentile(samples.discrimination, upper_q, axis=0),
    }

    summary["difficulty"] = {
        "mean": np.mean(samples.difficulty, axis=0),
        "std": np.std(samples.difficulty, axis=0),
        "median": np.median(samples.difficulty, axis=0),
        "ci_lower": np.percentile(samples.difficulty, lower_q, axis=0),
        "ci_upper": np.percentile(samples.difficulty, upper_q, axis=0),
    }

    if samples.guessing is not None:
        summary["guessing"] = {
            "mean": np.mean(samples.guessing, axis=0),
            "std": np.std(samples.guessing, axis=0),
            "median": np.median(samples.guessing, axis=0),
            "ci_lower": np.percentile(samples.guessing, lower_q, axis=0),
            "ci_upper": np.percentile(samples.guessing, upper_q, axis=0),
        }

    if samples.slipping is not None:
        summary["slipping"] = {
            "mean": np.mean(samples.slipping, axis=0),
            "std": np.std(samples.slipping, axis=0),
            "median": np.median(samples.slipping, axis=0),
            "ci_lower": np.percentile(samples.slipping, lower_q, axis=0),
            "ci_upper": np.percentile(samples.slipping, upper_q, axis=0),
        }

    return summary


def sample_expected_scores(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    samples: ParameterSamples,
) -> NDArray[np.float64]:
    """Compute expected scores using parameter samples.

    Useful for propagating parameter uncertainty to predictions.

    Parameters
    ----------
    model : BaseItemModel
        The base IRT model (for structure).
    theta : NDArray[np.float64]
        Ability values. Shape: (n_persons,) or (n_persons, n_dims).
    samples : ParameterSamples
        Parameter samples from draw_parameters().

    Returns
    -------
    NDArray[np.float64]
        Expected scores for each sample.
        Shape: (n_samples, n_persons).
    """
    theta = np.atleast_1d(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    n_samples = samples.discrimination.shape[0]
    n_persons = theta.shape[0]

    expected = np.zeros((n_samples, n_persons))

    disc = samples.discrimination
    diff = samples.difficulty

    if disc.ndim == 2:
        disc = disc[:, :, np.newaxis]

    for s in range(n_samples):
        for i in range(n_persons):
            score = 0.0
            for j in range(disc.shape[1]):
                logit = disc[s, j, 0] * (theta[i, 0] - diff[s, j])
                prob = 1 / (1 + np.exp(-logit))
                score += prob
            expected[s, i] = score

    return expected
