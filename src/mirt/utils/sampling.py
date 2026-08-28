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
        Difficulty samples. Shape: (n_samples, n_items).
    guessing : NDArray[np.float64] | None
        Guessing parameter samples if applicable.
    slipping : NDArray[np.float64] | None
        Slipping parameter samples if applicable.
    upper : NDArray[np.float64] | None
        Upper-asymptote samples if applicable.
    asymmetry : NDArray[np.float64] | None
        Logistic asymmetry samples if applicable.
    """

    discrimination: NDArray[np.float64]
    difficulty: NDArray[np.float64]
    guessing: NDArray[np.float64] | None = None
    slipping: NDArray[np.float64] | None = None
    upper: NDArray[np.float64] | None = None
    asymmetry: NDArray[np.float64] | None = None


def _draw_bounded_parameter(
    rng: np.random.Generator,
    values: NDArray[np.float64] | None,
    n_samples: int,
    lower: float,
    upper: float,
    scale: float = 0.02,
) -> NDArray[np.float64] | None:
    """Draw independent bounded samples for an optional item parameter."""
    if values is None:
        return None
    draws = rng.normal(values, scale, size=(n_samples, values.size))
    return np.clip(draws, lower, upper)


def _optional_model_parameter(
    model: "BaseItemModel",
    name: str,
    n_items: int,
) -> NDArray[np.float64] | None:
    """Read and validate an optional per-item model parameter."""
    values = getattr(model, name, None)
    if values is None:
        return None
    parameter = np.asarray(values, dtype=np.float64)
    if parameter.shape != (n_items,) or not np.all(np.isfinite(parameter)):
        raise ValueError(f"model.{name} must contain one finite value per item")
    return parameter


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
        A fitted model exposing discrimination and one difficulty per item.
    n_samples : int
        Number of samples to draw. Default 1000.
    vcov : NDArray[np.float64], optional
        Discrimination/difficulty variance-covariance matrix. If None, uses a
        compatible ``model.vcov`` when available or a diagonal default.
    method : str
        Sampling method. Currently only "mvn" (asymptotic multivariate
        normal sampling) is supported.
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
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if method != "mvn":
        raise ValueError("method must be 'mvn'")

    rng = np.random.default_rng(seed)

    try:
        disc = np.asarray(model.discrimination, dtype=np.float64)
        diff = np.asarray(model.difficulty, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(
            "model must expose discrimination and difficulty parameters"
        ) from exc

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)
    elif disc.ndim != 2:
        raise ValueError("model.discrimination must be one- or two-dimensional")

    n_items = disc.shape[0]
    n_dims = disc.shape[1]
    if diff.shape != (n_items,):
        raise ValueError("model.difficulty must contain one value per item")
    if not np.all(np.isfinite(disc)) or not np.all(np.isfinite(diff)):
        raise ValueError("model parameters must be finite")

    disc_flat = disc.ravel()
    diff_flat = diff.ravel()

    mean = np.concatenate([disc_flat, diff_flat])
    n_params = len(mean)

    if vcov is None:
        model_vcov = getattr(model, "vcov", None)
        if model_vcov is not None and np.shape(model_vcov) == (n_params, n_params):
            vcov = np.asarray(model_vcov, dtype=np.float64)
        else:
            vcov = np.eye(n_params, dtype=np.float64) * 0.01
    else:
        vcov = np.asarray(vcov, dtype=np.float64)
        if vcov.shape != (n_params, n_params):
            raise ValueError(f"vcov must have shape {(n_params, n_params)}")

    if not np.all(np.isfinite(vcov)):
        raise ValueError("vcov must contain only finite values")

    vcov = (vcov + vcov.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(vcov))
    if min_eig < REGULARIZATION_EPSILON:
        vcov = vcov + np.eye(n_params) * (REGULARIZATION_EPSILON - min_eig)

    samples = rng.multivariate_normal(mean, vcov, size=n_samples, check_valid="raise")

    n_disc = len(disc_flat)
    disc_samples = samples[:, :n_disc].reshape(n_samples, n_items, n_dims)
    diff_samples = samples[:, n_disc:].reshape(n_samples, -1)

    disc_samples = np.maximum(disc_samples, 0.01)

    if disc_samples.shape[2] == 1:
        disc_samples = disc_samples.squeeze(axis=2)

    guessing_samples = _draw_bounded_parameter(
        rng,
        _optional_model_parameter(model, "guessing", n_items),
        n_samples,
        0.0,
        0.5,
    )
    slipping_samples = _draw_bounded_parameter(
        rng,
        _optional_model_parameter(model, "slipping", n_items),
        n_samples,
        0.0,
        0.5,
    )
    upper_samples = _draw_bounded_parameter(
        rng,
        _optional_model_parameter(model, "upper", n_items),
        n_samples,
        0.5,
        1.0,
    )
    if guessing_samples is not None and upper_samples is not None:
        upper_samples = np.maximum(upper_samples, guessing_samples + PROB_EPSILON)

    asymmetry = _optional_model_parameter(model, "asymmetry", n_items)
    asymmetry_samples = None
    if asymmetry is not None:
        asymmetry_samples = np.maximum(
            rng.normal(asymmetry, 0.05, size=(n_samples, n_items)),
            PROB_EPSILON,
        )

    return ParameterSamples(
        discrimination=disc_samples,
        difficulty=diff_samples,
        guessing=guessing_samples,
        slipping=slipping_samples,
        upper=upper_samples,
        asymmetry=asymmetry_samples,
    )


def _optional_sample_array(
    values: NDArray[np.float64] | None,
    name: str,
    expected_shape: tuple[int, int],
) -> NDArray[np.float64] | None:
    """Validate an optional sampled item parameter."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"samples.{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"samples.{name} must contain only finite values")
    return array


def _validated_parameter_samples(
    samples: ParameterSamples,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Return consistently shaped, finite arrays from a sample container."""
    try:
        discrimination = np.asarray(samples.discrimination, dtype=np.float64)
        difficulty = np.asarray(samples.difficulty, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "samples must expose numeric discrimination and difficulty arrays"
        ) from exc

    if discrimination.ndim not in {2, 3}:
        raise ValueError("samples.discrimination must be two- or three-dimensional")
    if any(size == 0 for size in discrimination.shape):
        raise ValueError(
            "samples.discrimination must contain at least one draw, item, and factor"
        )
    if not np.all(np.isfinite(discrimination)):
        raise ValueError("samples.discrimination must contain only finite values")

    n_samples, n_items = discrimination.shape[:2]
    expected_shape = (n_samples, n_items)
    if difficulty.shape != expected_shape:
        raise ValueError(f"samples.difficulty must have shape {expected_shape}")
    if not np.all(np.isfinite(difficulty)):
        raise ValueError("samples.difficulty must contain only finite values")

    return (
        discrimination,
        difficulty,
        _optional_sample_array(samples.guessing, "guessing", expected_shape),
        _optional_sample_array(samples.slipping, "slipping", expected_shape),
        _optional_sample_array(samples.upper, "upper", expected_shape),
        _optional_sample_array(samples.asymmetry, "asymmetry", expected_shape),
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
    if not np.isfinite(credible_level) or not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1")

    alpha = 1 - credible_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    validated = _validated_parameter_samples(samples)
    parameter_arrays = dict(
        zip(
            (
                "discrimination",
                "difficulty",
                "guessing",
                "slipping",
                "upper",
                "asymmetry",
            ),
            validated,
            strict=True,
        )
    )

    return {
        name: {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "median": np.median(values, axis=0),
            "ci_lower": np.percentile(values, lower_q, axis=0),
            "ci_upper": np.percentile(values, upper_q, axis=0),
        }
        for name, values in parameter_arrays.items()
        if values is not None
    }


def _sigmoid_inplace(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply an overflow-free logistic transform without another full tensor."""
    np.negative(values, out=values)
    np.logaddexp(0.0, values, out=values)
    np.negative(values, out=values)
    np.exp(values, out=values)
    return values


def sample_expected_scores(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    samples: ParameterSamples,
    *,
    item_idx: int | None = None,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Compute expected scores using parameter samples.

    Useful for propagating parameter uncertainty to predictions.

    Parameters
    ----------
    model : BaseItemModel
        Logistic IRT model defining item and factor structure.
    theta : NDArray[np.float64]
        Ability values. Shape: (n_persons,) or (n_persons, n_dims).
    samples : ParameterSamples
        Parameter samples from draw_parameters().
    item_idx : int, optional
        Zero-based item index. When provided, uncertainty is propagated for
        only that item instead of allocating logits for the full item bank.
    chunk_size : int, optional
        Number of parameter samples to evaluate per vectorized chunk. By
        default, a memory-aware chunk size is selected automatically.

    Returns
    -------
    NDArray[np.float64]
        Expected scores for each sample. Shape: ``(n_samples, n_persons)``.
        With ``item_idx``, each value is that item's expected score; otherwise
        it is the expected total score across all items.
    """
    if model.model_name not in {"1PL", "2PL", "3PL", "4PL", "5PL"}:
        raise ValueError("sample_expected_scores requires a logistic item model")

    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1) if model.n_factors == 1 else theta.reshape(1, -1)
    if theta.ndim != 2 or theta.shape[1] != model.n_factors:
        raise ValueError(f"theta must have shape (n_persons, {model.n_factors})")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values")

    (
        disc,
        diff,
        guessing,
        sampled_slipping,
        sampled_upper,
        asymmetry,
    ) = _validated_parameter_samples(samples)
    if disc.ndim == 2:
        disc = disc[:, :, np.newaxis]

    n_samples, n_items, n_dims = disc.shape
    if n_items != model.n_items or n_dims != model.n_factors:
        raise ValueError(
            "sample discrimination dimensions must match the model's items and factors"
        )
    if item_idx is not None:
        if (
            isinstance(item_idx, bool)
            or not isinstance(item_idx, (int, np.integer))
            or item_idx < 0
            or item_idx >= n_items
        ):
            raise IndexError(f"item_idx must be an integer in [0, {n_items})")
        item_idx = int(item_idx)

    all_item_shape = (n_samples, n_items)

    if sampled_upper is not None and sampled_slipping is not None:
        raise ValueError("samples.upper and samples.slipping are mutually exclusive")

    def fixed_parameter(name: str) -> NDArray[np.float64] | None:
        values = _optional_model_parameter(model, name, n_items)
        return None if values is None else np.broadcast_to(values, all_item_shape)

    if guessing is None:
        guessing = fixed_parameter("guessing")
    if asymmetry is None:
        asymmetry = fixed_parameter("asymmetry")

    if sampled_upper is not None:
        upper = sampled_upper
        slipping = None
    elif sampled_slipping is not None:
        upper = None
        slipping = sampled_slipping
    else:
        upper = fixed_parameter("upper")
        slipping = None if upper is not None else fixed_parameter("slipping")

    if guessing is not None and np.any((guessing < 0.0) | (guessing > 1.0)):
        raise ValueError("samples.guessing must be between 0 and 1")
    if slipping is not None and np.any((slipping < 0.0) | (slipping > 1.0)):
        raise ValueError("samples.slipping must be between 0 and 1")
    if upper is not None and np.any((upper < 0.0) | (upper > 1.0)):
        raise ValueError("samples.upper must be between 0 and 1")
    if asymmetry is not None and np.any(asymmetry <= 0.0):
        raise ValueError("samples.asymmetry must be positive")

    lower = guessing if guessing is not None else np.zeros(all_item_shape)
    if upper is not None:
        effective_upper = upper
    elif slipping is not None:
        effective_upper = 1.0 - slipping
    else:
        effective_upper = np.ones(all_item_shape)
    if np.any(lower > effective_upper):
        raise ValueError("sample lower asymptotes cannot exceed upper asymptotes")

    if item_idx is not None:
        item_selection = slice(item_idx, item_idx + 1)
        disc = disc[:, item_selection, :]
        diff = diff[:, item_selection]
        lower = lower[:, item_selection]
        effective_upper = effective_upper[:, item_selection]
        if asymmetry is not None:
            asymmetry = asymmetry[:, item_selection]
    n_scored_items = disc.shape[1]

    n_persons = theta.shape[0]
    expected = np.empty((n_samples, n_persons), dtype=np.float64)
    if n_persons == 0:
        return expected

    if chunk_size is None:
        elements_per_sample = max(1, n_persons * n_scored_items)
        chunk_size = max(1, min(n_samples, 4_000_000 // elements_per_sample))
    elif (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        disc_chunk = disc[start:stop]
        if n_dims == 1:
            logits = theta[:, 0][None, :, None] - diff[start:stop, None, :]
            logits *= disc_chunk[:, None, :, 0]
        else:
            logits = np.einsum("pd,sjd->spj", theta, disc_chunk, optimize=True)
            logits -= disc_chunk.sum(axis=2)[:, None, :] * diff[start:stop, None, :]

        probabilities = _sigmoid_inplace(logits)
        if asymmetry is not None:
            np.power(
                probabilities,
                asymmetry[start:stop, None, :],
                out=probabilities,
            )
        probabilities *= (effective_upper[start:stop] - lower[start:stop])[:, None, :]
        probabilities += lower[start:stop, None, :]
        expected[start:stop] = probabilities.sum(axis=2)

    return expected
