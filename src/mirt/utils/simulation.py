from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid

SimulationModel = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]
_MAX_STABLE_LOGIT = np.finfo(np.float64).max
_MAX_POLYTOMOUS_CHUNK_ENTRIES = 1_000_000


def simdata(
    model: SimulationModel = "2PL",
    n_persons: int = 500,
    n_items: int = 20,
    n_categories: int = 2,
    n_factors: int = 1,
    theta: NDArray[np.float64] | None = None,
    discrimination: NDArray[np.float64] | None = None,
    difficulty: NDArray[np.float64] | None = None,
    guessing: NDArray[np.float64] | None = None,
    upper: NDArray[np.float64] | None = None,
    thresholds: NDArray[np.float64] | None = None,
    seed: int | None = None,
    steps: NDArray[np.float64] | None = None,
    slopes: NDArray[np.float64] | None = None,
    intercepts: NDArray[np.float64] | None = None,
) -> NDArray[np.int_]:
    """Simulate item response data from an IRT model.

    Generates response data by sampling from the specified IRT model
    given item parameters and person abilities.

    Parameters
    ----------
    model : {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}, default="2PL"
        IRT model to simulate from:

        - "1PL": One-parameter logistic (equal discrimination)
        - "2PL": Two-parameter logistic
        - "3PL": Three-parameter logistic (with guessing)
        - "4PL": Four-parameter logistic (with guessing and slipping)
        - "GRM": Graded Response Model (polytomous)
        - "GPCM": Generalized Partial Credit Model (polytomous)
        - "PCM": Partial Credit Model (unit discrimination)
        - "NRM": Nominal Response Model (unordered categories)

    n_persons : int, default=500
        Number of persons to simulate.
    n_items : int, default=20
        Number of items to simulate.
    n_categories : int, default=2
        Number of response categories for polytomous models.
    n_factors : int, default=1
        Number of latent factors (dimensions).
    theta : ndarray, optional
        Person abilities. Shape (n_persons,) for unidimensional or
        (n_persons, n_factors) for multidimensional. If None, sampled
        from standard normal.
    discrimination : ndarray, optional
        Item discrimination parameters. If None, sampled from
        log-normal distribution.
    difficulty : ndarray, optional
        Item difficulty parameters. If None, sampled from standard normal.
    guessing : ndarray, optional
        Lower asymptote (guessing) parameters for 3PL/4PL models.
        If None, defaults to 0.2 for 3PL/4PL, 0 otherwise.
    upper : ndarray, optional
        Upper asymptote (slipping) parameters for 4PL model.
        If None, defaults to 1.0.
    thresholds : ndarray of shape (n_items, n_categories-1), optional
        Category threshold parameters for GRM, or a backward-compatible alias
        for GPCM step parameters.
        If None, equally spaced around difficulty.
    seed : int, optional
        Random seed for reproducibility.
    steps : ndarray of shape (n_items, n_categories-1), optional
        Step parameters for GPCM and PCM models.
    slopes : ndarray, optional
        NRM category slopes. Shape (n_items, n_categories) when unidimensional
        or (n_items, n_categories, n_factors) when multidimensional.
    intercepts : ndarray of shape (n_items, n_categories), optional
        NRM category intercepts.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Simulated response matrix. For dichotomous models, values are
        0 or 1. For polytomous models, values are 0, 1, ..., n_categories-1.

    Examples
    --------
    >>> from mirt import simdata
    >>> # Simulate 2PL data
    >>> data = simdata(model="2PL", n_persons=1000, n_items=30, seed=42)
    >>> print(data.shape)
    (1000, 30)

    >>> # Simulate with known parameters
    >>> import numpy as np
    >>> theta = np.random.randn(500)
    >>> a = np.random.lognormal(0, 0.3, 20)
    >>> b = np.random.randn(20)
    >>> data = simdata(theta=theta, discrimination=a, difficulty=b)

    >>> # Simulate polytomous GRM data
    >>> data = simdata(model="GRM", n_categories=5, n_items=15)
    """
    model = model.upper()
    valid_models = {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}
    if model not in valid_models:
        raise ValueError(f"Unknown model: {model}")
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)):
        raise ValueError("n_items must be a positive integer")
    if isinstance(n_persons, bool) or not isinstance(n_persons, (int, np.integer)):
        raise ValueError("n_persons must be a positive integer")
    if isinstance(n_factors, bool) or not isinstance(n_factors, (int, np.integer)):
        raise ValueError("n_factors must be a positive integer")
    if isinstance(n_categories, bool) or not isinstance(
        n_categories, (int, np.integer)
    ):
        raise ValueError("n_categories must be a positive integer")
    if n_items < 1:
        raise ValueError("n_items must be a positive integer")
    if n_persons < 1:
        raise ValueError("n_persons must be a positive integer")
    if n_factors < 1:
        raise ValueError("n_factors must be a positive integer")
    if model in {"GRM", "GPCM", "PCM", "NRM"} and n_categories < 2:
        raise ValueError("n_categories must be at least 2")

    rng = np.random.default_rng(seed)

    if theta is None:
        if n_factors == 1:
            theta = rng.standard_normal(n_persons)
        else:
            theta = rng.standard_normal((n_persons, n_factors))
    else:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim not in (1, 2):
            raise ValueError("theta must be one- or two-dimensional")
        n_persons = theta.shape[0]
        if n_persons < 1:
            raise ValueError("theta must contain at least one person")

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    n_factors = theta.shape[1]

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain finite values")

    if model in {"GRM", "GPCM", "PCM", "NRM"} and (
        guessing is not None or upper is not None
    ):
        raise ValueError("guessing and upper are only valid for 3PL and 4PL")

    if model == "NRM":
        if discrimination is not None:
            raise ValueError("NRM simulation: use slopes instead of discrimination")
        if difficulty is not None:
            raise ValueError("NRM simulation: use intercepts instead of difficulty")
        if thresholds is not None or steps is not None:
            raise ValueError("NRM does not use thresholds or steps")

        slope_values = _prepare_nrm_slopes(
            slopes,
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            rng=rng,
        )
        intercept_values = _prepare_parameter(
            "intercepts",
            intercepts
            if intercepts is not None
            else _default_nrm_intercepts(n_items, n_categories, rng),
            (n_items, n_categories),
        )
        return _simulate_nrm(theta, slope_values, intercept_values, rng)

    if slopes is not None or intercepts is not None:
        raise ValueError("slopes and intercepts are only valid for NRM")

    expected_discrimination_shape = (
        (n_items,) if n_factors == 1 else (n_items, n_factors)
    )

    if model in {"1PL", "PCM"}:
        if discrimination is not None:
            supplied_discrimination = _prepare_discrimination(
                discrimination,
                n_items=n_items,
                n_factors=n_factors,
            )
            if not np.allclose(supplied_discrimination, 1.0):
                raise ValueError(f"{model} discrimination is fixed to 1.0")
        discrimination = np.ones(expected_discrimination_shape, dtype=np.float64)
    elif discrimination is None:
        discrimination = rng.lognormal(
            0,
            0.25,
            size=expected_discrimination_shape,
        )
    else:
        discrimination = _prepare_discrimination(
            discrimination,
            n_items=n_items,
            n_factors=n_factors,
        )

    if model == "PCM":
        if n_factors != 1:
            raise ValueError("PCM simulation only supports unidimensional theta")
        if difficulty is not None:
            raise ValueError("PCM simulation: use steps instead of difficulty")
        step_values = _prepare_steps(
            steps=steps,
            thresholds=thresholds,
            n_items=n_items,
            n_categories=n_categories,
        )
        return _simulate_gpcm(
            theta=theta,
            discrimination=discrimination,
            n_categories=n_categories,
            thresholds=step_values,
            rng=rng,
        )

    if model in ("1PL", "2PL", "3PL", "4PL"):
        if steps is not None:
            raise ValueError("steps are only valid for GPCM and PCM")
        difficulty = _prepare_parameter(
            "difficulty",
            difficulty if difficulty is not None else rng.normal(0, 1, size=n_items),
            (n_items,),
        )
        return _simulate_dichotomous(
            model=model,
            theta=theta,
            discrimination=discrimination,
            difficulty=difficulty,
            guessing=guessing,
            upper=upper,
            rng=rng,
        )
    if model == "GRM":
        if steps is not None:
            raise ValueError("GRM uses thresholds instead of steps")
        difficulty = _prepare_parameter(
            "difficulty",
            difficulty if difficulty is not None else rng.normal(0, 1, size=n_items),
            (n_items,),
        )
        return _simulate_grm(
            theta=theta,
            discrimination=discrimination,
            difficulty=difficulty,
            n_categories=n_categories,
            thresholds=thresholds,
            rng=rng,
        )
    if model == "GPCM":
        if difficulty is not None:
            raise ValueError("GPCM uses steps instead of difficulty")
        step_values = _prepare_steps(
            steps=steps,
            thresholds=thresholds,
            n_items=n_items,
            n_categories=n_categories,
        )
        return _simulate_gpcm(
            theta=theta,
            discrimination=discrimination,
            n_categories=n_categories,
            thresholds=step_values,
            rng=rng,
        )

    raise AssertionError("unreachable simulation model")


def _prepare_parameter(
    name: str,
    value: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Validate and normalize one simulation parameter array."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _prepare_discrimination(
    discrimination: NDArray[np.float64],
    *,
    n_items: int,
    n_factors: int,
) -> NDArray[np.float64]:
    """Validate discrimination while accepting a unidimensional column."""
    array = np.asarray(discrimination, dtype=np.float64)
    if n_factors == 1 and array.shape == (n_items, 1):
        array = array[:, 0]
    expected = (n_items,) if n_factors == 1 else (n_items, n_factors)
    return _prepare_parameter("discrimination", array, expected)


def _prepare_steps(
    *,
    steps: NDArray[np.float64] | None,
    thresholds: NDArray[np.float64] | None,
    n_items: int,
    n_categories: int,
) -> NDArray[np.float64]:
    """Resolve GPCM/PCM step parameters and their legacy alias."""
    if steps is not None and thresholds is not None:
        raise ValueError("provide either steps or thresholds, not both")
    values = steps if steps is not None else thresholds
    if values is None:
        values = np.tile(
            np.linspace(-1.0, 1.0, n_categories - 1),
            (n_items, 1),
        )
    return _prepare_parameter(
        "steps",
        values,
        (n_items, n_categories - 1),
    )


def _prepare_nrm_slopes(
    slopes: NDArray[np.float64] | None,
    *,
    n_items: int,
    n_categories: int,
    n_factors: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate or validate identified NRM category slopes."""
    shape = (
        (n_items, n_categories)
        if n_factors == 1
        else (n_items, n_categories, n_factors)
    )
    if slopes is None:
        values = rng.normal(0.0, 0.6, size=shape)
        values[:, 0, ...] = 0.0
    else:
        values = slopes
    return _prepare_parameter("slopes", values, shape)


def _default_nrm_intercepts(
    n_items: int,
    n_categories: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate identified NRM category intercepts."""
    values = rng.normal(0.0, 0.6, size=(n_items, n_categories))
    values[:, 0] = 0.0
    return values


def _stable_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize category logits without overflow at extreme parameters."""
    with np.errstate(over="ignore", invalid="ignore"):
        finite_logits = np.nan_to_num(
            logits,
            nan=0.0,
            posinf=_MAX_STABLE_LOGIT,
            neginf=-_MAX_STABLE_LOGIT,
        )
        finite_logits = np.clip(
            finite_logits,
            -_MAX_STABLE_LOGIT,
            _MAX_STABLE_LOGIT,
        )
        shifted = finite_logits - finite_logits.max(axis=-1, keepdims=True)
        weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / weights.sum(axis=-1, keepdims=True)


def _polytomous_item_chunk_size(
    n_persons: int,
    n_items: int,
    n_categories: int,
) -> int:
    """Select an item chunk that bounds person-item-category temporaries."""
    entries_per_item = n_persons * n_categories
    return max(
        1,
        min(
            n_items,
            _MAX_POLYTOMOUS_CHUNK_ENTRIES // max(1, entries_per_item),
        ),
    )


def _polytomous_threshold_predictors(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    start: int,
    stop: int,
) -> NDArray[np.float64]:
    """Evaluate threshold predictors for one bounded item chunk."""
    if discrimination.ndim == 1:
        return discrimination[None, start:stop, None] * (
            theta[:, 0, None, None] - thresholds[None, start:stop, :]
        )
    item_discrimination = discrimination[start:stop]
    return (
        np.dot(theta, item_discrimination.T)[:, :, None]
        - np.sum(item_discrimination, axis=1)[None, :, None]
        * thresholds[None, start:stop, :]
    )


def _sample_item_categories(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Draw person-item categories while retaining item-major seed order."""
    cumulative = probabilities[:, :, :-1]
    np.cumsum(cumulative, axis=2, out=cumulative)
    uniforms = rng.random((probabilities.shape[1], probabilities.shape[0])).T
    return np.sum(
        uniforms[:, :, None] >= cumulative,
        axis=2,
        dtype=np.int_,
    )


def _should_use_rust() -> bool:
    """Resolve the global backend preference without eager backend imports."""
    from mirt._backend_config import should_use_rust

    return should_use_rust()


def _next_backend_seed(rng: np.random.Generator) -> int:
    """Derive a reproducible seed after any generated parameter draws."""
    return int(rng.integers(0, np.iinfo(np.int64).max))


def _gpcm_rust_safe(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    steps: NDArray[np.float64],
) -> bool:
    """Return whether exponentiating Rust GPCM logits cannot overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        largest_increment = np.max(np.abs(discrimination)) * (
            np.max(np.abs(theta)) + np.max(np.abs(steps))
        )
        largest_logit = largest_increment * (steps.shape[1] + 1)
    return bool(np.isfinite(largest_logit) and largest_logit < 700.0)


def _simulate_dichotomous(
    model: str,
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    n_persons = theta.shape[0]
    n_items = len(difficulty)
    n_factors = theta.shape[1]

    if guessing is None:
        if model in ("3PL", "4PL"):
            guessing = np.full(n_items, 0.2)
        else:
            guessing = np.zeros(n_items)
    else:
        guessing = _prepare_parameter("guessing", guessing, (n_items,))

    if upper is None:
        upper = np.ones(n_items)
    else:
        upper = _prepare_parameter("upper", upper, (n_items,))

    if np.any((guessing < 0.0) | (guessing > 1.0)):
        raise ValueError("guessing values must be between 0 and 1")
    if np.any((upper < 0.0) | (upper > 1.0)):
        raise ValueError("upper values must be between 0 and 1")
    if np.any(guessing > upper):
        raise ValueError("guessing values cannot exceed upper")

    if n_factors == 1 and model != "4PL" and _should_use_rust():
        from mirt import _rust_backend

        backend_guessing = guessing if model == "3PL" else None
        return np.asarray(
            _rust_backend.simulate_dichotomous(
                theta.ravel(),
                discrimination.ravel(),
                difficulty,
                backend_guessing,
                seed=_next_backend_seed(rng),
            ),
            dtype=np.int_,
        )

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
        theta_1d = theta.ravel()

        z = a[None, :] * (theta_1d[:, None] - difficulty[None, :])
    else:
        a = discrimination
        z = np.dot(theta, a.T) - np.sum(a, axis=1) * difficulty

    p_star = sigmoid(z)

    if model == "1PL" or model == "2PL":
        probs = p_star
    elif model == "3PL":
        probs = guessing[None, :] + (1 - guessing[None, :]) * p_star
    elif model == "4PL":
        probs = guessing[None, :] + (upper[None, :] - guessing[None, :]) * p_star
    else:
        probs = p_star

    u = rng.random((n_persons, n_items))
    responses = (u < probs).astype(np.int_)

    return responses


def _simulate_grm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_categories: int,
    thresholds: NDArray[np.float64] | None,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    n_persons = theta.shape[0]
    n_items = len(difficulty)
    n_factors = theta.shape[1]

    if thresholds is None:
        thresholds = (
            difficulty[:, None]
            + np.linspace(
                -1.5,
                1.5,
                n_categories - 1,
            )[None, :]
        )
    thresholds = _prepare_parameter(
        "thresholds",
        thresholds,
        (n_items, n_categories - 1),
    )
    if np.any(np.diff(thresholds, axis=1) < 0):
        raise ValueError("GRM thresholds must be ordered within each item")

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
    else:
        a = discrimination

    if n_factors == 1 and _should_use_rust():
        from mirt import _rust_backend

        return np.asarray(
            _rust_backend.simulate_grm(
                theta,
                a,
                thresholds,
                seed=_next_backend_seed(rng),
            ),
            dtype=np.int_,
        )

    responses = np.empty((n_persons, n_items), dtype=np.int_)
    chunk_size = _polytomous_item_chunk_size(
        n_persons,
        n_items,
        n_categories,
    )

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        logits = _polytomous_threshold_predictors(
            theta,
            a,
            thresholds,
            start,
            stop,
        )

        cumulative = sigmoid(logits)
        probabilities = np.empty(
            (n_persons, stop - start, n_categories),
            dtype=np.float64,
        )
        probabilities[:, :, 0] = 1.0 - cumulative[:, :, 0]
        probabilities[:, :, -1] = cumulative[:, :, -1]
        if n_categories > 2:
            probabilities[:, :, 1:-1] = cumulative[:, :, :-1] - cumulative[:, :, 1:]
        np.maximum(probabilities, 0.0, out=probabilities)
        totals = probabilities.sum(axis=2, keepdims=True)
        np.divide(
            probabilities,
            totals,
            out=probabilities,
            where=totals > 0,
        )
        responses[:, start:stop] = _sample_item_categories(probabilities, rng)

    return responses


def _simulate_gpcm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    n_categories: int,
    thresholds: NDArray[np.float64] | None,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    n_persons = theta.shape[0]
    n_items = discrimination.shape[0] if discrimination.ndim > 0 else 1
    n_factors = theta.shape[1]

    if thresholds is None:
        thresholds = np.tile(
            np.linspace(-1.0, 1.0, n_categories - 1),
            (n_items, 1),
        )
    thresholds = _prepare_parameter(
        "steps",
        thresholds,
        (n_items, n_categories - 1),
    )

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
    else:
        a = discrimination

    if n_factors == 1 and _gpcm_rust_safe(theta, a, thresholds) and _should_use_rust():
        from mirt import _rust_backend

        return np.asarray(
            _rust_backend.simulate_gpcm(
                theta.ravel(),
                a,
                thresholds,
                seed=_next_backend_seed(rng),
            ),
            dtype=np.int_,
        )

    responses = np.empty((n_persons, n_items), dtype=np.int_)
    chunk_size = _polytomous_item_chunk_size(
        n_persons,
        n_items,
        n_categories,
    )
    increment_limit = _MAX_STABLE_LOGIT / n_categories

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        with np.errstate(over="ignore", invalid="ignore"):
            increments = _polytomous_threshold_predictors(
                theta,
                a,
                thresholds,
                start,
                stop,
            )
        np.nan_to_num(
            increments,
            copy=False,
            nan=0.0,
            posinf=_MAX_STABLE_LOGIT,
            neginf=-_MAX_STABLE_LOGIT,
        )
        np.clip(
            increments,
            -increment_limit,
            increment_limit,
            out=increments,
        )
        logits = np.zeros(
            (n_persons, stop - start, n_categories),
            dtype=np.float64,
        )
        logits[:, :, 1:] = np.cumsum(increments, axis=2)
        probabilities = _stable_softmax(logits)
        responses[:, start:stop] = _sample_item_categories(probabilities, rng)

    return responses


def _simulate_nrm(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate nominal categories from category-specific linear predictors."""
    n_persons = theta.shape[0]
    n_items, n_categories = intercepts.shape
    n_factors = theta.shape[1]
    responses = np.empty((n_persons, n_items), dtype=np.int_)
    chunk_size = _polytomous_item_chunk_size(
        n_persons,
        n_items,
        n_categories,
    )

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        with np.errstate(over="ignore", invalid="ignore"):
            if n_factors == 1:
                logits = (
                    theta[:, 0, None, None] * slopes[None, start:stop, :]
                    + intercepts[None, start:stop, :]
                )
            else:
                logits = (
                    np.einsum(
                        "pf,icf->pic",
                        theta,
                        slopes[start:stop],
                        optimize=True,
                    )
                    + intercepts[None, start:stop, :]
                )
        probabilities = _stable_softmax(logits)
        responses[:, start:stop] = _sample_item_categories(probabilities, rng)

    return responses


def generate_item_parameters(
    n_items: int,
    model: SimulationModel = "2PL",
    n_factors: int = 1,
    n_categories: int = 2,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Generate random item parameters for an IRT model.

    Creates a dictionary of item parameters with realistic distributions
    suitable for simulation studies.

    Parameters
    ----------
    n_items : int
        Number of items to generate parameters for.
    model : {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}, default="2PL"
        IRT model type determining which parameters to generate.
    n_factors : int, default=1
        Number of latent factors for multidimensional models.
    n_categories : int, default=2
        Number of response categories for polytomous models.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with parameter arrays:

        - "discrimination": Item discrimination (a) parameters.
          Shape (n_items,) or (n_items, n_factors).
        - "difficulty": Item difficulty (b) parameters for dichotomous models.
          Shape (n_items,).
        - "thresholds": Category thresholds for polytomous models.
          Shape (n_items, n_categories-1).
        - "steps": Category steps for PCM. Shape
          (n_items, n_categories-1).
        - "slopes" and "intercepts": Category parameters for NRM.
        - "guessing": Lower asymptote (c) for 3PL/4PL. Shape (n_items,).
        - "upper": Upper asymptote (d) for 4PL. Shape (n_items,).

    Examples
    --------
    >>> from mirt import generate_item_parameters
    >>> params = generate_item_parameters(n_items=20, model="2PL", seed=42)
    >>> print(params.keys())
    dict_keys(['discrimination', 'difficulty'])

    >>> # Generate 3PL parameters
    >>> params = generate_item_parameters(n_items=15, model="3PL")
    >>> print(params['guessing'].mean())  # Average guessing parameter
    """
    model = model.upper()
    valid_models = {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}
    if model not in valid_models:
        raise ValueError(f"Unknown model: {model}")
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)):
        raise ValueError("n_items must be a positive integer")
    if isinstance(n_factors, bool) or not isinstance(n_factors, (int, np.integer)):
        raise ValueError("n_factors must be a positive integer")
    if isinstance(n_categories, bool) or not isinstance(
        n_categories, (int, np.integer)
    ):
        raise ValueError("n_categories must be a positive integer")
    if n_items < 1:
        raise ValueError("n_items must be a positive integer")
    if n_factors < 1:
        raise ValueError("n_factors must be a positive integer")
    if model in {"GRM", "GPCM", "PCM", "NRM"} and n_categories < 2:
        raise ValueError("n_categories must be at least 2")
    if model == "PCM" and n_factors != 1:
        raise ValueError("PCM only supports one factor")

    rng = np.random.default_rng(seed)

    params: dict[str, NDArray[np.float64]] = {}

    if model == "NRM":
        slope_shape = (
            (n_items, n_categories)
            if n_factors == 1
            else (n_items, n_categories, n_factors)
        )
        params["slopes"] = rng.normal(0.0, 0.6, size=slope_shape)
        params["slopes"][:, 0, ...] = 0.0
        params["intercepts"] = _default_nrm_intercepts(
            n_items,
            n_categories,
            rng,
        )
        return params

    if model == "PCM":
        params["discrimination"] = np.ones(n_items)
        params["steps"] = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            base = rng.normal(0, 1)
            params["steps"][i] = base + np.linspace(
                -1.5,
                1.5,
                n_categories - 1,
            )
        return params

    if model != "1PL":
        if n_factors == 1:
            params["discrimination"] = rng.lognormal(0, 0.3, size=n_items)
        else:
            params["discrimination"] = rng.lognormal(0, 0.3, size=(n_items, n_factors))
    else:
        params["discrimination"] = np.ones(n_items)

    if model in ("1PL", "2PL", "3PL", "4PL"):
        params["difficulty"] = rng.normal(0, 1, size=n_items)
    else:
        params["thresholds"] = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            base = rng.normal(0, 1)
            params["thresholds"][i] = base + np.linspace(-1.5, 1.5, n_categories - 1)

    if model in ("3PL", "4PL"):
        params["guessing"] = rng.uniform(0.1, 0.3, size=n_items)

    if model == "4PL":
        params["upper"] = rng.uniform(0.9, 1.0, size=n_items)

    return params
