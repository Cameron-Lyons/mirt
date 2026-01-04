"""Data simulation utilities for IRT models."""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray


def simdata(
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM"] = "2PL",
    n_persons: int = 500,
    n_items: int = 20,
    n_categories: int = 2,
    n_factors: int = 1,
    theta: Optional[NDArray[np.float64]] = None,
    discrimination: Optional[NDArray[np.float64]] = None,
    difficulty: Optional[NDArray[np.float64]] = None,
    guessing: Optional[NDArray[np.float64]] = None,
    upper: Optional[NDArray[np.float64]] = None,
    thresholds: Optional[NDArray[np.float64]] = None,
    seed: Optional[int] = None,
) -> NDArray[np.int_]:
    """Simulate response data from IRT models.

    Parameters
    ----------
    model : {'1PL', '2PL', '3PL', '4PL', 'GRM', 'GPCM'}, default='2PL'
        IRT model to use for simulation.
    n_persons : int, default=500
        Number of persons to simulate.
    n_items : int, default=20
        Number of items.
    n_categories : int, default=2
        Number of response categories (2 for dichotomous, >2 for polytomous).
    n_factors : int, default=1
        Number of latent factors.
    theta : ndarray, optional
        Person ability values. If None, sampled from N(0, 1).
        Shape should be (n_persons,) or (n_persons, n_factors).
    discrimination : ndarray, optional
        Item discrimination parameters. If None, sampled from LogN(0, 0.25).
    difficulty : ndarray, optional
        Item difficulty parameters. If None, sampled from N(0, 1).
    guessing : ndarray, optional
        Guessing parameters for 3PL/4PL. If None, set to 0.2.
    upper : ndarray, optional
        Upper asymptote for 4PL. If None, set to 1.0.
    thresholds : ndarray, optional
        Threshold parameters for polytomous models.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Simulated response matrix.

    Examples
    --------
    >>> # Simulate 2PL data
    >>> responses = simdata(model='2PL', n_persons=500, n_items=20)
    >>> print(responses.shape)
    (500, 20)

    >>> # Simulate 3PL with specific parameters
    >>> a = np.random.lognormal(0, 0.3, size=20)
    >>> b = np.random.normal(0, 1, size=20)
    >>> c = np.full(20, 0.2)
    >>> responses = simdata(model='3PL', discrimination=a, difficulty=b, guessing=c)

    >>> # Simulate GRM (graded response) data with 5 categories
    >>> responses = simdata(model='GRM', n_categories=5)
    """
    rng = np.random.default_rng(seed)

    # Generate theta if not provided
    if theta is None:
        if n_factors == 1:
            theta = rng.standard_normal(n_persons)
        else:
            theta = rng.standard_normal((n_persons, n_factors))
    else:
        theta = np.asarray(theta)
        n_persons = theta.shape[0]

    # Ensure theta is 2D
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    # Generate item parameters if not provided
    if discrimination is None:
        if n_factors == 1:
            discrimination = rng.lognormal(0, 0.25, size=n_items)
        else:
            discrimination = rng.lognormal(0, 0.25, size=(n_items, n_factors))
    else:
        discrimination = np.asarray(discrimination)

    if difficulty is None:
        difficulty = rng.normal(0, 1, size=n_items)
    else:
        difficulty = np.asarray(difficulty)

    # Model-specific simulation
    if model in ("1PL", "2PL", "3PL", "4PL"):
        return _simulate_dichotomous(
            model=model,
            theta=theta,
            discrimination=discrimination,
            difficulty=difficulty,
            guessing=guessing,
            upper=upper,
            rng=rng,
        )
    elif model == "GRM":
        return _simulate_grm(
            theta=theta,
            discrimination=discrimination,
            difficulty=difficulty,
            n_categories=n_categories,
            thresholds=thresholds,
            rng=rng,
        )
    elif model == "GPCM":
        return _simulate_gpcm(
            theta=theta,
            discrimination=discrimination,
            n_categories=n_categories,
            thresholds=thresholds,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def _simulate_dichotomous(
    model: str,
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: Optional[NDArray[np.float64]],
    upper: Optional[NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate dichotomous (0/1) response data."""
    n_persons = theta.shape[0]
    n_items = len(difficulty)
    n_factors = theta.shape[1]

    # Default guessing and upper asymptote
    if guessing is None:
        if model in ("3PL", "4PL"):
            guessing = np.full(n_items, 0.2)
        else:
            guessing = np.zeros(n_items)
    else:
        guessing = np.asarray(guessing)

    if upper is None:
        upper = np.ones(n_items)
    else:
        upper = np.asarray(upper)

    # Compute probabilities
    if n_factors == 1:
        # Unidimensional
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
        theta_1d = theta.ravel()

        # z = a * (theta - b)
        z = a[None, :] * (theta_1d[:, None] - difficulty[None, :])
    else:
        # Multidimensional: z = a·θ - a_sum * b
        a = discrimination  # (n_items, n_factors)
        z = np.dot(theta, a.T) - np.sum(a, axis=1) * difficulty

    # Logistic function
    p_star = 1.0 / (1.0 + np.exp(-z))

    # Apply lower/upper asymptotes based on model
    if model == "1PL" or model == "2PL":
        probs = p_star
    elif model == "3PL":
        probs = guessing[None, :] + (1 - guessing[None, :]) * p_star
    elif model == "4PL":
        probs = guessing[None, :] + (upper[None, :] - guessing[None, :]) * p_star
    else:
        probs = p_star

    # Generate responses
    u = rng.random((n_persons, n_items))
    responses = (u < probs).astype(np.int_)

    return responses


def _simulate_grm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_categories: int,
    thresholds: Optional[NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate Graded Response Model data."""
    n_persons = theta.shape[0]
    n_items = len(difficulty)
    n_factors = theta.shape[1]

    # Generate thresholds if not provided
    if thresholds is None:
        # Create evenly spaced thresholds around difficulty
        thresholds = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            thresholds[i] = difficulty[i] + np.linspace(-1.5, 1.5, n_categories - 1)
    else:
        thresholds = np.asarray(thresholds)

    # Ensure discrimination is appropriate shape
    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
    else:
        a = discrimination

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        # Compute cumulative probabilities P*(X >= k)
        cum_probs = np.ones((n_persons, n_categories))

        for k in range(n_categories - 1):
            if n_factors == 1:
                z = a[i] * (theta.ravel() - thresholds[i, k])
            else:
                z = np.dot(theta, a[i]) - np.sum(a[i]) * thresholds[i, k]

            cum_probs[:, k + 1] = 1.0 / (1.0 + np.exp(-z))

        # Convert to category probabilities
        cat_probs = np.diff(np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1)
        cat_probs = np.maximum(cat_probs, 0)  # Ensure non-negative
        cat_probs = cat_probs / cat_probs.sum(axis=1, keepdims=True)  # Normalize

        # Sample responses
        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses


def _simulate_gpcm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    n_categories: int,
    thresholds: Optional[NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate Generalized Partial Credit Model data."""
    n_persons = theta.shape[0]
    n_items = discrimination.shape[0] if discrimination.ndim > 0 else 1
    n_factors = theta.shape[1]

    # Generate step parameters if not provided
    if thresholds is None:
        thresholds = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            thresholds[i] = np.linspace(-1, 1, n_categories - 1)
    else:
        thresholds = np.asarray(thresholds)

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
    else:
        a = discrimination

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        # Compute category probabilities
        numerators = np.zeros((n_persons, n_categories))

        for k in range(n_categories):
            cumsum = 0.0
            for v in range(k):
                if n_factors == 1:
                    cumsum += a[i] * (theta.ravel() - thresholds[i, v])
                else:
                    cumsum += np.dot(theta, a[i]) - np.sum(a[i]) * thresholds[i, v]

            numerators[:, k] = np.exp(cumsum)

        # Normalize
        cat_probs = numerators / numerators.sum(axis=1, keepdims=True)

        # Sample responses
        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses


def generate_item_parameters(
    n_items: int,
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM"] = "2PL",
    n_factors: int = 1,
    n_categories: int = 2,
    seed: Optional[int] = None,
) -> dict[str, NDArray[np.float64]]:
    """Generate random item parameters for a specified model.

    Parameters
    ----------
    n_items : int
        Number of items.
    model : str, default='2PL'
        IRT model type.
    n_factors : int, default=1
        Number of latent factors.
    n_categories : int, default=2
        Number of response categories (for polytomous models).
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary of parameter arrays.
    """
    rng = np.random.default_rng(seed)

    params: dict[str, NDArray[np.float64]] = {}

    # Discrimination
    if model != "1PL":
        if n_factors == 1:
            params["discrimination"] = rng.lognormal(0, 0.3, size=n_items)
        else:
            params["discrimination"] = rng.lognormal(0, 0.3, size=(n_items, n_factors))
    else:
        params["discrimination"] = np.ones(n_items)

    # Difficulty/thresholds
    if model in ("1PL", "2PL", "3PL", "4PL"):
        params["difficulty"] = rng.normal(0, 1, size=n_items)
    else:
        # Polytomous: generate ordered thresholds
        params["thresholds"] = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            base = rng.normal(0, 1)
            params["thresholds"][i] = base + np.linspace(-1.5, 1.5, n_categories - 1)

    # Guessing (3PL, 4PL)
    if model in ("3PL", "4PL"):
        params["guessing"] = rng.uniform(0.1, 0.3, size=n_items)

    # Upper asymptote (4PL)
    if model == "4PL":
        params["upper"] = rng.uniform(0.9, 1.0, size=n_items)

    return params
