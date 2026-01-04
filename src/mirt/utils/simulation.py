from typing import Literal

import numpy as np
from numpy.typing import NDArray


def simdata(
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM"] = "2PL",
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
) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed)

    if theta is None:
        if n_factors == 1:
            theta = rng.standard_normal(n_persons)
        else:
            theta = rng.standard_normal((n_persons, n_factors))
    else:
        theta = np.asarray(theta)
        n_persons = theta.shape[0]

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

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
        guessing = np.asarray(guessing)

    if upper is None:
        upper = np.ones(n_items)
    else:
        upper = np.asarray(upper)

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
        theta_1d = theta.ravel()

        z = a[None, :] * (theta_1d[:, None] - difficulty[None, :])
    else:
        a = discrimination
        z = np.dot(theta, a.T) - np.sum(a, axis=1) * difficulty

    p_star = 1.0 / (1.0 + np.exp(-z))

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
        thresholds = np.zeros((n_items, n_categories - 1))
        for i in range(n_items):
            thresholds[i] = difficulty[i] + np.linspace(-1.5, 1.5, n_categories - 1)
    else:
        thresholds = np.asarray(thresholds)

    if n_factors == 1:
        a = discrimination if discrimination.ndim == 1 else discrimination.ravel()
    else:
        a = discrimination

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        cum_probs = np.ones((n_persons, n_categories))

        for k in range(n_categories - 1):
            if n_factors == 1:
                z = a[i] * (theta.ravel() - thresholds[i, k])
            else:
                z = np.dot(theta, a[i]) - np.sum(a[i]) * thresholds[i, k]

            cum_probs[:, k + 1] = 1.0 / (1.0 + np.exp(-z))

        cat_probs = np.diff(
            np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1
        )
        cat_probs = np.maximum(cat_probs, 0)
        cat_probs = cat_probs / cat_probs.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

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
        numerators = np.zeros((n_persons, n_categories))

        for k in range(n_categories):
            cumsum = 0.0
            for v in range(k):
                if n_factors == 1:
                    cumsum += a[i] * (theta.ravel() - thresholds[i, v])
                else:
                    cumsum += np.dot(theta, a[i]) - np.sum(a[i]) * thresholds[i, v]

            numerators[:, k] = np.exp(cumsum)

        cat_probs = numerators / numerators.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses


def generate_item_parameters(
    n_items: int,
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM"] = "2PL",
    n_factors: int = 1,
    n_categories: int = 2,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)

    params: dict[str, NDArray[np.float64]] = {}

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
