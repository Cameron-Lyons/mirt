"""Native multigroup E-steps agree with per-group single-group likelihoods."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp

from mirt._backend_state import get_backend_preference, set_backend_preference
from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt._rust_backend import RUST_AVAILABLE
from mirt.backends.rust import likelihood, multigroup, polytomous
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.polytomous import NominalResponseModel

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")

N_ITEMS = 6
CATEGORIES = np.array([2, 3, 4, 3, 2, 4], dtype=np.int32)
MEANS = np.array([0.3, -0.4])
VARIANCES = np.array([0.8, 1.3])
QUADRATURE = GaussHermiteQuadrature(n_points=21)
POINTS = QUADRATURE.nodes.ravel()
WEIGHTS = QUADRATURE.weights


@pytest.fixture(autouse=True)
def _native_backend() -> Iterator[None]:
    previous = get_backend_preference()
    set_backend_preference("auto")
    try:
        yield
    finally:
        set_backend_preference(previous)


def _responses(n_persons: int, categories: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    responses = (rng.random((n_persons, categories.size)) * categories).astype(np.int32)
    responses[rng.random(responses.shape) < 0.1] = -1
    return responses


def _reference(
    log_likelihoods: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    posteriors = []
    group_lls = []
    for group, table in enumerate(log_likelihoods):
        log_mass = gaussian_log_quadrature_mass(
            POINTS, WEIGHTS, MEANS[group : group + 1], VARIANCES[group].reshape(1, 1)
        )
        log_joint = table + log_mass[None, :]
        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        posteriors.append(np.exp(log_joint - log_marginal))
        group_lls.append(float(log_marginal.sum()))
    return posteriors, np.array(group_lls)


def _dichotomous_case(guessing: bool) -> tuple[Callable[[], Any], list[np.ndarray]]:
    rng = np.random.default_rng(11)
    binary = np.full(N_ITEMS, 2)
    responses = [_responses(40, binary, 1), _responses(25, binary, 2)]
    # A steep item makes |z| exceed 23 on the outer nodes.
    disc = [np.r_[9.0, rng.uniform(0.6, 1.8, N_ITEMS - 1)] for _ in MEANS]
    diff = [np.r_[3.0, rng.normal(0.0, 1.0, N_ITEMS - 1)] for _ in MEANS]
    if not guessing:
        tables = [
            likelihood.compute_log_likelihoods_2pl(r, POINTS, a, b)
            for r, a, b in zip(responses, disc, diff, strict=True)
        ]
        return (
            lambda: multigroup.multigroup_e_step_2pl(
                responses, POINTS, WEIGHTS, disc, diff, MEANS, VARIANCES
            ),
            tables,
        )
    guess = [rng.uniform(0.05, 0.3, N_ITEMS) for _ in MEANS]
    tables = [
        likelihood.compute_log_likelihoods_3pl(r, POINTS, a, b, c)
        for r, a, b, c in zip(responses, disc, diff, guess, strict=True)
    ]
    return (
        lambda: multigroup.multigroup_e_step_3pl(
            responses, POINTS, WEIGHTS, disc, diff, guess, MEANS, VARIANCES
        ),
        tables,
    )


def _polytomous_case(model: str) -> tuple[Callable[[], Any], list[np.ndarray]]:
    rng = np.random.default_rng(23)
    responses = [_responses(40, CATEGORIES, 3), _responses(25, CATEGORIES, 4)]
    categories = [CATEGORIES.copy() for _ in MEANS]
    max_categories = int(CATEGORIES.max())
    if model == "nrm":
        groups = []
        for _ in MEANS:
            nrm = NominalResponseModel(N_ITEMS, n_categories=CATEGORIES.tolist())
            slopes = np.zeros((N_ITEMS, max_categories))
            intercepts = np.zeros((N_ITEMS, max_categories))
            for item, n_cat in enumerate(CATEGORIES):
                slopes[item, 1:n_cat] = np.sort(rng.uniform(0.3, 2.0, n_cat - 1))
                intercepts[item, 1:n_cat] = rng.normal(0.0, 1.0, n_cat - 1)
            nrm.set_parameters(slopes=slopes, intercepts=intercepts)
            groups.append(nrm)
        tables = [
            nrm.log_likelihood_batch(r, POINTS)
            for nrm, r in zip(groups, responses, strict=True)
        ]
        return (
            lambda: multigroup.multigroup_e_step_nrm(
                responses,
                POINTS,
                WEIGHTS,
                [nrm.slopes for nrm in groups],
                [nrm.intercepts for nrm in groups],
                categories,
                MEANS,
                VARIANCES,
            ),
            tables,
        )

    disc = [rng.uniform(0.6, 2.0, N_ITEMS) for _ in MEANS]
    if model == "grm":
        thresholds = [
            np.sort(rng.normal(0.0, 1.0, (N_ITEMS, max_categories - 1)), axis=1)
            for _ in MEANS
        ]
        tables = [
            polytomous.compute_log_likelihoods_grm(r, POINTS, a, b, CATEGORIES)
            for r, a, b in zip(responses, disc, thresholds, strict=True)
        ]
        return (
            lambda: multigroup.multigroup_e_step_grm(
                responses,
                POINTS,
                WEIGHTS,
                disc,
                thresholds,
                categories,
                MEANS,
                VARIANCES,
            ),
            tables,
        )
    steps = [rng.normal(0.0, 1.0, (N_ITEMS, max_categories)) for _ in MEANS]
    tables = [
        polytomous.compute_log_likelihoods_gpcm(r, POINTS, a, s, CATEGORIES)
        for r, a, s in zip(responses, disc, steps, strict=True)
    ]
    return (
        lambda: multigroup.multigroup_e_step_gpcm(
            responses, POINTS, WEIGHTS, disc, steps, categories, MEANS, VARIANCES
        ),
        tables,
    )


CASES = {
    "2pl": lambda: _dichotomous_case(guessing=False),
    "3pl": lambda: _dichotomous_case(guessing=True),
    "grm": lambda: _polytomous_case("grm"),
    "gpcm": lambda: _polytomous_case("gpcm"),
    "nrm": lambda: _polytomous_case("nrm"),
}


@pytest.mark.parametrize("model", list(CASES))
def test_multigroup_e_step_matches_single_group_likelihoods(model: str) -> None:
    run, tables = CASES[model]()
    expected_posteriors, expected_lls = _reference(tables)

    result = run()

    assert result is not None
    posteriors, group_lls = result
    assert len(posteriors) == len(tables)
    for posterior, expected in zip(posteriors, expected_posteriors, strict=True):
        assert posterior.flags.c_contiguous and posterior.dtype == np.float64
        np.testing.assert_allclose(posterior, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(group_lls, expected_lls, rtol=1e-12)


def test_steep_dichotomous_items_use_exact_log_probabilities() -> None:
    # The former kernel clipped each probability at 1e-10 before the logarithm,
    # so this pattern, unlikely at every node, scored at least 2 * log(1e-10).
    responses = np.array([[0, 1]], dtype=np.int32)
    disc, diff = np.array([9.0, 9.0]), np.array([-3.0, 3.0])
    result = multigroup.multigroup_e_step_2pl(
        [responses], POINTS, WEIGHTS, [disc], [diff], np.zeros(1), np.ones(1)
    )
    assert result is not None
    table = likelihood.compute_log_likelihoods_2pl(responses, POINTS, disc, diff)
    log_mass = gaussian_log_quadrature_mass(POINTS, WEIGHTS, np.zeros(1), np.eye(1))

    assert table.max() < 2.0 * np.log(1e-10)
    np.testing.assert_allclose(result[1], [logsumexp(table[0] + log_mass)], rtol=1e-12)


def test_multigroup_expected_counts_match_posterior_products() -> None:
    binary = np.full(N_ITEMS, 2)
    responses = [_responses(40, binary, 5), _responses(25, binary, 6)]
    rng = np.random.default_rng(9)
    posteriors = [rng.dirichlet(np.ones(POINTS.size), len(r)) for r in responses]
    # Strided posteriors must be handled like contiguous ones.
    strided = [np.asfortranarray(posterior) for posterior in posteriors]

    result = multigroup.multigroup_expected_counts(responses, strided)

    assert result is not None
    for r_k, n_k, group_responses, posterior in zip(
        *result, responses, posteriors, strict=True
    ):
        np.testing.assert_allclose(
            r_k, (group_responses == 1).T.astype(float) @ posterior, atol=1e-13
        )
        np.testing.assert_allclose(
            n_k, (group_responses >= 0).T.astype(float) @ posterior, atol=1e-13
        )


@pytest.mark.parametrize(
    ("posteriors", "message"),
    [
        ([np.full((3, POINTS.size), 1.0 / POINTS.size)], "one row per response row"),
        ([], "one entry per response matrix"),
    ],
)
def test_multigroup_expected_counts_validate_posteriors(
    posteriors: list[np.ndarray], message: str
) -> None:
    from mirt import mirt_rs

    responses = [np.array([[0, 1], [1, -1]], dtype=np.int32)]

    with pytest.raises(ValueError, match=message):
        mirt_rs.multigroup_expected_counts(responses, posteriors)


@pytest.mark.parametrize("model", ["grm", "gpcm", "nrm"])
def test_out_of_range_categories_raise_index_error(model: str) -> None:
    from mirt import mirt_rs

    responses = [np.array([[0, 3]], dtype=np.int32)]
    categories = [np.array([2, 3], dtype=np.int32)]
    parameters = {
        "grm": ([np.ones(2)], [np.zeros((2, 2))]),
        "gpcm": ([np.ones(2)], [np.zeros((2, 3))]),
        "nrm": ([np.zeros((2, 3))], [np.zeros((2, 3))]),
    }[model]
    kernel = getattr(mirt_rs, f"multigroup_e_step_{model}")

    with pytest.raises(IndexError, match="category range"):
        kernel(
            responses,
            POINTS,
            WEIGHTS,
            *parameters,
            categories,
            np.zeros(1),
            np.ones(1),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"disc": [np.ones(3)]}, "one entry per item"),
        ({"disc": [np.ones(2), np.ones(2)]}, "one entry per response matrix"),
        ({"means": np.zeros(2)}, "one entry per response matrix"),
        ({"variances": np.zeros(1)}, "variances finite and positive"),
        ({"weights": WEIGHTS[:-1]}, "same length"),
    ],
)
def test_invalid_group_inputs_raise_value_error(
    change: dict[str, Any], message: str
) -> None:
    from mirt import mirt_rs

    arguments = {
        "responses": [np.array([[0, 1]], dtype=np.int32)],
        "points": POINTS,
        "weights": WEIGHTS,
        "disc": [np.ones(2)],
        "diff": [np.zeros(2)],
        "means": np.zeros(1),
        "variances": np.ones(1),
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        mirt_rs.multigroup_e_step_2pl(*arguments.values())


def test_polytomous_parameter_shapes_are_validated() -> None:
    from mirt import mirt_rs

    with pytest.raises(ValueError, match="incompatible item parameters"):
        mirt_rs.multigroup_e_step_grm(
            [np.array([[0, 2]], dtype=np.int32)],
            POINTS,
            WEIGHTS,
            [np.ones(2)],
            [np.zeros((2, 1))],
            [np.array([2, 3], dtype=np.int32)],
            np.zeros(1),
            np.ones(1),
        )
