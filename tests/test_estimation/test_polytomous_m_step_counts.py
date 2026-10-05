"""All-item category counts and tolerances for polytomous EM M-steps."""

from copy import deepcopy

import numpy as np
import pytest

import mirt
from mirt import simdata
from mirt.estimation import _em_context
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)

native = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native backend unavailable"
)


@pytest.fixture(autouse=True)
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


@pytest.mark.parametrize("chunked", [False, True])
def test_category_counts_match_itemwise_counts(monkeypatch, chunked):
    rng = np.random.default_rng(5)
    n_categories = [2, 5, 3, 4]
    responses = np.column_stack([rng.integers(-1, k, 300) for k in n_categories])
    # Codes outside an item's categories must not leak into the next item.
    responses[::7, 2] = 3
    posterior = rng.dirichlet(np.ones(9), 300)
    if chunked:
        monkeypatch.setattr(_em_context, "_MAX_COUNT_ENTRIES", 20)
    context = EMFitContext(responses)

    counts = context.category_counts(n_categories, posterior)

    assert len(counts) == len(n_categories)
    for item_idx, n_cat in enumerate(n_categories):
        expected = context.expected_category_counts(item_idx, n_cat, posterior)
        assert counts[item_idx].shape == (9, n_cat)
        assert counts[item_idx].flags.c_contiguous
        np.testing.assert_allclose(counts[item_idx], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "factory", [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel]
)
def test_generic_polytomous_m_step_matches_itemwise_counts(factory):
    rng = np.random.default_rng(31)
    n_categories = [3, 4, 3]
    model = factory(3, n_categories=n_categories, n_factors=2)
    responses = np.column_stack([rng.integers(-1, k, 200) for k in n_categories])
    estimator = EMEstimator(n_quadpts=5, use_rust=False, item_optim_ftol=1e-12)
    estimator._quadrature = GaussHermiteQuadrature(5, 2)
    posterior = rng.dirichlet(np.ones(25), 200)
    reference = deepcopy(model)

    estimator._m_step(model, responses, posterior)
    for item_idx in range(reference.n_items):
        # Without counts the item optimizer accumulates its own indicators.
        optimal = estimator._optimize_item_params(
            reference, item_idx, responses, posterior, estimator._quadrature.nodes
        )
        estimator._set_item_params(reference, item_idx, optimal)

    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=1e-6)


@native
@pytest.mark.parametrize(
    ("factory", "requested", "precise", "expected"),
    [
        (GradedResponseModel, 1e-6, False, 1e-10),
        (GradedResponseModel, 1e-12, False, 1e-12),
        (GeneralizedPartialCredit, 1e-6, False, 1e-6),
        (GeneralizedPartialCredit, 1e-6, True, 1e-10),
    ],
)
def test_native_polytomous_m_step_tolerance(
    monkeypatch, factory, requested, precise, expected
):
    from mirt.backends.rust import polytomous_mstep

    seen = []

    def capture(*args, ftol, **kwargs):
        seen.append(ftol)
        return True

    monkeypatch.setattr(polytomous_mstep, "try_polytomous_m_step", capture)
    mirt.set_backend("rust")
    estimator = EMEstimator(n_quadpts=5, item_optim_ftol=requested)
    estimator._quadrature = GaussHermiteQuadrature(5)
    # SQUAREM requests precise M-steps for every model.
    estimator._precise_m_steps = precise
    responses = np.array([[0, 2], [1, 1], [2, 0]])
    estimator._m_step(factory(2, n_categories=3), responses, np.full((3, 5), 0.2))
    assert seen == [expected]


@native
def test_native_grm_em_converges_without_m_step_jitter():
    mirt.set_backend("rust")
    responses = simdata("GRM", n_persons=1000, n_items=15, n_categories=5, seed=1)
    reference = GradedResponseModel(15, n_categories=5)
    EMEstimator(
        tol=1e-8, max_iter=5000, item_optim_ftol=1e-12, compute_standard_errors=False
    ).fit(reference, responses)

    model = GradedResponseModel(15, n_categories=5)
    result = EMEstimator(compute_standard_errors=False).fit(model, responses)

    # A relative 1e-6 M-step stopped after about 100 noisy iterations with
    # parameters 5e-2 from the optimum.
    assert result.converged
    assert result.n_iterations <= 50
    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=5e-3)
