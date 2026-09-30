"""Prepared IRTree fitting agrees with direct visited-node calculations."""

import gc
import tracemalloc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, logsumexp

import mirt.estimation._em_context as context_module
import mirt.estimation.irtree_em as tree_module
from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.constants import PROB_EPSILON
from mirt.estimation._irtree_context import IRTreeFitContext
from mirt.estimation.irtree_em import IRTreeEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.irtree import IRTreeModel


def _problem(spec, persons=43, items=3, points=5):
    model = IRTreeModel(items, tree_spec=spec)
    rng = np.random.default_rng(973)
    model.set_parameters(
        discrimination=rng.uniform(0.6, 1.4, (items, model.n_nodes)),
        difficulty=rng.uniform(-0.5, 0.5, (items, model.n_nodes)),
    )
    responses = rng.integers(-1, 5, (persons, items))
    responses[0] = -1
    pseudo, traits, valid = model.expand_to_pseudo_items(responses)
    quadrature = GaussHermiteQuadrature(points, model.n_traits)
    posterior = rng.uniform(0.1, 1.0, (persons, len(quadrature.nodes)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior[::7] *= 3
    estimator = IRTreeEMEstimator(n_quadpts=points, max_iter=3)
    estimator._quadrature = quadrature
    return model, responses, pseudo, traits, valid, quadrature, posterior, estimator


def _direct_counts(pseudo, valid, posterior):
    shape = (*pseudo.shape[1:], posterior.shape[1])
    correct, total = np.zeros(shape), np.zeros(shape)
    for person, item, node in np.ndindex(pseudo.shape):
        if valid[person, item, node]:
            total[item, node] += posterior[person]
            if pseudo[person, item, node] == 1:
                correct[item, node] += posterior[person]
    return correct, total


def _direct_likelihood(model, pseudo, traits, valid, points):
    result = np.zeros((len(pseudo), len(points)))
    for person, item, node in np.ndindex(pseudo.shape):
        if not valid[person, item, node]:
            continue
        a = model._parameters["discrimination"][item, node]
        b = model._parameters["difficulty"][item, node]
        p = np.clip(
            expit(a * (points[:, traits[item, node]] - b)),
            PROB_EPSILON,
            1 - PROB_EPSILON,
        )
        result[person] += np.log(p) if pseudo[person, item, node] == 1 else np.log1p(-p)
    return result


def _direct_uncertainty(model, traits, points, totals):
    result = {
        name: np.full_like(value, np.nan) for name, value in model._parameters.items()
    }
    for item, node in np.ndindex(model._parameters["discrimination"].shape):
        a = model._parameters["discrimination"][item, node]
        b = model._parameters["difficulty"][item, node]
        centered = points[:, traits[item, node]] - b
        p = np.clip(expit(a * centered), PROB_EPSILON, 1 - PROB_EPSILON)
        information = np.zeros((2, 2))
        for q, weight in enumerate(totals[item, node] * p * (1 - p)):
            score = np.array([centered[q], -a])
            information += weight * np.outer(score, score)
        if np.linalg.matrix_rank(information) < 2:
            continue
        variance = np.diag(np.linalg.pinv(information, rcond=1e-10))
        if np.all(np.isfinite(variance) & (variance > 0)):
            result["discrimination"][item, node] = np.sqrt(variance[0])
            result["difficulty"][item, node] = np.sqrt(variance[1])
    return result


@pytest.mark.parametrize(
    "spec", ["bockenholt", "extreme_midpoint", "direction_intensity"]
)
@pytest.mark.parametrize("blocked", [False, True])
def test_prepared_likelihood_counts_and_covariances_match_direct_nodes(
    spec, blocked, monkeypatch
):
    model, _, pseudo, traits, valid, quadrature, posterior, estimator = _problem(spec)
    # Explicit masks can suppress a stored decision. Traits can differ by item.
    valid[::3, 0, 0] = False
    traits = (traits + np.arange(model.n_items)[:, None]) % model.n_traits
    for value in (pseudo, traits, valid, quadrature.nodes, posterior):
        value.setflags(write=False)
    if blocked:
        monkeypatch.setattr(context_module, "_MAX_COUNT_ENTRIES", 19)
        monkeypatch.setattr(tree_module, "_MAX_IRTREE_SCRATCH_ENTRIES", 17)
    expected_correct, expected_total = _direct_counts(pseudo, valid, posterior)
    with IRTreeFitContext(pseudo, valid) as context:
        likelihood = estimator._compute_log_likelihoods(
            model, pseudo, traits, valid, quadrature.nodes, context=context
        )
        np.testing.assert_allclose(
            likelihood,
            _direct_likelihood(model, pseudo, traits, valid, quadrature.nodes),
            atol=1e-13,
        )
        correct, total = estimator._expected_counts(
            pseudo, valid, posterior, context=context
        )
        np.testing.assert_allclose(correct, expected_correct, atol=1e-13)
        np.testing.assert_allclose(total, expected_total, atol=1e-13)
        np.testing.assert_allclose(
            context.expected_totals(posterior),
            expected_total.reshape(-1, len(quadrature.nodes)),
            atol=1e-13,
        )
        estimator._fit_context = context
        actual = estimator._compute_standard_errors(
            model, pseudo, traits, valid, posterior
        )
        estimator._fit_context = None
        expected = _direct_uncertainty(model, traits, quadrature.nodes, expected_total)
        for name in expected:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=2e-12, atol=1e-12
            )


@pytest.mark.parametrize("spec", ["bockenholt", "direction_intensity"])
def test_e_step_matches_direct_shifted_correlated_prior(spec):
    model, _, pseudo, traits, valid, quadrature, _, estimator = _problem(spec)
    mean = np.linspace(-0.3, 0.4, model.n_traits)
    covariance = np.full((model.n_traits, model.n_traits), 0.2)
    np.fill_diagonal(covariance, 1.1)
    log_prior = gaussian_log_quadrature_mass(
        quadrature.nodes, quadrature.weights, mean, covariance
    )
    joint = (
        _direct_likelihood(model, pseudo, traits, valid, quadrature.nodes) + log_prior
    )
    marginal = logsumexp(joint, axis=1)
    expected = np.exp(joint - marginal[:, None])
    posterior, actual = estimator._e_step(
        model, pseudo, traits, valid, mean, covariance, return_log=True
    )
    np.testing.assert_allclose(posterior, expected, atol=2e-14)
    np.testing.assert_allclose(actual, marginal, atol=1e-13)
    _, ordinary = estimator._e_step(model, pseudo, traits, valid, mean, covariance)
    np.testing.assert_allclose(ordinary, np.exp(marginal), atol=1e-13)


@pytest.mark.parametrize("kind", ["huge", "float32", "view", "list"])
@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
def test_borrowed_likelihood_buffers_remain_intact_and_prior_is_preserved(
    kind, override, monkeypatch
):
    model, _, pseudo, traits, valid, quadrature, _, estimator = _problem(
        "direction_intensity", persons=3
    )
    values = np.full(
        (3, len(quadrature.nodes)),
        -1e300 if kind == "huge" else -2.5,
        dtype=np.float32 if kind == "float32" else np.float64,
    )
    values.setflags(write=False)
    cached = (
        values.tolist()
        if kind == "list"
        else np.broadcast_to(values[:1], values.shape)
        if kind == "view"
        else values
    )
    if override == "instance":
        estimator._compute_log_likelihoods = lambda *args: cached
    elif override == "class":
        monkeypatch.setattr(
            IRTreeEMEstimator,
            "_compute_log_likelihoods",
            staticmethod(lambda *args: cached),
        )
    else:

        class BorrowedEstimator(IRTreeEMEstimator):
            @staticmethod
            def _compute_log_likelihoods(*args):
                return cached

        estimator = BorrowedEstimator(n_quadpts=5)
        estimator._quadrature = quadrature
    mean, covariance = np.array([0.3, -0.2]), np.array([[1.1, 0.2], [0.2, 0.8]])
    prior = gaussian_log_quadrature_mass(
        quadrature.nodes, quadrature.weights, mean, covariance
    )
    expected = np.exp(prior - logsumexp(prior))
    posterior, marginal = estimator._e_step(
        model, pseudo, traits, valid, mean, covariance, return_log=True
    )
    np.testing.assert_allclose(
        posterior, np.broadcast_to(expected, posterior.shape), atol=1e-14
    )
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(np.isfinite(marginal))
    np.testing.assert_array_equal(cached, values)


def test_node_objective_gradient_matches_clipped_likelihood(monkeypatch):
    model, _, pseudo, traits, valid, _, posterior, estimator = _problem(
        "direction_intensity", points=3
    )
    theta = np.column_stack([np.linspace(-40, 40, posterior.shape[1])] * model.n_traits)
    estimator._quadrature = SimpleNamespace(nodes=theta)
    correct, totals = _direct_counts(pseudo, valid, posterior)
    calls = []

    def optimize(objective, x0, *, jac, **kwargs):
        assert jac
        item, node = divmod(len(calls), model.n_nodes)
        points = theta[:, traits[item, node]]

        def reference(params):
            p = np.clip(
                expit(params[0] * (points - params[1])), PROB_EPSILON, 1 - PROB_EPSILON
            )
            return -np.sum(
                correct[item, node] * np.log(p)
                + (totals[item, node] - correct[item, node]) * np.log1p(-p)
            )

        trial = np.array([4.0, -0.3])
        value, gradient = objective(trial)
        np.testing.assert_allclose(value, reference(trial), atol=1e-12)
        for coordinate in range(2):
            offset = np.zeros(2)
            offset[coordinate] = 1e-5
            numerical = (reference(trial + offset) - reference(trial - offset)) / 2e-5
            np.testing.assert_allclose(
                gradient[coordinate], numerical, rtol=2e-6, atol=1e-7
            )
        calls.append((item, node))
        return SimpleNamespace(x=np.asarray(x0))

    monkeypatch.setattr(tree_module, "minimize", optimize)
    estimator._m_step(model, pseudo, traits, valid, posterior)
    assert len(calls) == model.n_items * model.n_nodes


@pytest.mark.parametrize("empty", [False, True])
def test_unvisited_and_rank_deficient_nodes_keep_undefined_uncertainty(empty):
    model, _, pseudo, traits, valid, quadrature, posterior, estimator = _problem(
        "direction_intensity", points=1
    )
    if empty:
        valid[:] = False
    actual = estimator._compute_standard_errors(model, pseudo, traits, valid, posterior)
    for values in actual.values():
        assert np.all(np.isnan(values))


def test_fit_reuses_final_posterior_and_releases_response_preparation(monkeypatch):
    model, responses, _, _, _, _, _, _ = _problem("direction_intensity")
    contexts, evaluations, preparations = [], [], []
    original_normalize = tree_module.normalize_log_posterior
    original_components = IRTreeFitContext.response_components

    def normalize(*args, **kwargs):
        evaluations.append(True)
        return original_normalize(*args, **kwargs)

    def components(self, *args):
        if self._components is None:
            contexts.append(weakref.ref(self))
            preparations.append(True)
        return original_components(self, *args)

    monkeypatch.setattr(tree_module, "normalize_log_posterior", normalize)
    monkeypatch.setattr(IRTreeFitContext, "response_components", components)
    estimator = IRTreeEMEstimator(n_quadpts=3, max_iter=4, tol=1e12)
    result = estimator.fit(model, responses)
    assert result.converged and result.n_iterations == 2
    assert evaluations == [True, True]
    assert preparations == [True]
    assert estimator._fit_context is None
    gc.collect()
    assert all(reference() is None for reference in contexts)


def test_custom_e_step_retains_its_final_callback_and_failed_fit_cleans_context(
    monkeypatch,
):
    model, responses, _, _, _, _, _, estimator = _problem("direction_intensity")
    original = estimator._e_step
    calls = []

    def e_step(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    estimator._e_step = e_step
    estimator.tol = 1e12
    estimator.fit(model, responses)
    assert len(calls) == 3

    def fail(*args, **kwargs):
        raise RuntimeError("failed node trial")

    monkeypatch.setattr(tree_module, "minimize", fail)
    with pytest.raises(RuntimeError, match="failed node trial"):
        estimator.fit(model, responses)
    assert estimator._fit_context is None


def test_e_step_peak_is_near_one_posterior_buffer():
    model, _, pseudo, traits, valid, _, _, estimator = _problem(
        "bockenholt", persons=2000, items=3, points=7
    )
    tracemalloc.start()
    try:
        posterior, _ = estimator._e_step(
            model,
            pseudo,
            traits,
            valid,
            np.zeros(model.n_traits),
            np.eye(model.n_traits),
            return_log=True,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * posterior.nbytes


class _DirectEstimator(IRTreeEMEstimator):
    @staticmethod
    def _compute_log_likelihoods(model, pseudo, traits, valid, theta):
        return _direct_likelihood(model, pseudo, traits, valid, theta)

    @staticmethod
    def _expected_counts(pseudo, valid, posterior):
        return _direct_counts(pseudo, valid, posterior)

    def _e_step(
        self, model, pseudo, traits, valid, mean, covariance, *, return_log=False
    ):
        prior = gaussian_log_quadrature_mass(
            self._quadrature.nodes, self._quadrature.weights, mean, covariance
        )
        joint = (
            _direct_likelihood(model, pseudo, traits, valid, self._quadrature.nodes)
            + prior
        )
        marginal = logsumexp(joint, axis=1)
        return np.exp(joint - marginal[:, None]), marginal if return_log else np.exp(
            marginal
        )

    def _compute_standard_errors(self, model, pseudo, traits, valid, posterior):
        _, totals = _direct_counts(pseudo, valid, posterior)
        return _direct_uncertainty(model, traits, self._quadrature.nodes, totals)


@pytest.mark.parametrize(
    "spec", ["bockenholt", "extreme_midpoint", "direction_intensity"]
)
def test_complete_fit_matches_direct_node_statistics_and_prior(spec):
    model, responses, _, _, _, _, _, _ = _problem(spec, persons=101, items=2, points=3)
    actual = IRTreeEMEstimator(n_quadpts=3, max_iter=3, tol=1e-8).fit(
        model.copy(), responses
    )
    expected = _DirectEstimator(n_quadpts=3, max_iter=3, tol=1e-8).fit(
        model.copy(), responses
    )
    assert actual.n_iterations == expected.n_iterations
    assert actual.n_parameters == expected.n_parameters
    np.testing.assert_allclose(
        actual.log_likelihood, expected.log_likelihood, atol=1e-6
    )
    for field in ("trait_means", "trait_covariance", "theta_estimates", "theta_se"):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=2e-5, atol=1e-6
        )
    for name in actual.model._parameters:
        np.testing.assert_allclose(
            actual.model._parameters[name],
            expected.model._parameters[name],
            rtol=2e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual.standard_errors[name],
            expected.standard_errors[name],
            rtol=2e-5,
            atol=1e-6,
            equal_nan=True,
        )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_direct_and_weighted_counts_keep_explicit_node_masks(weighted, cached):
    _, _, pseudo, _, valid, _, posterior, _ = _problem("bockenholt")
    valid[::3, 0, 0] = False
    weights = np.linspace(0.25, 2.0, len(pseudo)) if weighted else None
    expected_posterior = posterior if weights is None else posterior * weights[:, None]
    expected_correct, expected_total = _direct_counts(pseudo, valid, expected_posterior)
    for value in (pseudo, valid, posterior):
        value.setflags(write=False)
    if weights is not None:
        weights.setflags(write=False)
    with IRTreeFitContext(pseudo, valid) as context:
        correct, total = context.expected_counts(
            posterior, weights, cache_components=cached
        )
        np.testing.assert_allclose(
            correct, expected_correct.reshape(correct.shape), atol=1e-13
        )
        np.testing.assert_allclose(
            total, expected_total.reshape(total.shape), atol=1e-13
        )
