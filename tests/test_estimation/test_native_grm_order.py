"""The native graded M-step keeps thresholds ordered like the generic optimizer.

Disordered thresholds give negative middle-category probabilities; clipping
them inflated the reported log likelihood above the generating model's.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt
from mirt import mirt_rs
from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step
from mirt.estimation.em import EMEstimator
from mirt.models.polytomous import GradedResponseModel

pytestmark = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native backend unavailable"
)


@pytest.fixture(autouse=True)
def rust_backend():
    previous = mirt.get_backend()
    mirt.set_backend("rust")
    yield
    mirt.set_backend(previous)


def _marginal_log_likelihood(model, data):
    """Unclipped marginal log likelihood on a fine standard-normal grid."""
    points = np.linspace(-6.0, 6.0, 121)
    log_weights = -0.5 * points**2
    log_weights -= logsumexp(log_weights)
    probabilities = np.asarray(model.probability(points[:, None]))
    assert np.all(probabilities > 0)
    log_p = np.log(probabilities)
    total = np.zeros((data.shape[0], points.size))
    for item in range(data.shape[1]):
        total += log_p[:, item, data[:, item]].T
    return float(np.sum(logsumexp(total + log_weights, axis=1)))


def _assert_ordered(model):
    thresholds = model.parameters["thresholds"]
    assert np.all(np.diff(thresholds, axis=1) > 0), thresholds


def test_close_thresholds_match_the_generic_fit():
    rng = np.random.default_rng(0)
    close = GradedResponseModel(1, n_categories=4)
    close.set_parameters(
        discrimination=np.array([1.45]),
        thresholds=np.array([[-1.43110298, -0.96827733, -0.91293407]]),
    )
    others = GradedResponseModel(4, n_categories=4)
    others.set_parameters(
        discrimination=np.array([1.0, 1.9, 2.0, 1.0]),
        thresholds=np.array(
            [
                [-0.6, 0.26, 1.29],
                [-1.74, -1.22, 0.17],
                [-0.7, -0.58, 2.25],
                [-0.15, 0.46, 1.12],
            ]
        ),
    )
    theta = rng.standard_normal((3000, 1))
    data = np.column_stack(
        [close.simulate(theta, seed=1), others.simulate(theta, seed=2)]
    )
    native, generic = (
        EMEstimator(use_rust=use_rust).fit(GradedResponseModel(5, n_categories=4), data)
        for use_rust in (True, False)
    )
    _assert_ordered(native.model)
    assert native.log_likelihood == pytest.approx(generic.log_likelihood, abs=1e-3)
    for name, values in generic.model.parameters.items():
        np.testing.assert_allclose(native.model.parameters[name], values, atol=2e-3)


def test_sparse_categories_keep_probabilities_positive():
    rng = np.random.default_rng(0)
    n_persons, n_items = 1500, 12
    theta = rng.normal(size=n_persons)
    thresholds = np.sort(rng.normal(size=(n_items, 3)), axis=1)
    slopes = rng.uniform(1, 2, n_items)
    cumulative = 1 / (
        1 + np.exp(-slopes[None, :, None] * (theta[:, None, None] - thresholds[None]))
    )
    data = (rng.random((n_persons, n_items, 1)) < cumulative).sum(axis=2)
    native = mirt.fit_mirt(data, model="GRM", verbose=False)
    generic = mirt.fit_mirt(data, model="GRM", verbose=False, use_rust=False)
    _assert_ordered(native.model)
    assert native.log_likelihood == pytest.approx(generic.log_likelihood, abs=1e-3)
    assert _marginal_log_likelihood(native.model, data) == pytest.approx(
        _marginal_log_likelihood(generic.model, data), abs=1e-2
    )


@pytest.mark.parametrize("seed", range(4))
def test_native_fits_never_return_disordered_thresholds(seed):
    rng = np.random.default_rng(seed)
    n_persons, n_items, n_categories = 800, 10, 5
    # Nearly coincident middle thresholds leave sparse or empty categories.
    gaps = rng.uniform(0.0, 0.12, (n_items, n_categories - 2))
    gaps[:, 0] = rng.uniform(0.4, 1.0, n_items)
    start = rng.uniform(-1.5, 0.0, n_items)
    thresholds = start[:, None] + np.column_stack(
        [np.zeros(n_items), np.cumsum(gaps, axis=1)]
    )
    true = GradedResponseModel(n_items, n_categories=n_categories)
    true.set_parameters(
        discrimination=rng.uniform(0.8, 2.0, n_items), thresholds=thresholds
    )
    data = true.simulate(rng.standard_normal((n_persons, 1)), seed=seed)
    fit = mirt.fit_mirt(data, model="GRM", verbose=False)
    _assert_ordered(fit.model)
    # The ordered maximum likelihood fit beats the generating parameters. The
    # generic SLSQP fit of these tied thresholds sits within roundoff of its
    # ordering check, so it is no reliable reference here.
    assert _marginal_log_likelihood(fit.model, data) >= _marginal_log_likelihood(
        true, data
    )


def test_m_step_projects_a_disordered_start_onto_ordered_thresholds():
    rng = np.random.default_rng(5)
    true = GradedResponseModel(3, n_categories=4)
    true.set_parameters(
        discrimination=np.array([1.2, 1.5, 0.9]),
        thresholds=np.array([[-1.0, 0.0, 1.0], [-0.8, 0.1, 0.9], [-1.2, -0.3, 0.8]]),
    )
    theta = rng.standard_normal((600, 1))
    responses = true.simulate(theta, seed=5)
    # Item 1 loses its third category, so its optimum ties two thresholds.
    responses[responses[:, 1] == 2, 1] = 3
    points = np.linspace(-4.0, 4.0, 21)[:, None]
    posterior = np.exp(-2.0 * (points[:, 0] - theta) ** 2)
    posterior /= posterior.sum(axis=1, keepdims=True)
    starts = {
        "disordered": [[1.0, -1.0, 0.0], [0.5, 0.5, 0.5], [0.4, 0.2, -0.3]],
        "ordered": [[-1.0, -0.5, 0.0], [-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]],
    }
    fitted = {}
    for label, start in starts.items():
        model = true.copy()
        model.set_parameters(thresholds=np.array(start))
        assert try_polytomous_m_step(
            model,
            responses,
            posterior,
            points,
            max_iter=200,
            ftol=1e-12,
            epsilon=1e-10,
            n_jobs=1,
        )
        _assert_ordered(model)
        assert np.min(np.diff(model.parameters["thresholds"][1])) < 1e-5
        fitted[label] = model
    # The generic constrained optimizer agrees, including the tied item.
    reference = true.copy()
    reference.set_parameters(thresholds=np.array(starts["ordered"]))
    estimator = EMEstimator(item_optim_maxiter=200, item_optim_ftol=1e-12)
    for item in range(3):
        optimized = estimator._optimize_item_params(
            reference, item, responses, posterior, points
        )
        estimator._set_item_params(reference, item, optimized)
    for model in fitted.values():
        for name, values in reference.parameters.items():
            np.testing.assert_allclose(model.parameters[name], values, atol=1e-5)


def test_kernel_respects_fixed_thresholds():
    rng = np.random.default_rng(2)
    responses = rng.integers(0, 4, (200, 1)).astype(np.int32)
    posterior = rng.random((200, 9))
    posterior /= posterior.sum(axis=1, keepdims=True)
    points = np.linspace(-3.0, 3.0, 9)
    categories = np.array([4], dtype=np.int32)
    free = np.array([[True, True, False, True]])

    def fit(parameters):
        return mirt_rs.m_step_polytomous(
            responses,
            posterior,
            points,
            np.array([parameters]),
            free,
            categories,
            True,
            100,
            1e-10,
            1e-10,
            1,
            None,
        )[0]

    fitted = fit([1.0, 2.0, 1.5, -2.0])
    assert fitted[2] == 1.5
    assert np.all(np.diff(fitted[1:]) > 0)
    with pytest.raises(ValueError, match="fixed graded thresholds"):
        fit([1.0, 0.0, 6.0, 6.0])


def test_kernel_orders_items_with_unequal_categories_and_missing_data():
    rng = np.random.default_rng(3)
    theta = rng.standard_normal(400)
    responses = np.column_stack(
        [np.digitize(theta, [-0.5, 0.5]), np.digitize(theta, [-1.0, -0.2, 0.3, 1.1])]
    ).astype(np.int32)
    # Empty middle categories pull adjacent thresholds onto each other.
    responses[responses[:, 0] == 1, 0] = 2
    responses[responses[:, 1] == 2, 1] = 3
    responses[rng.random(responses.shape) < 0.1] = -1
    points = np.linspace(-4.0, 4.0, 15)
    posterior = np.exp(-2.0 * (points - theta[:, None]) ** 2)
    posterior /= posterior.sum(axis=1, keepdims=True)
    categories = np.array([3, 5], dtype=np.int32)
    # Disordered starts; the first item's padded storage is left alone.
    parameters = np.array([[1.0, 1.5, -1.5, 9.0, 9.0], [1.0, 2.0, 1.0, 0.0, -1.0]])
    free = np.array([[True, True, True, False, False], [True] * 5])
    fitted = [
        mirt_rs.m_step_polytomous(
            responses,
            posterior,
            points,
            parameters,
            free,
            categories,
            True,
            200,
            1e-12,
            1e-10,
            n_jobs,
            None,
        )
        for n_jobs in (1, 2)
    ]
    np.testing.assert_array_equal(fitted[0], fitted[1])
    np.testing.assert_array_equal(fitted[0][0, 3:], [9.0, 9.0])
    for item, k in enumerate(categories):
        gaps = np.diff(fitted[0][item, 1:k])
        assert np.all(gaps >= 1e-6 * (1 - 1e-9)), fitted[0]
        assert np.min(gaps) < 1e-5, fitted[0]
