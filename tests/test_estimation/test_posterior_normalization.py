"""Owned-buffer normalization, extreme shifts, and nonfinite EM rows."""

import numpy as np
import pytest

import mirt
from mirt.estimation import _posterior
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.utils.numeric import logsumexp

_BACKENDS = [
    "numpy",
    pytest.param(
        "rust",
        marks=pytest.mark.skipif(
            not mirt.is_rust_available(), reason="Rust extension unavailable"
        ),
    ),
]


@pytest.fixture(autouse=True, params=_BACKENDS)
def backend(request):
    """Run every test on the NumPy path and on the native kernel."""
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    yield request.param
    mirt.set_backend(previous)


@pytest.mark.parametrize("blocked", [False, True])
def test_normalization_matches_independent_likelihood_reference(monkeypatch, blocked):
    rng = np.random.default_rng(219)
    likelihood = rng.normal(-500, 200, (17, 21))[:, ::2]
    prior = rng.random(11)
    prior /= prior.sum()
    expected_log = logsumexp(likelihood + np.log(prior), axis=1)
    expected = np.exp(likelihood + np.log(prior) - expected_log[:, None])
    if blocked:
        monkeypatch.setattr(_posterior, "_MAX_NORMALIZATION_ELEMENTS", 13)
    prior.flags.writeable = False
    actual, actual_log = normalize_log_posterior(likelihood, np.log(prior))
    assert actual is likelihood
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=1e-14)
    np.testing.assert_allclose(actual_log, expected_log, rtol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_unrepresentable_normalizing_offset_still_gives_unit_posterior(offset):
    # At this scale, log(3) cannot be represented accurately when added to the
    # offset. Subtracting the rounded marginal no longer gives unit row sums.
    joint = np.full((4, 3), offset)
    posterior, marginal = normalize_log_posterior(joint, np.zeros(3))
    np.testing.assert_allclose(posterior, 1.0 / 3.0, atol=0.0)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(marginal, offset + np.log(3.0))


def test_nonfinite_log_marginal_semantics_are_preserved():
    joint = np.array([[-np.inf, -np.inf], [np.inf, 0.0], [np.nan, 0.0]])
    posterior, marginal = normalize_log_posterior(joint, np.zeros(2))
    np.testing.assert_array_equal(marginal, [-np.inf, np.inf, np.nan])
    assert np.isnan(posterior).all()


@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_common_likelihood_offset_preserves_nonuniform_prior(offset):
    prior = np.array([0.05, 0.15, 0.8])
    posterior, marginal = normalize_log_posterior(
        np.full((4, 3), offset), np.log(prior)
    )
    np.testing.assert_allclose(posterior, np.tile(prior, (4, 1)), atol=1e-15)
    np.testing.assert_array_equal(marginal, offset)


def test_zero_prior_mass_excludes_largest_likelihood_without_underflow():
    posterior, marginal = normalize_log_posterior(
        np.array([[0.0, -1000.0]]), np.array([-np.inf, 0.0])
    )
    np.testing.assert_array_equal(posterior, [[0.0, 1.0]])
    np.testing.assert_array_equal(marginal, [-1000.0])


def test_opposite_infinities_propagate_undefined_log_joint():
    posterior, marginal = normalize_log_posterior(
        np.array([[np.inf, 0.0]]), np.array([-np.inf, 0.0])
    )
    assert np.isnan(posterior).all()
    assert np.isnan(marginal).all()


@pytest.mark.parametrize("blocked", [False, True])
def test_no_prior_normalization_reuses_owned_storage(monkeypatch, blocked):
    values = np.random.default_rng(391).normal(-500, 100, (17, 11))
    expected_log = logsumexp(values, axis=1)
    expected = np.exp(values - expected_log[:, None])
    if blocked:
        monkeypatch.setattr(_posterior, "_MAX_NORMALIZATION_ELEMENTS", 13)
    posterior, marginal = normalize_log_posterior(values)
    assert posterior is values
    np.testing.assert_allclose(posterior, expected, rtol=8e-14, atol=1e-15)
    np.testing.assert_allclose(marginal, expected_log, atol=1e-13)


@pytest.mark.parametrize("kind", ["ordinary", "weighted"])
@pytest.mark.parametrize("offset", [-1e16, -1e100, 1e16, 1e100])
def test_estimator_e_steps_preserve_prior_with_large_custom_likelihood_offset(
    kind, offset
):
    model = TwoParameterLogistic(1)
    model.log_likelihood = lambda data, _theta: np.full(len(data), offset)
    model.log_likelihood_batch = lambda data, theta: np.full(
        (len(data), len(theta)), offset
    )
    responses = np.array([[0], [1]])
    mean, covariance = np.array([0.4]), np.array([[1.7]])
    density = GaussianDensity(mean=mean, cov=covariance)
    estimator = (
        EMEstimator(n_quadpts=7, use_gpu=False)
        if kind == "ordinary"
        else WeightedEMEstimator(n_quadpts=7)
    )
    estimator._quadrature = GaussHermiteQuadrature(n_points=7)
    if kind == "ordinary":
        estimator._latent_density = density
        posterior, marginal = estimator._e_step(model, responses)
    else:
        posterior, marginal = estimator._e_step_weighted(
            model, responses, mean, covariance, np.ones(2)
        )
    prior = np.exp(
        density.log_quadrature_mass(
            estimator._quadrature.nodes, estimator._quadrature.weights
        )
    )
    np.testing.assert_allclose(posterior, np.tile(prior, (2, 1)), atol=1e-15)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(marginal, offset)


@pytest.mark.parametrize("blocked", [False, True])
def test_rows_far_below_the_largest_prior_mass_are_normalized_exactly(
    monkeypatch, blocked
):
    # The joint is shifted by the largest prior mass. A likelihood that peaks
    # where the prior mass is e^-900 times smaller would underflow there, and
    # one at e^-300 would lose its smallest terms, so such rows are
    # normalized again from their own maxima.
    prior = np.array([0.0, -300.0, -900.0, -906.0, -np.inf])
    likelihood = np.array(
        [
            [-2000.0, -2000.0, 0.0, -1.0, 0.0],
            [-2000.0, 0.0, -2000.0, -800.0, -5.0],
            [-1.0, -0.5, 0.0, -3.0, -np.inf],
        ]
    )
    expected_log = logsumexp(likelihood + prior, axis=1)
    expected = np.exp(likelihood + prior - expected_log[:, None])
    if blocked:
        monkeypatch.setattr(_posterior, "_MAX_NORMALIZATION_ELEMENTS", 5)
    posterior, marginal = normalize_log_posterior(likelihood.copy(), prior)
    np.testing.assert_allclose(posterior, expected, rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(marginal, expected_log, rtol=1e-14)
    assert posterior[0, 3] > 0.0


def _edge_case_rows(rng):
    """Likelihood rows with large offsets, underflow and nonfinite entries."""
    rows = rng.normal(-500.0, 200.0, (40, 7))
    rows[0] = 1e300
    rows[1] = -np.inf
    rows[2, 3] = np.inf
    rows[3, 1] = np.nan
    rows[4] = [-2000.0, 0.0, -2000.0, -800.0, -5.0, -np.inf, -1e5]
    rows[5, ::2] = -np.inf
    rows[6] = [np.inf, 0.0, -np.inf, 1.0, 2.0, 3.0, 4.0]
    return rows


@pytest.mark.parametrize(
    "prior_kind", ["none", "uniform", "steep", "zero_mass", "all_zero", "nan"]
)
def test_native_and_numpy_normalization_agree_on_edge_cases(prior_kind, backend):
    if backend != "rust":
        pytest.skip("compares the native kernel against the NumPy path")
    rng = np.random.default_rng(23)
    rows = _edge_case_rows(rng)
    prior = {
        "none": None,
        "uniform": np.full(7, -np.log(7.0)),
        "steep": np.array([0.0, -300.0, -900.0, -906.0, -np.inf, -2.0, -1.0]),
        "zero_mass": np.array([-np.inf, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]),
        "all_zero": np.full(7, -np.inf),
        "nan": np.array([0.0, np.nan, -1.0, -2.0, -3.0, -4.0, -5.0]),
    }[prior_kind]
    with np.errstate(all="ignore"):
        native, native_log = normalize_log_posterior(rows.copy(), prior)
        mirt.set_backend("numpy")
        expected, expected_log = normalize_log_posterior(rows.copy(), prior)

    np.testing.assert_allclose(
        native, expected, rtol=1e-13, atol=1e-300, equal_nan=True
    )
    np.testing.assert_allclose(native_log, expected_log, rtol=1e-14, equal_nan=True)


def test_native_kernel_serves_only_owned_contiguous_buffers(monkeypatch, backend):
    calls = []

    def record(log_joint, prior):
        calls.append(log_joint)
        return np.zeros(log_joint.shape[0])

    monkeypatch.setattr(_posterior, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        _posterior,
        "mirt_rs",
        type("Native", (), {"normalize_log_posterior_inplace": staticmethod(record)}),
    )
    owned = np.zeros((3, 4))
    assert normalize_log_posterior(owned, np.zeros(4))[0] is owned
    assert calls == [owned]
    # Strided, single-precision and read-only buffers take the NumPy path.
    for buffer in (np.zeros((4, 3)).T, np.zeros((3, 4), dtype=np.float32)):
        normalize_log_posterior(buffer, np.zeros(4))
    readonly = np.zeros((3, 4))
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        normalize_log_posterior(readonly, np.zeros(4))
    assert len(calls) == 1
