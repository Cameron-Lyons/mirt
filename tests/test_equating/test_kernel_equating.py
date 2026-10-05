"""Tests for Gaussian kernel equating."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr

import mirt
from mirt.equating import (
    KernelEquatingResult,
    LinkingConstants,
    LinkingResult,
    ScoreEquatingResult,
    irt_kernel_equating,
    kernel_equating,
    lord_wingersky_recursion,
    score_equating_summary,
)
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _binomial(n, p):
    scores = np.arange(n + 1)
    log_mass = (
        gammaln(n + 1)
        - gammaln(scores + 1)
        - gammaln(n - scores + 1)
        + scores * np.log(p)
        + (n - scores) * np.log1p(-p)
    )
    return np.exp(log_mass)


def _moments(probabilities):
    scores = np.arange(len(probabilities))
    mean = probabilities @ scores
    return mean, probabilities @ (scores - mean) ** 2


def _continuized_cdf(probabilities, bandwidth):
    """Independent transcription of von Davier et al. (2004, eq. 4.8)."""
    scores = np.arange(len(probabilities))
    mean, variance = _moments(probabilities)
    shrink = np.sqrt(variance / (variance + bandwidth**2))

    def cdf(x):
        standardized = (x - shrink * scores - (1 - shrink) * mean) / (
            shrink * bandwidth
        )
        return float(probabilities @ ndtr(standardized))

    def density(x):
        standardized = (x - shrink * scores - (1 - shrink) * mean) / (
            shrink * bandwidth
        )
        kernel = np.exp(-0.5 * standardized**2) / np.sqrt(2 * np.pi)
        return float(probabilities @ kernel / (shrink * bandwidth))

    return cdf, density


@pytest.fixture
def forms():
    old = _binomial(20, 0.55)
    new = _binomial(25, 0.5)
    return old / old.sum(), new / new.sum()


def test_returns_score_equating_result_subclass(forms):
    result = kernel_equating(*forms)

    assert isinstance(result, KernelEquatingResult)
    assert isinstance(result, ScoreEquatingResult)
    assert result.method == "kernel"
    np.testing.assert_array_equal(result.old_scores, np.arange(21))
    assert result.theta.shape == (0,)
    assert result.standard_errors is None
    assert result.bandwidth_old > 0.0 and result.bandwidth_new > 0.0
    assert "kernel" in score_equating_summary(result)


@pytest.mark.parametrize("bandwidth", [0.3, 0.6, 2.0])
def test_continuization_preserves_mean_and_variance(forms, bandwidth):
    old, _ = forms
    _, density = _continuized_cdf(old, bandwidth)
    mean, variance = _moments(old)

    first = quad(lambda x: x * density(x), -30, 50, limit=400)[0]
    second = quad(lambda x: (x - mean) ** 2 * density(x), -30, 50, limit=400)[0]

    assert first == pytest.approx(mean, abs=1e-8)
    assert second == pytest.approx(variance, abs=1e-8)


@pytest.mark.parametrize("bandwidth", [(0.45, 0.8), 1.7])
def test_equated_scores_invert_independent_continuized_cdfs(forms, bandwidth):
    old, new = forms
    pair = bandwidth if isinstance(bandwidth, tuple) else (bandwidth, bandwidth)
    cdf_old, _ = _continuized_cdf(old, pair[0])
    cdf_new, _ = _continuized_cdf(new, pair[1])

    result = kernel_equating(old, new, bandwidth=bandwidth)

    expected = [
        brentq(lambda y, p=cdf_old(x): cdf_new(y) - p, -40, 70, xtol=1e-14)
        for x in range(len(old))
    ]
    np.testing.assert_allclose(result.new_scores, expected, atol=1e-9)
    assert (result.bandwidth_old, result.bandwidth_new) == pair


def test_equal_distributions_give_identity(forms):
    old, _ = forms

    for bandwidth in ("penalty", 0.4, "linear"):
        result = kernel_equating(old, old, bandwidth=bandwidth)
        np.testing.assert_allclose(result.new_scores, np.arange(21), atol=1e-9)


def test_large_bandwidth_is_linear_equating(forms):
    old, new = forms
    mean_old, variance_old = _moments(old)
    mean_new, variance_new = _moments(new)

    result = kernel_equating(old, new, bandwidth="linear")

    linear = mean_new + np.sqrt(variance_new / variance_old) * (
        np.arange(21) - mean_old
    )
    np.testing.assert_allclose(result.new_scores, linear, atol=1e-6)
    assert result.bandwidth_old == pytest.approx(1000 * np.sqrt(variance_old))


def test_equating_is_strictly_increasing_and_frequency_invariant(forms):
    old, new = forms
    frequencies = np.round(old * 5000) + 1.0

    result = kernel_equating(frequencies, new)
    normalized = kernel_equating(frequencies / frequencies.sum(), new)

    assert np.all(np.diff(result.new_scores) > 0.0)
    np.testing.assert_allclose(result.new_scores, normalized.new_scores, atol=1e-12)


def test_penalty_bandwidth_minimizes_density_misfit(forms):
    old, new = forms

    result = kernel_equating(old, new)

    def misfit(probabilities, bandwidth):
        _, density = _continuized_cdf(probabilities, bandwidth)
        fitted = np.array([density(x) for x in range(len(probabilities))])
        return np.sum((probabilities - fitted) ** 2)

    for probabilities, selected in (
        (old, result.bandwidth_old),
        (new, result.bandwidth_new),
    ):
        grid = np.geomspace(0.05, 30.0, 400)
        best = min(misfit(probabilities, h) for h in grid)
        assert misfit(probabilities, selected) <= best + 1e-15
        assert 0.4 < selected < 1.0


def test_second_penalty_avoids_u_shaped_densities():
    jagged = np.array([0.25, 0.05, 0.3, 0.05, 0.25, 0.05, 0.05])

    def u_shapes(bandwidth):
        _, density = _continuized_cdf(jagged / jagged.sum(), bandwidth)
        count = 0
        for x in range(len(jagged)):
            left = density(x - 0.25 + 1e-7) - density(x - 0.25 - 1e-7)
            right = density(x + 0.25 + 1e-7) - density(x + 0.25 - 1e-7)
            count += left < 0 and right >= 0
        return count

    misfit_only = kernel_equating(jagged, jagged)
    penalized = kernel_equating(jagged, jagged, kappa=1.0)

    assert u_shapes(misfit_only.bandwidth_old) == 2
    assert u_shapes(penalized.bandwidth_old) == 0


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_loglinear_presmoothing_preserves_moments(degree):
    rng = np.random.default_rng(degree)
    counts = rng.multinomial(400, _binomial(15, 0.6)).astype(float)
    new = _binomial(15, 0.5)

    result = kernel_equating(counts, new, presmoothing=(degree, None))

    scores = np.arange(16)
    observed = counts / counts.sum()
    for power in range(1, degree + 1):
        assert result.score_dist_old @ scores**power == pytest.approx(
            observed @ scores**power, rel=1e-10
        )
    if degree < 4:
        moment = degree + 1
        assert result.score_dist_old @ scores**moment != pytest.approx(
            observed @ scores**moment, rel=1e-6
        )
    np.testing.assert_allclose(result.score_dist_new, new / new.sum(), atol=1e-15)
    assert np.all(result.score_dist_old > 0.0)


def _directional_derivatives(equate, probabilities):
    """Differentiate along e_j - r, which keeps the distribution normalized."""
    columns = []
    for score, mass in enumerate(probabilities):
        direction = -probabilities.copy()
        direction[score] += 1.0
        step = 1e-3 * mass
        upper = equate(probabilities + step * direction)
        lower = equate(probabilities - step * direction)
        columns.append((upper - lower) / (2 * step))
    return np.column_stack(columns)


def _cubic_loglinear(n_scores, coefficients):
    """Return a distribution that a degree-3 log-linear model fits exactly."""
    u = np.linspace(-1.0, 1.0, n_scores)
    log_mass = np.polynomial.polynomial.polyval(u, [0.0, *coefficients])
    return np.exp(log_mass) / np.exp(log_mass).sum()


@pytest.mark.parametrize("presmoothing", [None, 3])
def test_standard_errors_match_numerical_delta_method(presmoothing):
    # The log-linear covariance assumes the fitted model holds, so the
    # population distributions lie exactly on a cubic log-linear model.
    old = _cubic_loglinear(11, [0.9, -2.4, 0.3])
    new = _cubic_loglinear(13, [-0.4, -1.8, -0.5])
    bandwidth = (0.6, 0.7)
    n_old, n_new = 900, 1200

    result = kernel_equating(
        old,
        new,
        bandwidth=bandwidth,
        presmoothing=presmoothing,
        n_old=n_old,
        n_new=n_new,
    )

    def equated(old_dist, new_dist):
        return kernel_equating(
            old_dist, new_dist, bandwidth=bandwidth, presmoothing=presmoothing
        ).new_scores

    # Multinomial covariance (D - r r^T) / N equals sum_j r_j d_j d_j^T / N
    # for the directions d_j = e_j - r used below.
    derivative_old = _directional_derivatives(lambda r: equated(r, new), old)
    derivative_new = _directional_derivatives(lambda s: equated(old, s), new)
    variance = (derivative_old**2 @ old) / n_old + (derivative_new**2 @ new) / n_new

    np.testing.assert_allclose(result.standard_errors, np.sqrt(variance), rtol=1e-5)


def test_linear_bandwidth_standard_errors_match_linear_equating_delta_method():
    rng = np.random.default_rng(3)
    old = rng.dirichlet(np.full(21, 3.0))
    new = rng.dirichlet(np.full(26, 3.0))
    n_old, n_new = 800, 1100

    result = kernel_equating(old, new, bandwidth="linear", n_old=n_old, n_new=n_new)

    # Delta method for mu_Y + (sigma_Y / sigma_X) (x - mu_X), differentiating
    # the moments with respect to unnormalized score probabilities.
    x, y = np.arange(21.0), np.arange(26.0)
    mean_old, variance_old = _moments(old)
    mean_new, variance_new = _moments(new)
    ratio = np.sqrt(variance_new / variance_old)
    deviation = x[:, None] - mean_old
    jacobian_old = -ratio * (x + deviation * (x - mean_old) ** 2 / (2 * variance_old))
    jacobian_new = y + deviation * (y - mean_new) ** 2 / (
        2 * np.sqrt(variance_old * variance_new)
    )
    covariance_old = (np.diag(old) - np.outer(old, old)) / n_old
    covariance_new = (np.diag(new) - np.outer(new, new)) / n_new
    variance = np.einsum(
        "ij,jk,ik->i", jacobian_old, covariance_old, jacobian_old
    ) + np.einsum("ij,jk,ik->i", jacobian_new, covariance_new, jacobian_new)

    np.testing.assert_allclose(result.standard_errors, np.sqrt(variance), rtol=1e-8)


def test_presmoothing_degree_must_be_below_observed_score_count():
    # Two observed scores cannot identify a cubic log-linear model.
    concentrated = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])

    with pytest.raises(ValueError, match="between 1 and 1, below its number"):
        kernel_equating(
            concentrated, np.ones(6), presmoothing=(3, None), n_old=10, n_new=10
        )


@pytest.mark.parametrize("degree", [6, 8, 10])
def test_high_degree_presmoothing_reaches_the_likelihood_maximum(degree):
    # A narrow observed range at the top of a long scale makes the fitted
    # tails tiny and the likelihood nearly flat along them.
    counts = np.zeros(61)
    counts[47:61] = [2, 9, 14, 21, 28, 55, 82, 76, 85, 66, 40, 16, 5, 1]
    observed = counts / counts.sum()

    result = kernel_equating(
        counts, _binomial(60, 0.8), bandwidth=0.6, presmoothing=(degree, None)
    )

    # The maximum likelihood fit reproduces the first `degree` moments, here
    # of the standardized score, whose tail powers expose unfitted tails.
    mean, variance = _moments(observed)
    standardized = (np.arange(61) - mean) / np.sqrt(variance)
    for power in range(1, degree + 1):
        assert result.score_dist_old @ standardized**power == pytest.approx(
            observed @ standardized**power, rel=1e-9, abs=1e-9
        )


def test_irt_kernel_equating_continuizes_lord_wingersky_distributions():
    old = ThreeParameterLogistic(6).set_parameters(
        discrimination=np.linspace(0.8, 1.6, 6),
        difficulty=np.linspace(-1.2, 1.4, 6),
        guessing=np.full(6, 0.15),
    )
    new = TwoParameterLogistic(8).set_parameters(
        discrimination=np.linspace(0.7, 1.5, 8),
        difficulty=np.linspace(-1.0, 1.6, 8),
    )
    theta = np.linspace(-3.0, 3.0, 25)
    weights = np.exp(-0.5 * theta**2)

    result = irt_kernel_equating(
        old, new, theta_distribution=weights, theta_grid=theta, items_new=[7, 1, 2]
    )

    old_dist = lord_wingersky_recursion(old, theta, weights)
    new_dist = lord_wingersky_recursion(new, theta, weights, items=[7, 1, 2])
    expected = kernel_equating(old_dist, new_dist)
    assert result.method == "irt_kernel"
    np.testing.assert_array_equal(result.theta, theta)
    np.testing.assert_allclose(result.new_scores, expected.new_scores, atol=1e-12)
    assert result.bandwidth_old == pytest.approx(expected.bandwidth_old)
    assert result.standard_errors is None


@pytest.mark.parametrize("model_type", [TwoParameterLogistic, GradedResponseModel])
@pytest.mark.parametrize(("bandwidth", "tolerance"), [(0.6, 1e-10), ("penalty", 1e-7)])
def test_irt_kernel_equating_is_invariant_to_linked_calibrations(
    model_type, bandwidth, tolerance
):
    A, B = 1.4, -0.7
    discrimination = np.array([0.7, 1.0, 1.3, 1.6])
    if model_type is GradedResponseModel:
        thresholds = np.array(
            [[-1.5, 0.5, 0.0], [-1.2, 0.0, 1.3], [-0.5, 1.5, 0.0], [-1.0, 0.4, 1.8]]
        )
        old = model_type(4, n_categories=[3, 4, 3, 4]).set_parameters(
            discrimination=discrimination, thresholds=thresholds
        )
        new = model_type(4, n_categories=[3, 4, 3, 4]).set_parameters(
            discrimination=A * discrimination, thresholds=(thresholds - B) / A
        )
    else:
        difficulty = np.array([-1.2, -0.3, 0.4, 1.5])
        old = model_type(4).set_parameters(
            discrimination=discrimination, difficulty=difficulty
        )
        new = model_type(4).set_parameters(
            discrimination=A * discrimination, difficulty=(difficulty - B) / A
        )
    linking = LinkingResult(LinkingConstants(A=A, B=B), anchor_items=[0, 1, 2, 3])

    result = irt_kernel_equating(old, new, linking_result=linking, bandwidth=bandwidth)

    # Selected bandwidths agree only to the optimizer's tolerance.
    np.testing.assert_allclose(result.new_scores, result.old_scores, atol=tolerance)


def test_kernel_equating_is_exported_at_top_level():
    assert mirt.kernel_equating is kernel_equating
    assert mirt.irt_kernel_equating is irt_kernel_equating
    assert mirt.KernelEquatingResult is KernelEquatingResult


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bandwidth": "wide"}, "bandwidth must be"),
        ({"bandwidth": 0.0}, "finite and positive"),
        ({"bandwidth": np.nan}, "finite and positive"),
        ({"bandwidth": (0.5, -1.0)}, "finite and positive"),
        ({"bandwidth": (0.5, 0.5, 0.5)}, "two entries"),
        ({"bandwidth": True}, "bandwidth must be"),
        ({"kappa": -1.0}, "kappa"),
        ({"presmoothing": 0}, "between 1 and"),
        ({"presmoothing": 20}, "between 1 and"),
        ({"presmoothing": 2.5}, "integers"),
        ({"n_old": 100}, "together"),
        ({"n_old": 0, "n_new": 10}, "at least 1"),
        ({"n_old": 10.5, "n_new": 10}, "integer"),
    ],
)
def test_invalid_options_are_rejected(forms, kwargs, message):
    with pytest.raises(ValueError, match=message):
        kernel_equating(*forms, **kwargs)


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        (np.array([0.0, 1.0, 0.0]), "two scores"),
        (np.array([0.5, -0.1, 0.6]), "non-negative"),
        (np.zeros(3), "positive finite mass"),
    ],
)
def test_invalid_distributions_are_rejected(distribution, message):
    valid = np.array([0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match=message):
        kernel_equating(distribution, valid)
    with pytest.raises(ValueError, match=message):
        kernel_equating(valid, distribution)
