"""Preconditioned Robbins-Monro updates of the NumPy MHRM estimator."""

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from mirt import MHRMEstimator, TwoParameterLogistic, fit_mirt, simdata
from mirt.estimation.mcmc import _precondition, _robbins_monro_step
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import OneParameterLogistic, ThreeParameterLogistic
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)


class _CountingTwoPL(TwoParameterLogistic):
    """A custom curve hook, which forces the numerically differentiated path."""

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self.probability_calls = 0

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        return super().probability(theta, item_idx)


def _reference_step(a, b, theta, responses, gain, previous):
    """Cai (2010) update with the analytic complete-data score and Hessian."""
    observed = responses >= 0
    theta, y = theta[observed, 0], responses[observed].astype(float)
    centered = theta - b
    p = expit(a * centered)
    residual, weight = p - y, p * (1 - p)
    gradient = np.array([residual @ centered, -a * residual.sum()])
    cross = -(a * (weight @ centered) + residual.sum())
    hessian = np.array([[weight @ centered**2, cross], [cross, a * a * weight.sum()]])
    information = (
        hessian if previous is None else previous + gain * (hessian - previous)
    )
    return np.array([a, b]) - gain * np.linalg.solve(information, gradient), information


def _data(n_persons=300, seed=8):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, (n_persons, 1))
    probability = expit(np.array([1.3, 0.9, 1.6]) * (theta - [-0.4, 0.3, 0.8]))
    responses = (rng.random(probability.shape) < probability).astype(np.int64)
    responses[rng.random(responses.shape) < 0.1] = -1
    responses = np.column_stack((responses, np.full(n_persons, -1)))
    return theta, responses


@pytest.mark.parametrize("model_class", [TwoParameterLogistic, _CountingTwoPL])
@pytest.mark.parametrize("gain", [1.0, 0.15])
def test_update_matches_the_preconditioned_robbins_monro_reference(model_class, gain):
    theta, responses = _data()
    initial = {
        "discrimination": np.array([1.1, 1.0, 1.2, 0.8]),
        "difficulty": np.array([-0.2, 0.1, 0.6, 0.4]),
    }
    model = model_class(4).set_parameters(**initial)
    previous = np.array([[90.0, -10.0], [-10.0, 70.0]])
    information = [previous.copy(), None, previous.copy(), None]

    MHRMEstimator(use_rust=False)._update_parameters(
        model, responses, theta, gain, information
    )

    for item in range(3):
        start = previous if item != 1 else None
        expected, expected_information = _reference_step(
            initial["discrimination"][item],
            initial["difficulty"][item],
            theta,
            responses[:, item],
            gain,
            start,
        )
        fitted = [
            model.parameters["discrimination"][item],
            model.parameters["difficulty"][item],
        ]
        np.testing.assert_allclose(fitted, expected, rtol=1e-4)
        np.testing.assert_allclose(information[item], expected_information, rtol=1e-4)
    # An item without responses keeps its parameters and has no information.
    assert model.parameters["discrimination"][3] == initial["discrimination"][3]
    assert model.parameters["difficulty"][3] == initial["difficulty"][3]
    assert information[3] is None
    if model_class is _CountingTwoPL:
        # The numerical path restores the model after each trial evaluation.
        assert model.probability_calls > 0


def test_fixed_slopes_never_move():
    theta, responses = _data()
    model = OneParameterLogistic(4)
    information = [None] * 4
    estimator = MHRMEstimator(use_rust=False)
    for gain in (1.0, 0.5, 0.25):
        estimator._update_parameters(model, responses, theta, gain, information)
    np.testing.assert_array_equal(model.parameters["discrimination"], 1.0)
    assert np.all(model.parameters["difficulty"][:3] != 0.0)
    assert all(block.shape == (1, 1) for block in information[:3])


def test_graded_thresholds_stay_ordered():
    rng = np.random.default_rng(4)
    theta = rng.normal(0.0, 1.0, (200, 1))
    # With an empty middle category, crossing thresholds would raise the
    # clipped complete-data likelihood without bound.
    responses = np.where(theta[:, 0] > 0.2, 2, 0)[:, None]
    model = GradedResponseModel(1, 3)
    information = [None]
    estimator = MHRMEstimator(use_rust=False)
    for _ in range(20):
        estimator._update_parameters(model, responses, theta, 1.0, information)
        thresholds = model.parameters["thresholds"][0]
        assert thresholds[1] - thresholds[0] >= 1e-6 - 1e-12


def test_step_holds_coordinates_a_bound_would_cut():
    information = np.array([[4.0, 1.0], [1.0, 2.0]])
    lower, upper = np.array([0.1, -6.0]), np.array([5.0, 6.0])
    gradient = np.array([3.0, -1.0])

    # Descent would lower the first coordinate below its bound: hold it and
    # solve the remaining one-coordinate block.
    step = _robbins_monro_step(
        information, gradient, np.array([0.1, 0.0]), lower, upper, 0.5
    )
    np.testing.assert_allclose(step, [0.0, 0.5 * 1.0 / 2.0])

    # Inside the box the full block is solved and long steps are shortened.
    step = _robbins_monro_step(
        information, 50 * gradient, np.array([1.0, 0.0]), lower, upper, 1.0
    )
    expected = -np.linalg.solve(information, 50 * gradient)
    np.testing.assert_allclose(step, expected / np.max(np.abs(expected)))


def test_precondition_uses_curvature_magnitudes():
    positive = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([1.0, -2.0])
    np.testing.assert_allclose(
        _precondition(positive, rhs), np.linalg.solve(positive, rhs)
    )

    indefinite = np.diag([2.0, -4.0])
    np.testing.assert_allclose(_precondition(indefinite, rhs), [0.5, -0.5])
    assert np.all(np.isfinite(_precondition(np.zeros((2, 2)), rhs)))


def test_python_fit_persists_parameter_updates() -> None:
    responses = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    model = TwoParameterLogistic(3)
    initial = {name: values.copy() for name, values in model.parameters.items()}

    result = MHRMEstimator(
        n_cycles=6,
        burnin=2,
        use_rust=False,
        seed=23,
    ).fit(model, responses)

    assert any(
        not np.array_equal(initial[name], values)
        for name, values in result.model.parameters.items()
    )


N_ITEMS = 6
_A = np.random.default_rng(12).uniform(1.0, 2.0, N_ITEMS)
_B = np.linspace(-1.2, 1.2, N_ITEMS)
FAMILIES = {
    "1PL": (
        lambda: simdata("1PL", n_persons=800, n_items=N_ITEMS, difficulty=_B, seed=3),
        lambda: OneParameterLogistic(N_ITEMS),
    ),
    "3PL": (
        lambda: simdata(
            "3PL",
            n_persons=1500,
            n_items=N_ITEMS,
            discrimination=_A,
            difficulty=_B,
            guessing=np.full(N_ITEMS, 0.15),
            seed=3,
        ),
        lambda: ThreeParameterLogistic(N_ITEMS),
    ),
    "GRM": (
        lambda: simdata(
            "GRM",
            n_persons=800,
            n_items=N_ITEMS,
            n_categories=3,
            discrimination=_A,
            thresholds=np.column_stack([_B - 0.7, _B + 0.7]),
            seed=3,
        ),
        lambda: GradedResponseModel(N_ITEMS, 3),
    ),
    "GPCM": (
        lambda: simdata(
            "GPCM",
            n_persons=800,
            n_items=N_ITEMS,
            n_categories=3,
            discrimination=_A,
            seed=3,
        ),
        lambda: GeneralizedPartialCredit(N_ITEMS, 3),
    ),
    "NRM": (
        lambda: simdata(
            "NRM",
            n_persons=800,
            n_items=N_ITEMS,
            n_categories=3,
            slopes=np.column_stack([np.zeros(N_ITEMS), _A / 2, _A]),
            intercepts=np.column_stack([np.zeros(N_ITEMS), np.full(N_ITEMS, 0.3), -_B]),
            seed=3,
        ),
        lambda: NominalResponseModel(N_ITEMS, 3),
    ),
}


def _marginal_log_likelihood(model, responses):
    quadrature = GaussHermiteQuadrature(41, model.n_factors)
    joint = model.log_likelihood_batch(responses, quadrature.nodes)
    return float(logsumexp(joint + np.log(quadrature.weights), axis=1).sum())


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_reaches_the_em_likelihood(family):
    # Previously guessing stayed at its start and polytomous items raised.
    simulate, build = FAMILIES[family]
    responses = simulate()
    em = fit_mirt(responses, family, compute_standard_errors=False).model
    mhrm = (
        MHRMEstimator(n_cycles=250, burnin=60, seed=0, use_rust=False)
        .fit(build(), responses)
        .model
    )

    # Monte Carlo error should cost well under half a log-likelihood unit per
    # parameter; weakly identified 3PL estimates can differ more than curves.
    gap = _marginal_log_likelihood(em, responses) - _marginal_log_likelihood(
        mhrm, responses
    )
    assert gap < 0.5 * em.n_parameters
