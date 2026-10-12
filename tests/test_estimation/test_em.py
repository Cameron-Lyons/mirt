"""Tests for EM estimation algorithm."""

import warnings

import numpy as np
import pytest

import mirt
from mirt.estimation import em as em_module
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import (
    GaussHermiteQuadrature,
    RectangularQuadrature,
    build_quadrature,
    default_quadpts,
)
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


class TestEMEstimator:
    """Tests for EM algorithm."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = EMEstimator(n_quadpts=21, max_iter=100, tol=1e-3)

        assert estimator.n_quadpts == 21
        assert estimator.max_iter == 100
        assert estimator.tol == 1e-3

    def test_fit_basic(self, dichotomous_responses):
        """Test basic model fitting."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert result.log_likelihood < 0
        assert result.n_iterations > 0
        assert result.aic > 0
        assert result.bic > 0

    def test_convergence_history(self, dichotomous_responses):
        """Test convergence history is tracked."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=20)

        estimator.fit(model, responses)

        history = estimator.convergence_history
        assert len(history) > 0
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 0.1

    def test_parameter_recovery(self, rng):
        """Test recovery of known parameters."""
        n_persons, n_items = 500, 10

        true_a = np.ones(n_items) * 1.5
        true_b = np.linspace(-2, 2, n_items)

        theta = rng.standard_normal(n_persons)
        probs = 1 / (1 + np.exp(-true_a * (theta[:, None] - true_b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=21, max_iter=200, tol=1e-4)
        result = estimator.fit(model, responses)

        est_b = result.model.difficulty
        correlation = np.corrcoef(true_b, est_b)[0, 1]
        assert correlation > 0.8

    def test_standard_errors(self, dichotomous_responses):
        """Test standard errors are computed."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)

        assert "discrimination" in result.standard_errors
        assert "difficulty" in result.standard_errors

        se_disc = result.standard_errors["discrimination"]
        se_diff = result.standard_errors["difficulty"]
        assert np.all((se_disc > 0) | np.isnan(se_disc))
        assert np.all((se_diff > 0) | np.isnan(se_diff))

    def test_missing_data(self, dichotomous_responses, rng):
        """Test handling of missing data."""
        responses = dichotomous_responses["responses"].copy()
        n_items = dichotomous_responses["n_items"]

        mask = rng.random(responses.shape) < 0.1
        responses[mask] = -1

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)
        assert result.converged or result.n_iterations == 50


# Quadrature defaults


def test_default_quadrature_points_follow_r_mirt_by_dimension():
    assert [default_quadpts(f) for f in range(1, 7)] == [61, 31, 15, 9, 7, 5]
    with pytest.raises(MirtValidationError, match="MHRM"):
        default_quadpts(7)


def test_auto_rule_is_rectangular_for_fine_grids():
    grid = build_quadrature(61, 1)
    assert isinstance(grid, RectangularQuadrature)
    np.testing.assert_allclose(grid.nodes[:, 0], np.linspace(-6.0, 6.0, 61))
    density = np.exp(-0.5 * grid.nodes[:, 0] ** 2)
    np.testing.assert_allclose(grid.weights, density / density.sum())
    assert isinstance(build_quadrature(9, 4), GaussHermiteQuadrature)
    assert isinstance(build_quadrature(31, 1, "gauss-hermite"), GaussHermiteQuadrature)
    assert build_quadrature(15, 3).n_total_points == 15**3


def test_em_resolves_the_grid_by_the_number_of_factors(monkeypatch):
    class Stop(Exception):
        pass

    grids = []

    def record(n_points, n_dimensions, rule):
        grids.append((n_points, n_dimensions, rule))
        raise Stop

    monkeypatch.setattr(em_module, "build_quadrature", record)
    responses = np.zeros((10, 8), dtype=int)
    responses[::2] = 1
    for n_factors in (1, 2, 4):
        with pytest.raises(Stop), warnings.catch_warnings():
            warnings.simplefilter("error")
            EMEstimator().fit(TwoParameterLogistic(8, n_factors=n_factors), responses)
    assert grids == [(61, 1, "auto"), (31, 2, "auto"), (9, 4, "auto")]
    # The former 21 points per dimension give 194,481 nodes for four factors.
    with pytest.warns(RuntimeWarning, match="194,481 nodes.*MHRM"), pytest.raises(Stop):
        EMEstimator(n_quadpts=21).fit(
            TwoParameterLogistic(8, n_factors=4), np.tile(responses, (700, 1))
        )


def test_four_factor_default_fit_is_feasible():
    rng = np.random.default_rng(0)
    slopes = np.zeros((16, 4))
    for factor in range(4):
        slopes[4 * factor : 4 * factor + 4, factor] = rng.uniform(1.0, 2.0, 4)
    theta = rng.standard_normal((400, 4))
    logits = theta @ slopes.T + rng.uniform(-1.0, 1.0, 16)
    data = (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    result = mirt.fit_mirt(data, n_factors=4, max_iter=3, compute_standard_errors=False)
    assert np.isfinite(result.log_likelihood)


def test_default_grid_integrates_long_discriminating_tests():
    # 21 Gauss-Hermite nodes, the former default, shrink these slopes by 15%
    # on average and report a log-likelihood 54 units below the fine grid's.
    rng = np.random.default_rng(5)
    true = GradedResponseModel(25, n_categories=4).set_parameters(
        discrimination=rng.uniform(2.0, 3.5, 25),
        thresholds=np.sort(rng.normal(0.0, 1.0, (25, 3)), axis=1),
    )
    data = true.simulate(rng.standard_normal((800, 1)), seed=6)
    default = mirt.fit_mirt(data, "GRM", compute_standard_errors=False, tol=1e-6)
    fine = mirt.fit_mirt(
        data,
        "GRM",
        n_quadpts=201,
        quadrature="rectangular",
        compute_standard_errors=False,
        tol=1e-6,
    )
    assert default.log_likelihood == pytest.approx(fine.log_likelihood, abs=0.05)
    np.testing.assert_allclose(
        default.model.discrimination, fine.model.discrimination, atol=0.01
    )


# Convergence controls


@pytest.mark.parametrize("model", ["2PL", "GRM"])
@pytest.mark.parametrize(
    "options",
    [
        {"n_quadpts": 4},
        {"n_quadpts": 0},
        {"n_quadpts": 21.0},
        {"n_quadpts": True},
        {"tol": float("nan")},
        {"tol": float("inf")},
        {"tol": 0.0},
        {"max_iter": 2.5},
        {"max_iter": 0},
        {"quadrature": "simpson"},
    ],
)
@pytest.mark.parametrize("use_rust", [True, False])
def test_fit_mirt_validates_controls_on_every_path(model, options, use_rust):
    data = mirt.simdata("2PL", n_persons=50, n_items=4, seed=1)
    with pytest.raises(MirtValidationError):
        mirt.fit_mirt(data, model=model, use_rust=use_rust, **options)


# Parameter expansion (PX-EM)


def _px_cases():
    from mirt.models.dichotomous import OneParameterLogistic
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        NominalResponseModel,
        PartialCreditModel,
    )

    return {
        "1PL": (
            lambda: OneParameterLogistic(10, estimate_discrimination=True),
            dict(model="2PL", n_items=10),
        ),
        "2PL": (lambda: TwoParameterLogistic(10), dict(model="2PL", n_items=10)),
        "MIRT": (
            lambda: MultidimensionalModel(
                10, 1, model_type="confirmatory", loading_pattern=np.ones((10, 1))
            ),
            dict(model="2PL", n_items=10),
        ),
        "GRM": (
            lambda: GradedResponseModel(8, n_categories=4),
            dict(model="GRM", n_items=8, n_categories=4),
        ),
        "GPCM": (
            lambda: GeneralizedPartialCredit(8, n_categories=4),
            dict(model="GPCM", n_items=8, n_categories=4),
        ),
        "PCM": (
            lambda: PartialCreditModel(8, n_categories=4, estimate_discrimination=True),
            dict(model="GPCM", n_items=8, n_categories=4),
        ),
        "NRM": (
            lambda: NominalResponseModel(8, n_categories=3),
            dict(model="GRM", n_items=8, n_categories=3),
        ),
    }


def _px_fit(factory, responses, **options):
    model = factory()
    options = {
        "max_iter": 5000,
        "compute_standard_errors": False,
        "use_gpu": False,
        **options,
    }
    estimator = EMEstimator(**options)
    return model, estimator.fit(model, responses), estimator


@pytest.mark.parametrize("kind", list(_px_cases()))
def test_px_em_reaches_the_plain_em_optimum_in_fewer_iterations(kind):
    factory, simulation = _px_cases()[kind]
    responses = mirt.simdata(n_persons=1000, seed=7, **simulation)
    reference, reference_fit, _ = _px_fit(factory, responses, px_em=False, tol=1e-9)
    _, plain_fit, _ = _px_fit(factory, responses, px_em=False, tol=1e-5)
    model, fit, estimator = _px_fit(factory, responses, tol=1e-5)

    assert fit.converged
    assert fit.n_iterations < plain_fit.n_iterations
    assert fit.log_likelihood == pytest.approx(reference_fit.log_likelihood, abs=1e-4)
    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-3)
    history = np.asarray(estimator.convergence_history)
    assert history[-1] == fit.log_likelihood
    assert np.all(np.diff(history) >= -1e-8)
    if kind in ("1PL", "PCM"):
        assert np.unique(model.parameters["discrimination"]).size == 1


def test_px_em_standard_errors_match_plain_em():
    responses = mirt.simdata("GRM", n_persons=1000, n_items=8, n_categories=4, seed=7)
    _, plain_fit, _ = _px_fit(
        _px_cases()["GRM"][0],
        responses,
        px_em=False,
        tol=1e-8,
        compute_standard_errors=True,
    )
    _, fit, _ = _px_fit(
        _px_cases()["GRM"][0], responses, tol=1e-8, compute_standard_errors=True
    )
    for name, errors in plain_fit.standard_errors.items():
        np.testing.assert_allclose(fit.standard_errors[name], errors, rtol=1e-3)


def _px_skipped_cases():
    from mirt.estimation.priors import LogNormalPrior
    from mirt.models.dichotomous import OneParameterLogistic, ThreeParameterLogistic

    def masked():
        model = TwoParameterLogistic(8)
        fixed = np.ones(8, dtype=bool)
        fixed[0] = False
        return model.set_free_parameter_masks({"difficulty": fixed})

    two = lambda: TwoParameterLogistic(8)  # noqa: E731
    return {
        "gauss-hermite": (two, dict(n_quadpts=21, quadrature="gauss-hermite")),
        "coarse": (two, dict(n_quadpts=21)),
        "priors": (two, dict(item_priors={"discrimination": LogNormalPrior(0, 0.5)})),
        "constraints": (two, dict(constraints=[("discrimination", [0, 1])])),
        "masks": (masked, {}),
        "density": (two, dict(latent_density="empirical")),
        "3PL": (lambda: ThreeParameterLogistic(8), {}),
        "unit-slope": (lambda: OneParameterLogistic(8), {}),
        "disabled": (two, dict(px_em=False)),
    }


@pytest.mark.parametrize("kind", list(_px_skipped_cases()))
def test_px_em_leaves_other_fits_to_plain_em(kind, monkeypatch):
    factory, options = _px_skipped_cases()[kind]
    responses = mirt.simdata("2PL", n_persons=400, n_items=8, seed=3)
    _, plain_fit, plain = _px_fit(factory, responses, **{**options, "px_em": False})

    def refuse(*args):
        raise AssertionError("no expansion step expected")

    monkeypatch.setattr(EMEstimator, "_expand_latent_scale", refuse)
    _, fit, estimator = _px_fit(factory, responses, **options)

    assert fit.n_iterations == plain_fit.n_iterations
    np.testing.assert_allclose(
        estimator.convergence_history, plain.convergence_history, rtol=1e-12
    )


def test_px_em_skips_nonstandard_gaussian_populations(monkeypatch):
    responses = mirt.simdata("2PL", n_persons=400, n_items=8, seed=3)

    def refuse(*args):
        raise AssertionError("no expansion step expected")

    monkeypatch.setattr(EMEstimator, "_expand_latent_scale", refuse)
    result = EMEstimator(compute_standard_errors=False, use_gpu=False).fit(
        TwoParameterLogistic(8), responses, prior_mean=np.array([0.5])
    )
    assert result.converged


def test_rejected_expansion_steps_are_undone(monkeypatch):
    responses = mirt.simdata("2PL", n_persons=1000, n_items=10, seed=7)
    _, plain_fit, plain = _px_fit(
        lambda: TwoParameterLogistic(10), responses, px_em=False
    )
    calls = []

    def harmful(model, layout, nodes, node_counts):
        calls.append(layout)
        params = model.parameters
        return {
            "discrimination": params["discrimination"] * 8.0,
            "difficulty": params["difficulty"] + 2.0,
        }

    monkeypatch.setattr(em_module, "_latent_scale_updates", harmful)
    model, fit, estimator = _px_fit(lambda: TwoParameterLogistic(10), responses)

    # The rejected step is undone and plain EM takes over from its own
    # M-step point, so the fit follows plain EM exactly.
    assert len(calls) == 1
    assert fit.n_iterations == plain_fit.n_iterations
    np.testing.assert_allclose(
        estimator.convergence_history, plain.convergence_history, rtol=1e-12
    )


def _px_fold_cases():
    from mirt.models.dichotomous import OneParameterLogistic
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        NominalResponseModel,
        PartialCreditModel,
    )

    rng = np.random.default_rng(31)
    pattern = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    def uniform(low, high, shape):
        return rng.uniform(low, high, shape)

    return {
        "1PL": OneParameterLogistic(4, estimate_discrimination=True).set_parameters(
            discrimination=np.full(4, 1.3), difficulty=uniform(-2, 2, 4)
        ),
        "2PL": TwoParameterLogistic(4).set_parameters(
            discrimination=uniform(0.5, 2, 4), difficulty=uniform(-2, 2, 4)
        ),
        "GRM": GradedResponseModel(4, n_categories=[3, 4, 4, 2]).set_parameters(
            discrimination=uniform(0.5, 2, 4),
            thresholds=np.sort(uniform(-2, 2, (4, 3)), axis=1),
        ),
        "GPCM": GeneralizedPartialCredit(4, n_categories=4).set_parameters(
            discrimination=uniform(0.5, 2, 4), steps=uniform(-2, 2, (4, 3))
        ),
        "PCM": PartialCreditModel(
            4, n_categories=[3, 4, 4, 4], estimate_discrimination=True
        ).set_parameters(discrimination=np.full(4, 0.8), steps=uniform(-2, 2, (4, 3))),
        "NRM": NominalResponseModel(4, n_categories=[3, 4, 4, 4]).set_parameters(
            slopes=uniform(-1, 2, (4, 4)), intercepts=uniform(-1, 1, (4, 4))
        ),
        "2PL-2D": TwoParameterLogistic(4, n_factors=2).set_parameters(
            discrimination=uniform(0.5, 2, (4, 2)), difficulty=uniform(-1, 1, 4)
        ),
        "GPCM-2D": GeneralizedPartialCredit(
            4, n_categories=3, n_factors=2
        ).set_parameters(
            discrimination=uniform(0.5, 2, (4, 2)), steps=uniform(-1, 1, (4, 2))
        ),
        "NRM-2D": NominalResponseModel(4, n_categories=3, n_factors=2).set_parameters(
            slopes=uniform(-1, 2, (4, 3, 2)), intercepts=uniform(-1, 1, (4, 3))
        ),
        "MIRT-2D": MultidimensionalModel(4, 2).set_parameters(
            slopes=uniform(-1, 2, (4, 2)), intercepts=uniform(-1, 1, 4)
        ),
        "MIRT-pattern": MultidimensionalModel(
            4, 2, model_type="confirmatory", loading_pattern=pattern
        ).set_parameters(
            slopes=uniform(0.5, 2, (4, 2)) * pattern, intercepts=uniform(-1, 1, 4)
        ),
    }


@pytest.mark.parametrize("kind", list(_px_fold_cases()))
def test_expansion_step_keeps_every_item_curve(kind):
    model = _px_fold_cases()[kind]
    n_factors = model.n_factors
    grid = RectangularQuadrature(31 if n_factors > 1 else 61, n_factors)
    nodes = grid.nodes
    # Expected counts of a correlated population with mean m and covariance C.
    mean = np.array([0.3, -0.2])[:n_factors]
    covariance = np.array([[1.4, 0.3], [0.3, 0.8]])[:n_factors, :n_factors]
    centered = nodes - mean
    counts = 500.0 * np.exp(
        -0.5 * np.sum(centered @ np.linalg.inv(covariance) * centered, axis=1)
    )
    layout = em_module._latent_scale_layout(model)
    updates = em_module._latent_scale_updates(model, layout, nodes, counts)
    assert updates is not None

    total = counts.sum()
    moments = counts @ nodes / total
    spread = nodes - moments
    factor = np.linalg.cholesky((spread * counts[:, None]).T @ spread / total)
    if kind == "MIRT-pattern":
        # Loading patterns keep their zeros under a diagonal rescaling.
        factor = np.diag(np.sqrt(np.diag(factor @ factor.T)))
        assert np.all(updates["slopes"][model.loading_pattern == 0] == 0.0)
    rescaled = model.copy()
    rescaled.set_parameters(**updates)
    theta = np.random.default_rng(2).normal(size=(50, n_factors))
    np.testing.assert_allclose(
        rescaled.probability(theta),
        model.probability(moments + theta @ factor.T),
        rtol=1e-10,
        atol=1e-12,
    )
    if kind in ("1PL", "PCM"):
        assert np.unique(updates["discrimination"]).size == 1


def test_expansion_step_is_skipped_outside_the_parameter_box():
    from mirt.models.polytomous import GeneralizedPartialCredit

    model = GeneralizedPartialCredit(3, n_categories=3).set_parameters(
        discrimination=np.ones(3), steps=np.array([[-1.0, 5.5], [0.0, 1.0], [-1, 1]])
    )
    nodes = RectangularQuadrature(61).nodes
    # Mean -0.5 and SD 0.9 move the step at 5.5 to 6.7, beyond the box of 6.
    counts = np.exp(-0.5 * ((nodes[:, 0] + 0.5) / 0.9) ** 2)
    layout = em_module._latent_scale_layout(model)
    assert em_module._latent_scale_updates(model, layout, nodes, counts) is None
    model.set_parameters(steps=np.array([[-1.0, 4.5], [0.0, 1.0], [-1, 1]]))
    updates = em_module._latent_scale_updates(model, layout, nodes, counts)
    assert updates is not None
    assert np.all(np.abs(updates["steps"]) <= 6.0)


@pytest.mark.parametrize("value", ["yes", None, 1])
def test_px_em_is_validated(value):
    with pytest.raises(MirtValidationError, match="px_em"):
        EMEstimator(px_em=value)
