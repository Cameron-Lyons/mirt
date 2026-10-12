"""Sampler controls and native/NumPy agreement for MHRM and Gibbs estimation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import mirt
from mirt import GibbsSampler, MHRMEstimator, TwoParameterLogistic, simdata
from mirt.exceptions import MirtValidationError

native = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native backend unavailable"
)


@pytest.fixture(autouse=True)
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n_iter": 0}, "n_iter"),
        ({"n_iter": 10.0}, "n_iter"),
        ({"burnin": -1}, "burnin"),
        ({"burnin": True}, "burnin"),
        ({"thin": 0}, "thin"),
        ({"n_iter": 20, "burnin": 20}, "burnin"),
        ({"n_iter": 20, "burnin": 30}, "burnin"),
    ],
)
def test_gibbs_sampler_rejects_invalid_schedules(
    kwargs: dict[str, Any], parameter: str
) -> None:
    with pytest.raises(MirtValidationError, match=parameter) as info:
        GibbsSampler(**kwargs)
    assert info.value.context["parameter"] == parameter


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n_cycles": 0}, "n_cycles"),
        ({"n_cycles": 2.5}, "n_cycles"),
        ({"burnin": -1}, "burnin"),
        ({"proposal_sd": 0.0}, "proposal_sd"),
        ({"proposal_sd": np.nan}, "proposal_sd"),
        ({"proposal_sd": "0.5"}, "proposal_sd"),
        ({"gain_sequence": "fast"}, "gain_sequence"),
        ({"n_chains": 0}, "n_chains"),
        ({"n_chains": 1.5}, "n_chains"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_mhrm_estimator_rejects_invalid_controls(
    kwargs: dict[str, Any], parameter: str
) -> None:
    with pytest.raises(MirtValidationError, match=parameter) as info:
        MHRMEstimator(**kwargs)
    assert info.value.context["parameter"] == parameter


def test_mhrm_estimator_allows_burnin_beyond_cycles() -> None:
    estimator = MHRMEstimator(n_cycles=10, burnin=500)

    assert estimator.burnin == 500


@native
@pytest.mark.parametrize(("n_iter", "burnin", "thin"), [(60, 20, 3), (10, 0, 3)])
def test_gibbs_backends_keep_the_same_number_of_draws(
    n_iter: int, burnin: int, thin: int
) -> None:
    responses = simdata(n_persons=80, n_items=4, seed=0)
    lengths = []
    for use_rust in (True, False):
        result = GibbsSampler(
            n_iter=n_iter, burnin=burnin, thin=thin, seed=1, use_rust=use_rust
        ).fit(TwoParameterLogistic(4), responses)
        lengths.append({name: chain.shape[0] for name, chain in result.chains.items()})

    expected = -(-(n_iter - burnin) // thin)
    assert lengths[0] == lengths[1]
    assert set(lengths[0].values()) == {expected}


@native
def test_default_native_mhrm_fit_moves_item_parameters() -> None:
    responses = simdata(n_persons=500, n_items=10, seed=1)

    result = mirt.fit_mirt(responses, "2PL", estimation="MHRM", use_rust=True)

    parameters = result.model.parameters
    assert np.std(parameters["difficulty"]) > 0.3
    assert not np.allclose(parameters["discrimination"], 1.0)


def _recovery_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(101)
    difficulty = np.linspace(-1.5, 1.5, 10)
    responses = simdata(
        n_persons=1000,
        n_items=10,
        discrimination=rng.uniform(0.8, 2.0, 10),
        difficulty=difficulty,
        seed=1,
    )
    return responses, difficulty


@native
@pytest.mark.parametrize("gain_sequence", ["standard", "adaptive"])
def test_native_mhrm_tracks_numpy_mhrm_and_em(gain_sequence: str) -> None:
    responses, true_difficulty = _recovery_data()
    results = {}
    estimates = {}
    for use_rust in (True, False):
        result = MHRMEstimator(
            n_cycles=400,
            burnin=100,
            gain_sequence=gain_sequence,
            seed=11,
            use_rust=use_rust,
        ).fit(TwoParameterLogistic(10), responses)
        results[use_rust] = result
        estimates[use_rust] = result.model.parameters
    em = mirt.fit_mirt(responses, "2PL").model.parameters

    # The kernels run the same algorithm on different random streams.
    native, numpy = estimates[True], estimates[False]
    for name, tolerance in (("difficulty", 0.1), ("discrimination", 0.2)):
        assert np.max(np.abs(native[name] - numpy[name])) < tolerance
        assert np.max(np.abs(native[name] - em[name])) < tolerance
    assert np.corrcoef(native["difficulty"], true_difficulty)[0, 1] > 0.98
    # Both backends score the fit at MAP abilities, so information criteria
    # agree; the final ability draw would sit several percent lower.
    assert results[True].log_likelihood == pytest.approx(
        results[False].log_likelihood, rel=0.01
    )
    assert results[True].aic == pytest.approx(results[False].aic, rel=0.01)


def _em_and_mhrm(use_rust: bool, seed: int):
    responses, _ = _recovery_data()
    em = mirt.fit_mirt(responses, "2PL").model.parameters
    # fit_mirt(estimation="MHRM") runs 500 cycles with 125 burn-in cycles.
    mhrm = MHRMEstimator(n_cycles=500, burnin=125, seed=seed, use_rust=use_rust)
    return em, mhrm.fit(TwoParameterLogistic(10), responses).model.parameters


@pytest.mark.parametrize(
    "use_rust", [pytest.param(True, marks=native), pytest.param(False)]
)
def test_default_mhrm_schedule_matches_em_without_shrinkage(use_rust: bool) -> None:
    em, mhrm = _em_and_mhrm(use_rust, seed=3)

    # Unpreconditioned 1/(cycle+1) steps left difficulties ~0.6x of EM's range.
    assert np.ptp(mhrm["difficulty"]) / np.ptp(em["difficulty"]) == pytest.approx(
        1.0, abs=0.05
    )
    assert np.max(np.abs(mhrm["difficulty"] - em["difficulty"])) < 0.1
    assert np.max(np.abs(mhrm["discrimination"] - em["discrimination"])) < 0.25


@native
def test_default_fit_mirt_mhrm_is_not_shrunk(monkeypatch) -> None:
    responses, _ = _recovery_data()
    em = mirt.fit_mirt(responses, "2PL").model.parameters["difficulty"]
    schedules = []
    fit = MHRMEstimator.fit

    def seeded_fit(self, model, data, **kwargs):
        # fit_mirt draws a fresh seed; pin it so the test is deterministic.
        schedules.append((self.n_cycles, self.burnin))
        self.seed = 7
        return fit(self, model, data, **kwargs)

    monkeypatch.setattr(MHRMEstimator, "fit", seeded_fit)
    mhrm = mirt.fit_mirt(responses, "2PL", estimation="MHRM").model.parameters
    assert schedules == [(500, 125)]
    assert np.ptp(mhrm["difficulty"]) / np.ptp(em) > 0.9
    assert np.max(np.abs(mhrm["difficulty"] - em)) < 0.15


def _grid_log_likelihood(model, responses) -> float:
    """Marginal log-likelihood on EM's default grid."""
    from scipy.special import logsumexp

    from mirt.estimation.quadrature import build_quadrature, default_quadpts

    quadrature = build_quadrature(default_quadpts(model.n_factors), model.n_factors)
    joint = model.log_likelihood_batch(responses, quadrature.nodes)
    return float(logsumexp(joint + np.log(quadrature.weights), axis=1).sum())


@pytest.mark.parametrize(
    "use_rust", [pytest.param(True, marks=native), pytest.param(False)]
)
def test_mhrm_stops_once_changes_stay_below_tol(use_rust: bool) -> None:
    responses, _ = _recovery_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = MHRMEstimator(
            n_cycles=5000, burnin=100, seed=5, use_rust=use_rust
        ).fit(TwoParameterLogistic(10), responses)

    # Every fit used to run all cycles and report converged=True.
    assert result.converged is True
    assert 100 < result.n_iterations < 5000
    em = mirt.fit_mirt(responses, "2PL", compute_standard_errors=False)
    for name in ("discrimination", "difficulty"):
        assert (
            np.max(np.abs(result.model.parameters[name] - em.model.parameters[name]))
            < 0.15
        )


def test_mhrm_reports_cycles_that_run_out() -> None:
    responses, _ = _recovery_data()
    with pytest.warns(RuntimeWarning, match="increase n_cycles"):
        result = MHRMEstimator(n_cycles=60, burnin=50, seed=5, use_rust=False).fit(
            TwoParameterLogistic(10), responses
        )
    assert result.converged is False
    assert result.n_iterations == 60


@pytest.mark.parametrize(
    ("use_rust", "family"),
    [
        pytest.param(True, "2PL", marks=native),
        (False, "2PL"),
        (False, "GRM"),
    ],
)
def test_mhrm_log_likelihood_is_marginal_at_the_estimates(
    use_rust: bool, family: str
) -> None:
    from mirt.models.polytomous import GradedResponseModel

    if family == "GRM":
        responses = simdata("GRM", n_persons=600, n_items=6, n_categories=3, seed=2)
        model = GradedResponseModel(6, 3)
    else:
        responses, _ = _recovery_data()
        model = TwoParameterLogistic(10)
    em = mirt.fit_mirt(responses, family, compute_standard_errors=False)
    result = MHRMEstimator(n_cycles=2000, burnin=100, seed=3, use_rust=use_rust).fit(
        model, responses
    )

    expected = _grid_log_likelihood(result.model, responses)
    assert result.log_likelihood == pytest.approx(expected, rel=1e-10)
    # The likelihood at pseudo-MAP abilities sat about 1300 units above EM's.
    assert abs(result.log_likelihood - em.log_likelihood) < 2.0
    n_parameters = result.model.n_parameters
    assert result.aic == pytest.approx(-2 * expected + 2 * n_parameters)


@pytest.mark.parametrize(
    "use_rust", [pytest.param(True, marks=native), pytest.param(False)]
)
def test_mhrm_latent_prior_sets_the_scale_like_em(use_rust: bool) -> None:
    from mirt.estimation.em import EMEstimator

    responses, _ = _recovery_data()
    mean, cov = np.array([1.0]), np.array([[4.0]])
    em = EMEstimator(compute_standard_errors=False).fit(
        TwoParameterLogistic(10), responses, prior_mean=mean, prior_cov=cov
    )
    result = MHRMEstimator(n_cycles=3000, burnin=100, seed=2, use_rust=use_rust).fit(
        TwoParameterLogistic(10), responses, prior_mean=mean, prior_cov=cov
    )

    # prior_mean and prior_cov used to be swallowed by **kwargs.
    for name, tolerance in (("discrimination", 0.12), ("difficulty", 0.2)):
        difference = result.model.parameters[name] - em.model.parameters[name]
        assert np.max(np.abs(difference)) < tolerance
    np.testing.assert_array_equal(result.latent_mean, mean)
    np.testing.assert_array_equal(result.latent_covariance, cov)
    # The rescaled model has the same marginal likelihood.
    assert result.log_likelihood == pytest.approx(em.log_likelihood, abs=2.0)


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"prior_mean": np.zeros(2)}, "prior_mean"),
        ({"prior_mean": np.array([np.nan])}, "prior_mean"),
        ({"prior_cov": np.array([[-1.0]])}, "prior_cov"),
        ({"prior_cov": np.eye(2)}, "prior_cov"),
    ],
)
def test_mhrm_rejects_invalid_latent_priors(
    kwargs: dict[str, Any], parameter: str
) -> None:
    responses = simdata(n_persons=50, n_items=4, seed=0)
    with pytest.raises(MirtValidationError) as info:
        MHRMEstimator(n_cycles=5, burnin=1).fit(
            TwoParameterLogistic(4), responses, **kwargs
        )
    assert info.value.context["parameter"] == parameter


def test_fits_reject_unknown_keywords() -> None:
    responses = simdata(n_persons=50, n_items=4, seed=0)
    # A misspelled keyword used to be accepted and ignored.
    with pytest.raises(TypeError, match="priro_mean"):
        MHRMEstimator(n_cycles=5, burnin=1).fit(
            TwoParameterLogistic(4), responses, priro_mean=1.0
        )
    with pytest.raises(TypeError, match="prior_mean"):
        GibbsSampler(n_iter=5, burnin=1).fit(
            TwoParameterLogistic(4), responses, prior_mean=np.zeros(1)
        )


def test_mhrm_chains_average_several_imputations(monkeypatch) -> None:
    import mirt.backends.rust.estimation as rust_estimation

    responses, _ = _recovery_data()

    def native(*args, **kwargs):
        raise AssertionError("the native kernel draws a single chain")

    monkeypatch.setattr(rust_estimation, "mhrm_fit_2pl", native)
    results = [
        MHRMEstimator(
            n_cycles=2000, burnin=100, seed=4, n_chains=n_chains, use_rust=False
        ).fit(TwoParameterLogistic(10), responses)
        for n_chains in (1, 3)
    ]
    single, several = (result.model.parameters["difficulty"] for result in results)
    # n_chains used to be ignored.
    assert not np.allclose(single, several)
    em = mirt.fit_mirt(responses, "2PL", compute_standard_errors=False)
    assert np.max(np.abs(several - em.model.parameters["difficulty"])) < 0.1
    with pytest.warns(RuntimeWarning):
        MHRMEstimator(n_cycles=60, burnin=50, n_chains=2, use_rust=True).fit(
            TwoParameterLogistic(10), responses
        )


@pytest.mark.parametrize(
    "use_rust", [pytest.param(True, marks=native), pytest.param(False)]
)
def test_gibbs_priors_shape_the_posterior(use_rust: bool) -> None:
    from mirt.estimation.priors import NormalPrior

    responses = simdata(n_persons=200, n_items=4, seed=1)
    result = GibbsSampler(
        n_iter=300,
        burnin=100,
        seed=1,
        use_rust=use_rust,
        priors={"difficulty": NormalPrior(2.0, 0.01)},
    ).fit(TwoParameterLogistic(4), responses)

    # The priors argument used to be stored and never read.
    np.testing.assert_allclose(result.model.parameters["difficulty"], 2.0, atol=0.05)


@pytest.mark.parametrize(
    "priors",
    [
        {"discrimination": ("lognormal", 0, 1)},
        {"discrimination": {"mean": 1.0, "sd": 0.5}},
        "weak",
    ],
)
def test_gibbs_rejects_malformed_priors(priors: Any) -> None:
    with pytest.raises(MirtValidationError) as info:
        GibbsSampler(priors=priors)
    assert info.value.context["parameter"] == "priors"


def test_gibbs_rejects_a_nonstandard_ability_prior_and_unknown_names() -> None:
    from mirt.estimation.priors import NormalPrior, PriorSpecification

    with pytest.raises(MirtValidationError, match="standard normal"):
        GibbsSampler(priors=PriorSpecification(theta=NormalPrior(0.0, 2.0)))
    responses = simdata(n_persons=50, n_items=4, seed=0)
    sampler = GibbsSampler(n_iter=5, burnin=1, priors={"guessing": NormalPrior()})
    with pytest.raises(MirtValidationError, match="guessing"):
        sampler.fit(TwoParameterLogistic(4), responses)


def test_confirmatory_mhrm_holds_a_loading_pattern_and_matches_em() -> None:
    from mirt.estimation.em import EMEstimator

    rng = np.random.default_rng(9)
    n_items = 8
    pattern = np.zeros((n_items, 2))
    pattern[:4, 0] = rng.uniform(1.0, 1.8, 4)
    pattern[4:, 1] = rng.uniform(1.0, 1.8, 4)
    generator = TwoParameterLogistic(n_items, n_factors=2)
    generator.set_parameters(
        discrimination=pattern, difficulty=np.linspace(-1.0, 1.0, n_items)
    )
    theta = rng.standard_normal((1000, 2))
    responses = (rng.random((1000, n_items)) < generator.probability(theta)).astype(int)

    def model():
        built = TwoParameterLogistic(n_items, n_factors=2)
        start = (pattern != 0).astype(float)
        built.set_parameters(discrimination=start)
        return built.set_free_parameter_masks({"discrimination": pattern != 0})

    em = EMEstimator(compute_standard_errors=False).fit(
        model(), responses, start="model"
    )
    # MHRMEstimator used to refuse any fixed coordinate.
    result = MHRMEstimator(n_cycles=3000, burnin=150, seed=1).fit(model(), responses)

    slopes = result.model.parameters["discrimination"]
    assert np.all(slopes[pattern == 0] == 0.0)
    assert result.n_parameters == em.n_parameters == 16
    assert np.max(np.abs(slopes - em.model.parameters["discrimination"])) < 0.15
    difficulty = (
        result.model.parameters["difficulty"] - em.model.parameters["difficulty"]
    )
    assert np.max(np.abs(difficulty)) < 0.1
    assert result.log_likelihood == pytest.approx(em.log_likelihood, abs=2.0)


@pytest.mark.parametrize("estimation", ["MHRM", "MCMC"])
def test_fit_mirt_seed_makes_stochastic_fits_reproducible(estimation):
    import mirt

    responses = mirt.simdata(n_persons=300, n_items=6, seed=4)
    fits = [
        mirt.fit_mirt(
            responses,
            estimation=estimation,
            max_iter=300,
            seed=seed,
            compute_standard_errors=False,
        )
        for seed in (7, 7, 8)
    ]
    first, repeat, other = (fit.model.parameters["discrimination"] for fit in fits)

    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_fit_mirt_passes_priors_to_the_gibbs_sampler():
    import mirt
    from mirt.estimation.priors import NormalPrior

    responses = mirt.simdata(n_persons=300, n_items=6, seed=4)
    default, shifted = (
        mirt.fit_mirt(
            responses,
            estimation="MCMC",
            max_iter=400,
            seed=3,
            priors=priors,
            compute_standard_errors=False,
        )
        for priors in (None, {"difficulty": NormalPrior(5.0, 0.1)})
    )

    assert np.mean(shifted.model.parameters["difficulty"]) > 3.0
    assert np.mean(default.model.parameters["difficulty"]) < 2.0
    with pytest.raises(mirt.MirtValidationError, match="priors"):
        mirt.fit_mirt(
            responses, estimation="MHRM", priors={"difficulty": NormalPrior(0, 1)}
        )
