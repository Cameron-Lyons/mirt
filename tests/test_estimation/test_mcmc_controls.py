"""Sampler controls and native/NumPy agreement for MHRM and Gibbs estimation."""

from __future__ import annotations

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
