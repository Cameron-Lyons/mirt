"""MH-RM standard errors come from the observed information at the estimates."""

from __future__ import annotations

import numpy as np
import pytest

from mirt import fit_mirt, simdata
from mirt._rust_backend import RUST_AVAILABLE
from mirt.estimation.base import _parameter_bounds
from mirt.estimation.em import EMEstimator
from mirt.estimation.mcmc import MHRMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import estimate_covariance
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _em_errors(model, data):
    result = EMEstimator(se_method="oakes").fit(model, data)
    assert result.se_method == "oakes"
    return result.standard_errors


@pytest.mark.parametrize(
    "use_rust",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not RUST_AVAILABLE, reason="Rust extension unavailable"
            ),
        ),
        False,
    ],
    ids=["native", "numpy"],
)
def test_2pl_errors_match_em_observed_information(use_rust):
    data = simdata(model="2PL", n_persons=1000, n_items=10, seed=3)
    reference = _em_errors(TwoParameterLogistic(10), data)
    result = MHRMEstimator(n_cycles=300, burnin=75, seed=1, use_rust=use_rust).fit(
        TwoParameterLogistic(10), data
    )

    assert result.se_method == "oakes"
    assert result.vcov is not None and result.vcov.shape == (20, 20)
    assert result.vcov_labels[0].startswith("discrimination[")
    for name, errors in reference.items():
        np.testing.assert_allclose(result.standard_errors[name], errors, rtol=0.15)
    np.testing.assert_allclose(
        np.sqrt(np.diag(result.vcov)),
        np.concatenate([result.standard_errors[name] for name in reference]),
    )


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (
            GradedResponseModel(5, n_categories=3),
            simdata(model="GRM", n_persons=400, n_items=5, n_categories=3, seed=5),
        ),
        (
            TwoParameterLogistic(6),
            np.where(
                np.random.default_rng(0).random((400, 6)) < 0.1,
                -1,
                simdata(model="2PL", n_persons=400, n_items=6, seed=8),
            ),
        ),
    ],
    ids=["graded", "2pl_missing"],
)
def test_errors_are_the_observed_information_at_the_estimates(model, data):
    result = MHRMEstimator(n_cycles=60, burnin=20, seed=2, use_rust=False).fit(
        model, data
    )
    quadrature = GaussHermiteQuadrature(n_points=21, n_dimensions=1)
    reference = estimate_covariance(
        result.model.copy(),
        data,
        quadrature,
        quadrature.weights,
        "oakes",
        bounds=lambda name: _parameter_bounds(result.model, name),
    )

    assert result.se_method == "oakes"
    np.testing.assert_allclose(result.vcov, reference.covariance, rtol=1e-8)
    for name, errors in reference.standard_errors.items():
        np.testing.assert_allclose(result.standard_errors[name], errors, rtol=1e-8)


def test_other_models_label_the_iterate_spread():
    data = simdata(model="2PL", n_persons=300, n_items=6, n_factors=2, seed=1)
    result = fit_mirt(data, "2PL", n_factors=2, estimation="MHRM", max_iter=40)

    assert result.se_method == "mhrm_iterate_sd"
    assert result.vcov is None
    assert result.standard_errors["discrimination"].shape == (6, 2)


def test_fit_mirt_forwards_standard_error_options(monkeypatch):
    data = simdata(model="2PL", n_persons=200, n_items=5, seed=2)
    skipped = fit_mirt(
        data, estimation="MHRM", max_iter=40, compute_standard_errors=False
    )
    assert skipped.standard_errors == {}
    assert skipped.se_method is None and skipped.vcov is None

    seen = {}
    original = MHRMEstimator.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(MHRMEstimator, "__init__", spy)
    fit_mirt(data, estimation="MHRM", max_iter=40, n_quadpts=31)
    assert seen["n_quadpts"] == 31 and seen["compute_standard_errors"] is True


@pytest.mark.parametrize("use_rust", [True, False], ids=["native", "numpy"])
def test_observations_count_persons(use_rust):
    data = simdata(model="2PL", n_persons=150, n_items=5, seed=6)
    result = MHRMEstimator(n_cycles=30, burnin=10, seed=0, use_rust=use_rust).fit(
        TwoParameterLogistic(5), data
    )
    assert result.n_observations == 150


@pytest.mark.parametrize(
    ("options", "parameter"),
    [
        ({"n_quadpts": 1}, "n_quadpts"),
        ({"n_quadpts": 2.5}, "n_quadpts"),
        ({"compute_standard_errors": "yes"}, "compute_standard_errors"),
    ],
)
def test_invalid_standard_error_options_are_rejected(options, parameter):
    with pytest.raises(MirtValidationError) as error:
        MHRMEstimator(**options)
    assert error.value.context["parameter"] == parameter
