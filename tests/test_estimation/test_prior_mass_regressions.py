"""Regression coverage for normalized Gauss-Hermite prior masses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import mirt
from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt._rust_backend import (
    RUST_AVAILABLE,
    cat_eap_update,
    e_step_complete,
    em_iteration_2pl,
    multigroup_e_step_2pl,
    multigroup_e_step_3pl,
    multigroup_e_step_gpcm,
    multigroup_e_step_grm,
    multigroup_e_step_nrm,
)
from mirt.estimation.em import EMEstimator
from mirt.estimation.irtree_em import IRTreeEMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.regularized import RegularizedMIRTEstimator
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.irtree import IRTreeModel
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.latent import MultigroupLatentDensity
from mirt.multigroup.model import MultigroupModel


def _assert_gaussian_moments(
    mass: np.ndarray,
    quadrature: GaussHermiteQuadrature,
    mean: float,
    variance: float,
) -> None:
    points = quadrature.nodes.ravel()
    observed_mean = float(mass @ points)
    np.testing.assert_allclose(mass.sum(), 1.0, atol=2e-14)
    np.testing.assert_allclose(observed_mean, mean, atol=1e-10)
    np.testing.assert_allclose(
        mass @ np.square(points - observed_mean), variance, atol=1e-10
    )


@pytest.mark.parametrize("mean, variance", [(0.0, 1.0), (0.75, 0.8), (-0.5, 1.2)])
def test_gaussian_density_ratio_recovers_requested_moments(
    mean: float,
    variance: float,
) -> None:
    quadrature = GaussHermiteQuadrature(n_points=31)
    mass = np.exp(
        gaussian_log_quadrature_mass(
            quadrature.nodes,
            quadrature.weights,
            np.array([mean]),
            np.array([[variance]]),
        )
    )

    _assert_gaussian_moments(mass, quadrature, mean, variance)
    if mean == 0.0 and variance == 1.0:
        np.testing.assert_allclose(mass, quadrature.weights, atol=1e-15)


def test_python_e_steps_retain_standard_prior_without_information() -> None:
    responses = np.array([[-1]], dtype=np.int_)

    core = EMEstimator(n_quadpts=21, use_gpu=False)
    core._quadrature = GaussHermiteQuadrature(n_points=21)
    core._latent_density = GaussianDensity()
    posterior, marginal = core._e_step(TwoParameterLogistic(1), responses)
    np.testing.assert_allclose(posterior[0], core._quadrature.weights, atol=1e-14)
    np.testing.assert_allclose(marginal, 1.0, atol=1e-14)

    weighted = WeightedEMEstimator(n_quadpts=21)
    weighted._quadrature = GaussHermiteQuadrature(n_points=21)
    posterior, marginal = weighted._e_step_weighted(
        TwoParameterLogistic(1),
        responses,
        prior_mean=np.zeros(1),
        prior_cov=np.eye(1),
        weights=np.ones(1),
    )
    np.testing.assert_allclose(posterior[0], weighted._quadrature.weights, atol=1e-14)
    np.testing.assert_allclose(marginal, 1.0, atol=1e-14)


def test_multidimensional_python_e_steps_retain_standard_prior() -> None:
    regularized = RegularizedMIRTEstimator(n_factors=2, n_quadpts=7)
    regularized._quadrature = GaussHermiteQuadrature(n_points=7, n_dimensions=2)
    posterior, marginal = regularized._e_step(
        responses=np.array([[-1]], dtype=np.int_),
        loadings=np.ones((1, 2)),
        intercepts=np.zeros(1),
        latent_density=GaussianDensity(n_dimensions=2),
    )
    np.testing.assert_allclose(
        posterior[0], regularized._quadrature.weights, atol=1e-14
    )
    np.testing.assert_allclose(marginal, 1.0, atol=1e-14)

    model = IRTreeModel(n_items=1, tree_spec="direction_intensity")
    n_nodes = model.parameters["discrimination"].shape[1]
    irtree = IRTreeEMEstimator(n_quadpts=7)
    irtree._quadrature = GaussHermiteQuadrature(n_points=7, n_dimensions=model.n_traits)
    posterior, marginal = irtree._e_step(
        model,
        np.full((1, 1, n_nodes), -1, dtype=np.int_),
        np.zeros((1, n_nodes), dtype=np.int_),
        np.zeros((1, 1, n_nodes), dtype=np.bool_),
        trait_mean=np.zeros(model.n_traits),
        trait_cov=np.eye(model.n_traits),
    )
    np.testing.assert_allclose(posterior[0], irtree._quadrature.weights, atol=1e-14)
    np.testing.assert_allclose(marginal, 1.0, atol=1e-14)


def test_multigroup_python_e_steps_use_normalized_group_masses() -> None:
    previous_backend = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        model = MultigroupModel(TwoParameterLogistic(n_items=1), n_groups=2)
        estimator = MultigroupEMEstimator(n_quadpts=21)
        estimator._quadrature = GaussHermiteQuadrature(n_points=21)
        estimator._latent_density = MultigroupLatentDensity(n_groups=2)
        responses = [np.array([[-1]], dtype=np.int_) for _ in range(2)]

        for method in (estimator._e_step, estimator._e_step_python):
            posterior, group_lls = method(model, responses)
            for group_posterior in posterior:
                np.testing.assert_allclose(
                    group_posterior[0], estimator._quadrature.weights, atol=1e-14
                )
            np.testing.assert_allclose(group_lls, np.zeros(2), atol=1e-14)
    finally:
        mirt.set_backend(previous_backend)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_complete_e_step_and_cat_update_preserve_prior_moments(backend: str) -> None:
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("Rust extension is not available")

    previous_backend = mirt.get_backend()
    try:
        mirt.set_backend(backend)
        quadrature = GaussHermiteQuadrature(n_points=31)
        mean, variance = 0.75, 0.8
        posterior, marginal = e_step_complete(
            np.array([[-1]], dtype=np.int_),
            quadrature.nodes.ravel(),
            quadrature.weights,
            np.ones(1),
            np.zeros(1),
            prior_mean=mean,
            prior_var=variance,
        )
        _assert_gaussian_moments(posterior[0], quadrature, mean, variance)
        np.testing.assert_allclose(marginal, 1.0, atol=1e-14)

        theta, standard_error = cat_eap_update(
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
            np.ones(1),
            np.zeros(1),
            quadrature.nodes.ravel(),
            quadrature.weights,
        )
        np.testing.assert_allclose(theta, 0.0, atol=1e-14)
        np.testing.assert_allclose(standard_error, 1.0, atol=1e-14)
    finally:
        mirt.set_backend(previous_backend)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is not available")
def test_native_iteration_uses_normalized_shifted_prior() -> None:
    previous_backend = mirt.get_backend()
    try:
        mirt.set_backend("rust")
        quadrature = GaussHermiteQuadrature(n_points=31)
        result = em_iteration_2pl(
            np.array([[-1]], dtype=np.int_),
            quadrature.nodes.ravel(),
            quadrature.weights,
            np.ones(1),
            np.zeros(1),
            prior_mean=0.75,
            prior_var=0.8,
            max_m_iter=1,
        )
        assert result is not None
        posterior, log_likelihood = result[2], result[3]
        _assert_gaussian_moments(posterior[0], quadrature, 0.75, 0.8)
        np.testing.assert_allclose(log_likelihood, 0.0, atol=1e-14)
    finally:
        mirt.set_backend(previous_backend)


def _native_fit_responses() -> np.ndarray:
    rng = np.random.default_rng(4127)
    theta = rng.standard_normal(600)
    discrimination = np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.8])
    difficulty = np.array([-1.4, -0.8, -0.25, 0.3, 0.9, 1.4])
    probability = 1.0 / (
        1.0 + np.exp(-discrimination[None, :] * (theta[:, None] - difficulty[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(np.int_)
    responses[rng.random(responses.shape) < 0.1] = -1
    return responses


@pytest.mark.parametrize("n_quadpts", [15, 21])
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is not available")
def test_native_fit_tracks_python_with_accurate_quadrature(n_quadpts: int) -> None:
    responses = _native_fit_responses()
    previous_backend = mirt.get_backend()
    try:
        mirt.set_backend("rust")
        native = mirt.fit_mirt(
            responses,
            model="2PL",
            n_quadpts=n_quadpts,
            max_iter=100,
            tol=1e-6,
            use_rust=True,
        )
        mirt.set_backend("numpy")
        python = mirt.fit_mirt(
            responses,
            model="2PL",
            n_quadpts=n_quadpts,
            max_iter=100,
            tol=1e-6,
            use_rust=False,
        )
    finally:
        mirt.set_backend(previous_backend)

    relative_likelihood_gap = abs(native.log_likelihood - python.log_likelihood) / abs(
        python.log_likelihood
    )
    assert relative_likelihood_gap < 1e-7
    np.testing.assert_allclose(
        native.model.discrimination,
        python.model.discrimination,
        rtol=2e-3,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        native.model.difficulty,
        python.model.difficulty,
        rtol=2e-3,
        atol=5e-4,
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is not available")
def test_native_fit_reports_likelihood_for_returned_parameters() -> None:
    responses = _native_fit_responses()[:100]
    result = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=9,
        max_iter=1,
        use_rust=True,
    )
    quadrature = GaussHermiteQuadrature(n_points=9)
    _, marginal = e_step_complete(
        responses,
        quadrature.nodes.ravel(),
        quadrature.weights,
        result.model.discrimination,
        result.model.difficulty,
    )
    expected_likelihood = float(np.log(marginal).sum())

    assert result.log_likelihood == pytest.approx(expected_likelihood, abs=1e-10)
    assert result.aic == pytest.approx(
        -2.0 * expected_likelihood + 2.0 * result.n_parameters
    )
    assert result.bic == pytest.approx(
        -2.0 * expected_likelihood + np.log(len(responses)) * result.n_parameters
    )


@pytest.mark.parametrize(
    "e_step",
    [
        multigroup_e_step_2pl,
        multigroup_e_step_3pl,
        multigroup_e_step_grm,
        multigroup_e_step_gpcm,
        multigroup_e_step_nrm,
    ],
)
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is not available")
def test_native_multigroup_families_normalize_each_group_prior(
    e_step: Callable,
) -> None:
    previous_backend = mirt.get_backend()
    try:
        mirt.set_backend("rust")
        quadrature = GaussHermiteQuadrature(n_points=31)
        means = np.array([0.75, -0.5])
        variances = np.array([0.8, 1.2])
        responses = [np.array([[-1]], dtype=np.int_) for _ in means]
        discrimination = [np.ones(1) for _ in means]
        difficulty = [np.zeros(1) for _ in means]
        categories = [np.array([3], dtype=np.int_) for _ in means]

        if e_step is multigroup_e_step_2pl:
            result = e_step(
                responses,
                quadrature.nodes.ravel(),
                quadrature.weights,
                discrimination,
                difficulty,
                means,
                variances,
            )
        elif e_step is multigroup_e_step_3pl:
            result = e_step(
                responses,
                quadrature.nodes.ravel(),
                quadrature.weights,
                discrimination,
                difficulty,
                [np.full(1, 0.2) for _ in means],
                means,
                variances,
            )
        else:
            if e_step is multigroup_e_step_grm:
                first_parameters = discrimination
                second_parameters = [np.zeros((1, 2)) for _ in means]
            elif e_step is multigroup_e_step_gpcm:
                first_parameters = discrimination
                second_parameters = [np.zeros((1, 3)) for _ in means]
            else:
                first_parameters = [np.zeros((1, 3)) for _ in means]
                second_parameters = [np.zeros((1, 3)) for _ in means]
            result = e_step(
                responses,
                quadrature.nodes.ravel(),
                quadrature.weights,
                first_parameters,
                second_parameters,
                categories,
                means,
                variances,
            )

        assert result is not None
        posterior, group_lls = result
        for group_index, (mean, variance) in enumerate(
            zip(means, variances, strict=True)
        ):
            _assert_gaussian_moments(
                np.asarray(posterior[group_index])[0],
                quadrature,
                float(mean),
                float(variance),
            )
        np.testing.assert_allclose(group_lls, np.zeros(2), atol=1e-14)
    finally:
        mirt.set_backend(previous_backend)
