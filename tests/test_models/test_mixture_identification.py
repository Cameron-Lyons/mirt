"""Simplex dimension, reconstruction, and independently integrated mixture fits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp, roots_hermite

from mirt.estimation.bl import BLEstimator
from mirt.models.mixture import MixtureIRT, fit_mixture_irt


@pytest.mark.parametrize(
    ("family", "item_dimension"), [("1PL", 1), ("2PL", 2), ("3PL", 3)]
)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_mixture_counts_independent_simplex_masses(family, item_dimension, n_classes):
    model = MixtureIRT(4, n_classes, family)
    expected = n_classes - 1 + n_classes * 4 * item_dimension
    assert model.n_parameters == expected
    assert_array_equal(
        model.free_parameter_masks["class_proportions"], np.arange(n_classes) > 0
    )
    mask = model.free_parameter_masks["class_proportions"]
    mask[1] = False
    model.set_free_parameter_masks({"class_proportions": mask})
    copied = model.copy()
    assert copied.n_parameters == model.n_parameters == expected - 1
    copied.set_free_parameter_masks(None)
    assert copied.n_parameters == expected
    assert model.n_parameters == expected - 1
    with pytest.raises(ValueError, match="cannot free"):
        model.set_free_parameter_masks(
            {"class_proportions": np.ones(n_classes, dtype=bool)}
        )


def test_valid_independent_simplex_unpacking_rebuilds_dependent_mass_and_curves():
    model = MixtureIRT(4, 3, "1PL").set_parameters(
        class_proportions=np.array([0.2, 0.3, 0.5])
    )
    estimator = BLEstimator(n_quadpts=5, max_iter=1)
    vector, _, layout = estimator._flatten_parameters(model)
    proportions_layout = layout["class_proportions"]
    assert_array_equal(proportions_layout["free_indices"], [1, 2])
    vector[proportions_layout["start_idx"] : proportions_layout["end_idx"]] = [
        0.25,
        0.35,
    ]
    estimator._unflatten_parameters(model, vector, layout)
    assert_allclose(model.class_proportions, [0.4, 0.25, 0.35])
    theta = np.array([-2, -0.1, 1.4])
    expected = np.zeros((3, 4))
    for k, mass in enumerate([0.4, 0.25, 0.35]):
        expected += mass * expit(
            theta[:, None] - model.parameters[f"difficulty_class{k}"]
        )
    assert_allclose(model.probability(theta), expected, atol=1e-15)
    with pytest.raises(ValueError, match="sum to at most 1"):
        model._canonical_parameter_values(
            "class_proportions", np.array([0.1, 0.8, 0.7])
        )


@pytest.mark.parametrize(
    "proportions", [[0.0, 0.4, 0.6], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]
)
def test_public_mixture_setter_preserves_accepted_normalized_simplex(proportions):
    values = np.asarray(proportions)
    model = MixtureIRT(4, 3).set_parameters(class_proportions=values)
    assert_array_equal(model.class_proportions, values)
    values[:] = 0.0
    assert_allclose(model.class_proportions.sum(), 1.0)


def test_dependent_zero_mass_tolerates_only_summation_roundoff():
    model = MixtureIRT(4, 21, "1PL")
    proportions = np.r_[0.0, np.full(20, 0.05)]
    assert proportions[1:].sum() > 1
    model.set_parameters(class_proportions=proportions)
    canonical = model._canonical_parameter_values("class_proportions", proportions)
    assert_array_equal(canonical, proportions)
    estimator = BLEstimator(n_quadpts=5, max_iter=1)
    vector, _, layout = estimator._flatten_parameters(model)
    estimator._unflatten_parameters(model, vector, layout)
    assert_array_equal(model.class_proportions, proportions)
    invalid = proportions.copy()
    invalid[1] += 1e-10
    with pytest.raises(ValueError, match="sum to at most 1"):
        model._canonical_parameter_values("class_proportions", invalid)


@pytest.mark.parametrize(
    ("family", "item_dimension"), [("1PL", 1), ("2PL", 2), ("3PL", 3)]
)
def test_dedicated_mixture_fit_reports_independent_dimension_and_joint_likelihood(
    family, item_dimension
):
    rng = np.random.default_rng(84)
    responses = rng.binomial(1, [0.25, 0.45, 0.65, 0.8], size=(180, 4))
    responses[::13, 1] = -1
    model, posterior = fit_mixture_irt(
        responses, n_classes=3, base_model=family, max_iter=15, tol=1e-3, n_quadpts=11
    )
    nodes, weights = roots_hermite(11)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    log_joint = np.empty((180, 3, 11))
    for k in range(3):
        item = model.get_class_parameters(k)
        curves = item["guessing"] + (1 - item["guessing"]) * expit(
            item["discrimination"] * (nodes[:, None] - item["difficulty"])
        )
        observed = responses >= 0
        terms = np.where(responses[:, None, :] == 1, np.log(curves), np.log1p(-curves))
        log_joint[:, k] = (
            np.where(observed[:, None, :], terms, 0).sum(axis=2)
            + np.log(model.class_proportions[k])
            + np.log(weights)
        )
    log_evidence = logsumexp(log_joint, axis=(1, 2))
    independent_ll = log_evidence.sum()
    independent_posterior = np.exp(log_joint - log_evidence[:, None, None]).sum(axis=2)
    info = model.convergence_info
    dimension = 2 + 3 * 4 * item_dimension
    assert info is not None
    assert info["n_parameters"] == model.n_parameters == dimension
    assert_allclose(info["log_likelihood"], independent_ll, atol=1e-10)
    assert_allclose(info["aic"], -2 * independent_ll + 2 * dimension, atol=1e-10)
    assert_allclose(
        info["bic"], -2 * independent_ll + np.log(180) * dimension, atol=1e-10
    )
    assert_allclose(posterior, independent_posterior, atol=1e-12)
    assert np.all(model.class_proportions >= 0)
    assert_allclose(model.class_proportions.sum(), 1, atol=1e-15)
    assert np.all(np.diff(info["log_likelihood_history"]) >= -1e-9)
