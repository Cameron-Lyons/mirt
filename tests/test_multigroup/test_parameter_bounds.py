"""The multigroup item optimizer uses the single-group parameter boxes."""

import numpy as np
import pytest

from mirt.estimation.base import _parameter_bounds
from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.multigroup import MultigroupEMEstimator, MultigroupModel

MODELS = {
    "1PL": lambda: OneParameterLogistic(4),
    "2PL": lambda: TwoParameterLogistic(4),
    "2PL-2D": lambda: TwoParameterLogistic(4, n_factors=2),
    "3PL": lambda: ThreeParameterLogistic(4),
    "4PL": lambda: FourParameterLogistic(4),
    "5PL": lambda: FiveParameterLogistic(4),
    "GRM": lambda: GradedResponseModel(4, n_categories=4),
    "GRM-2D": lambda: GradedResponseModel(4, n_categories=4, n_factors=2),
    "GPCM": lambda: GeneralizedPartialCredit(4, n_categories=4),
    "PCM": lambda: PartialCreditModel(4, n_categories=3),
    "NRM": lambda: NominalResponseModel(4, n_categories=3),
}


@pytest.mark.parametrize("kind", list(MODELS))
def test_multigroup_bounds_match_single_group_bounds(kind):
    model = MODELS[kind]()
    for name in model.parameters:
        expected = _parameter_bounds(model, name)
        if kind == "2PL-2D" and name == "discrimination":
            # Exploratory binary slopes may be negative in multigroup fits.
            expected = (-5.0, 5.0)
        assert MultigroupEMEstimator._parameter_bound(model, name) == expected


def test_five_parameter_logistic_fits_with_positive_asymmetry():
    rng = np.random.default_rng(4)
    n_items = 6
    slopes = rng.uniform(0.8, 1.6, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    responses = []
    for shift in (0.0, 0.3):
        theta = rng.normal(shift, 1.0, 300)
        probability = 1.0 / (1.0 + np.exp(-slopes * (theta[:, None] - difficulty)))
        responses.append((rng.random(probability.shape) < probability).astype(int))
    model = MultigroupModel(FiveParameterLogistic(n_items=n_items), n_groups=2)

    # The asymmetry box was (-10, 10), so the first M-step tried nonpositive
    # values and the model rejected them.
    result = MultigroupEMEstimator(n_quadpts=11, max_iter=3).fit(
        model, responses, invariance="configural"
    )

    assert np.isfinite(result.log_likelihood)
    for group in range(2):
        asymmetry = model.get_group_model(group).parameters["asymmetry"]
        assert np.all((asymmetry >= 0.1) & (asymmetry <= 5.0))
