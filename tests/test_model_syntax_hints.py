"""Warnings that point model-syntax fits to faster or exact alternatives."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mirt
from mirt.estimation.priors import NormalPrior

BIFACTOR = "G = 1-9\nS1 = 1-3\nS2 = 4-6\nS3 = 7-9"


def _bifactor_responses(n_persons: int = 120) -> np.ndarray:
    rng = np.random.default_rng(3)
    specific = np.repeat([0, 1, 2], 3)
    theta = rng.standard_normal((n_persons, 4))
    logits = 1.1 * theta[:, :1] + 0.7 * theta[:, 1 + specific] - 0.2
    return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)


def _fit(spec: str, **kwargs: object) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mirt.fit_mirt(
            _bifactor_responses(),
            spec=spec,
            n_quadpts=5,
            max_iter=1,
            compute_standard_errors=False,
            **kwargs,
        )
    return [entry for entry in caught if "bfactor" in str(entry.message)]


def test_bifactor_shaped_syntax_points_to_bfactor() -> None:
    hints = _fit(BIFACTOR)

    assert len(hints) == 1
    assert hints[0].category is RuntimeWarning
    assert hints[0].filename == __file__
    message = str(hints[0].message)
    assert "general factor G" in message
    assert "625 quadrature nodes" in message
    assert "0 for S1, 1 for S2, 2 for S3" in message


@pytest.mark.parametrize(
    ("spec", "kwargs"),
    [
        (BIFACTOR + "\nCOV = S1*S2", {}),
        ("F1 = 1-5\nF2 = 4-9\nF3 = 7-9", {}),
        ("G = 1-9\nS1 = 1-3\nS2 = 4-9", {"priors": {"slopes": NormalPrior(1.0, 1.0)}}),
        ("G = 1-9\nS1 = 1-5\nS2 = 4-9", {}),
        ("G = 1-9\nS1 = 1-9", {}),
        ("G = 1-9\nS1 = 1-3\nS2 = 4-6\nS3 = 7-9\nCONSTRAIN = (1-3, a1)", {}),
    ],
    ids=["covariance", "no-general", "priors", "overlap", "one-specific", "constrain"],
)
def test_other_structures_do_not_point_to_bfactor(
    spec: str, kwargs: dict[str, object]
) -> None:
    assert _fit(spec, **kwargs) == []


def test_grm_bifactor_shape_points_to_graded_bfactor() -> None:
    hints = _fit(BIFACTOR, model="GRM")

    assert len(hints) == 1
    assert 'mirt.bfactor(data, specific_factors, model="graded")' in str(
        hints[0].message
    )


def test_squarem_with_estimated_covariance_warns_and_runs_plain_em() -> None:
    data = mirt.simdata("2PL", n_persons=300, n_items=6, seed=5)
    spec = "F1 = 1-3\nF2 = 4-6\nCOV = F1*F2"
    plain = mirt.fit_mirt(
        data, spec=spec, n_quadpts=7, max_iter=30, compute_standard_errors=False
    )

    with pytest.warns(UserWarning, match="fixed Gaussian latent density"):
        accelerated = mirt.fit_mirt(
            data,
            spec=spec,
            n_quadpts=7,
            max_iter=30,
            accelerate="squarem",
            compute_standard_errors=False,
        )

    assert accelerated.n_iterations == plain.n_iterations
    assert accelerated.log_likelihood == pytest.approx(plain.log_likelihood)
