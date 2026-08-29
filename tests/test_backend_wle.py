from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.backends import rust as rb
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.scoring.wle import WLEScorer


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    responses = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, -1, 1, 0],
            [1, -8, -1, 0, 1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    discrimination = np.array([0.7, 1.1, 1.6, 0.9, 1.3])
    difficulty = np.array([-1.2, -0.4, 0.1, 0.8, 1.5])
    return responses, discrimination, difficulty


def test_numpy_wle_batch_matches_scalar_reference_with_missing_data() -> None:
    responses, discrimination, difficulty = _inputs()
    model = TwoParameterLogistic(n_items=responses.shape[1])
    model.set_parameters(
        discrimination=discrimination,
        difficulty=difficulty,
    )
    model._is_fitted = True
    scorer = WLEScorer(bounds=(-4.5, 5.0), tol=1e-8)
    normalized = np.where(responses >= 0, responses, -1)
    expected = [scorer._estimate_person(model, row) for row in normalized]

    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        theta, standard_error = rb.compute_wle_scores(
            responses,
            discrimination,
            difficulty,
            theta_min=-4.5,
            theta_max=5.0,
            tol=1e-8,
            n_jobs=-1,
        )
    finally:
        mirt.set_backend(previous)

    np.testing.assert_allclose(theta, np.asarray(expected)[:, 0], atol=1e-7)
    np.testing.assert_allclose(
        standard_error,
        np.asarray(expected)[:, 1],
        atol=1e-7,
    )
    assert theta[-1] == 0.0
    assert np.isinf(standard_error[-1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"responses": np.ones(3)}, "two-dimensional"),
        ({"responses": np.array([[0.5, 1.0]])}, "integer-valued"),
        ({"responses": np.array([[0, 2]])}, "only 0 or 1"),
        ({"discrimination": np.ones(3)}, "shape"),
        ({"difficulty": np.array([0.0, np.nan])}, "finite"),
        ({"theta_min": 2.0, "theta_max": -2.0}, "strictly increasing"),
        ({"tol": 0.0}, "positive"),
        ({"n_jobs": 0}, "positive integer"),
        ({"n_jobs": True}, "positive integer"),
    ],
)
def test_wle_batch_validates_inputs(kwargs: dict[str, object], message: str) -> None:
    responses = np.array([[0, 1], [1, 0]])
    options: dict[str, object] = {
        "responses": responses,
        "discrimination": np.ones(2),
        "difficulty": np.zeros(2),
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        rb.compute_wle_scores(**options)  # type: ignore[arg-type]


def test_wle_batch_supports_empty_response_matrices() -> None:
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        theta, standard_error = rb.compute_wle_scores(
            np.empty((0, 2), dtype=np.int64),
            np.ones(2),
            np.zeros(2),
        )
    finally:
        mirt.set_backend(previous)

    assert theta.shape == (0,)
    assert standard_error.shape == (0,)
