"""Sample datasets for IRT analysis.

This module provides classic IRT datasets commonly used in psychometric research.
Dataset prototypes are generated once per process and copied on demand so repeated
loads remain fast without sharing mutable response matrices between callers.
"""

from collections.abc import Callable
from difflib import get_close_matches
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

_DATASET_LOADERS: dict[str, Callable[[], dict[str, Any]]] = {
    "LSAT6": lambda: _load_lsat6(),
    "LSAT7": lambda: _load_lsat7(),
    "SAT12": lambda: _load_sat12(),
    "Science": lambda: _load_science(),
    "verbal_aggression": lambda: _load_verbal_aggression(),
    "fraction_subtraction": lambda: _load_fraction_subtraction(),
    "ASVAB": lambda: _load_asvab(),
    "Attitude": lambda: _load_attitude(),
    "Bock1997": lambda: _load_bock1997(),
    "deAyala": lambda: _load_deayala(),
    "SLF": lambda: _load_slf(),
}
_DATASET_NAME_INDEX = {name.casefold(): name for name in _DATASET_LOADERS}


def _resolve_dataset_name(name: str) -> str:
    """Resolve a case-insensitive dataset name to its canonical spelling."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")

    normalized = name.strip().casefold()
    try:
        return _DATASET_NAME_INDEX[normalized]
    except KeyError:
        available = ", ".join(_DATASET_LOADERS)
        matches = get_close_matches(normalized, _DATASET_NAME_INDEX, n=1, cutoff=0.6)
        suggestion = (
            f" Did you mean '{_DATASET_NAME_INDEX[matches[0]]}'?" if matches else ""
        )
        raise ValueError(
            f"Unknown dataset: {name}.{suggestion} Available: {available}"
        ) from None


@cache
def _load_dataset_prototype(name: str) -> dict[str, Any]:
    """Generate and retain the private prototype for a canonical dataset name."""
    prototype = _DATASET_LOADERS[name]()
    _freeze_dataset_arrays(prototype)
    return prototype


def _freeze_dataset_arrays(value: Any) -> None:
    """Make every array in a cached prototype immutable in place."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_dataset_arrays(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _freeze_dataset_arrays(item)


def _clone_dataset_value(value: Any, *, copy_arrays: bool) -> Any:
    """Clone dataset containers while optionally sharing read-only array storage."""
    if isinstance(value, np.ndarray):
        if copy_arrays:
            return value.copy()
        view = value.view()
        view.flags.writeable = False
        return view
    if isinstance(value, dict):
        return {
            key: _clone_dataset_value(item, copy_arrays=copy_arrays)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_dataset_value(item, copy_arrays=copy_arrays) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _clone_dataset_value(item, copy_arrays=copy_arrays) for item in value
        )
    if isinstance(value, set):
        return {_clone_dataset_value(item, copy_arrays=copy_arrays) for item in value}
    return value


def load_dataset(name: str, *, copy: bool = True) -> dict[str, Any]:
    """Load a sample dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset. Available datasets:
        - 'LSAT6': Law School Admission Test, Section 6 (1000 x 5)
        - 'LSAT7': Law School Admission Test, Section 7 (1000 x 5)
        - 'SAT12': Scholastic Assessment Test items (500 x 12)
        - 'Science': Science assessment (500 x 25)
        - 'verbal_aggression': Verbal Aggression rating scale (316 x 24)
        - 'fraction_subtraction': Fraction subtraction skills (536 x 20)
        - 'ASVAB': Armed Services Vocational Aptitude Battery (1000 x 25)
        - 'Attitude': Attitude scale (500 x 10)
        - 'Bock1997': Bock (1997) nominal response data (1000 x 5)
        - 'deAyala': de Ayala GPCM example data (500 x 10)
        - 'SLF': Science Literacy Foundation data (500 x 15)
    copy : bool, default=True
        Return independent, writable copies of array values. Set to False to share
        cached array storage through read-only views, which minimizes allocations
        for workloads that treat sample data as immutable.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data': Response matrix (NDArray)
        - 'description': Dataset description
        - 'n_persons': Number of respondents
        - 'n_items': Number of items
        - 'source': Citation/reference
        - Additional metadata depending on dataset
    """
    canonical_name = _resolve_dataset_name(name)
    prototype = _load_dataset_prototype(canonical_name)
    return _clone_dataset_value(prototype, copy_arrays=copy)


def describe_dataset(name: str) -> dict[str, Any]:
    """Return lightweight metadata for a sample dataset.

    Array-valued response data and simulation parameters are omitted, making this
    suitable for discovery interfaces that do not need response-matrix copies.
    Loading metadata also warms the process-local dataset cache.

    Parameters
    ----------
    name : str
        Dataset name. Matching is case-insensitive and ignores surrounding space.

    Returns
    -------
    dict
        Dataset name and non-array metadata such as description, dimensions,
        item names, and source.
    """
    canonical_name = _resolve_dataset_name(name)
    prototype = _load_dataset_prototype(canonical_name)
    metadata = {
        key: value
        for key, value in prototype.items()
        if key != "data" and not isinstance(value, np.ndarray)
    }
    return {
        "name": canonical_name,
        **_clone_dataset_value(metadata, copy_arrays=True),
    }


def list_datasets() -> list[str]:
    """List available dataset names in stable display order."""
    return list(_DATASET_LOADERS.keys())


def _inverse_cdf_sample(
    probabilities: np.ndarray,
    uniforms: np.ndarray,
) -> NDArray[np.int_]:
    """Draw categorical values from pre-generated uniforms in one batch."""
    cumulative = np.cumsum(probabilities[..., :-1], axis=-1)
    return np.sum(uniforms[..., None] >= cumulative, axis=-1).astype(
        np.int_, copy=False
    )


def _load_lsat6() -> dict[str, Any]:
    """LSAT Section 6 data from Bock & Lieberman (1970).

    5 binary items from the Law School Admission Test.
    Classic dataset used in IRT literature.
    """
    patterns = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int_,
    )

    frequencies = np.array(
        [
            3,
            6,
            2,
            11,
            1,
            1,
            3,
            4,
            1,
            8,
            0,
            16,
            3,
            15,
            10,
            56,
            0,
            3,
            0,
            4,
            1,
            6,
            2,
            20,
            3,
            28,
            15,
            81,
            16,
            56,
            21,
            173,
        ]
    )

    data = np.repeat(patterns, frequencies, axis=0)

    return {
        "data": data,
        "description": "LSAT Section 6: 5 binary items from Law School Admission Test",
        "n_persons": data.shape[0],
        "n_items": data.shape[1],
        "item_names": [f"Item{i + 1}" for i in range(5)],
        "source": "Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. Psychometrika, 35, 179-197.",
    }


def _load_lsat7() -> dict[str, Any]:
    """LSAT Section 7 data from Bock & Aitkin (1981).

    5 binary items from the Law School Admission Test.
    """
    patterns = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int_,
    )

    frequencies = np.array(
        [
            12,
            19,
            1,
            7,
            3,
            19,
            3,
            17,
            10,
            5,
            3,
            7,
            7,
            23,
            13,
            59,
            4,
            28,
            3,
            14,
            8,
            51,
            15,
            90,
            6,
            63,
            39,
            175,
            35,
            89,
            42,
            110,
        ]
    )

    data = np.repeat(patterns, frequencies, axis=0)

    return {
        "data": data,
        "description": "LSAT Section 7: 5 binary items from Law School Admission Test",
        "n_persons": data.shape[0],
        "n_items": data.shape[1],
        "item_names": [f"Item{i + 1}" for i in range(5)],
        "source": "Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. Psychometrika, 46, 443-459.",
    }


def _load_sat12() -> dict[str, Any]:
    """SAT-like assessment data (simulated based on typical SAT characteristics).

    12 binary items with varying difficulty and discrimination.
    """
    rng = np.random.default_rng(12345)
    n_persons = 500
    n_items = 12

    theta = rng.standard_normal(n_persons)

    discrimination = np.array(
        [0.8, 1.2, 1.0, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 1.2, 0.85]
    )
    difficulty = np.array(
        [-1.5, -1.0, -0.5, 0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    )

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    prob = 1 / (1 + np.exp(-z))
    data = (rng.random((n_persons, n_items)) < prob).astype(np.int_)

    return {
        "data": data,
        "description": "SAT12: 12 binary items simulated with SAT-like characteristics",
        "n_persons": n_persons,
        "n_items": n_items,
        "item_names": [f"Item{i + 1}" for i in range(n_items)],
        "true_discrimination": discrimination,
        "true_difficulty": difficulty,
        "true_theta": theta,
        "source": "Simulated data based on typical SAT item characteristics",
    }


def _load_science() -> dict[str, Any]:
    """Science assessment data (simulated based on educational assessment).

    25 binary items covering various science topics.
    """
    rng = np.random.default_rng(54321)
    n_persons = 500
    n_items = 25

    theta = rng.standard_normal(n_persons)

    discrimination = rng.uniform(0.5, 2.0, n_items)
    difficulty = rng.uniform(-2.5, 2.5, n_items)

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    prob = 1 / (1 + np.exp(-z))
    data = (rng.random((n_persons, n_items)) < prob).astype(np.int_)

    return {
        "data": data,
        "description": "Science: 25 binary items from science achievement assessment",
        "n_persons": n_persons,
        "n_items": n_items,
        "item_names": [f"Sci{i + 1}" for i in range(n_items)],
        "true_discrimination": discrimination,
        "true_difficulty": difficulty,
        "true_theta": theta,
        "source": "Simulated educational assessment data",
    }


def _load_verbal_aggression() -> dict[str, Any]:
    """Verbal Aggression data based on De Boeck & Wilson (2004).

    24 items measuring verbal aggression tendencies.
    Polytomous responses: 0 = no, 1 = perhaps, 2 = yes
    """
    rng = np.random.default_rng(98765)
    n_persons = 316
    n_items = 24
    n_categories = 3

    theta = rng.standard_normal(n_persons)

    discrimination = rng.uniform(0.8, 1.8, n_items)
    threshold1 = rng.uniform(-1.5, 0.5, n_items)
    threshold2 = threshold1 + rng.uniform(0.5, 2.0, n_items)

    logits = discrimination[None, :] * (theta[:, None] - threshold1[None, :])
    upper_logits = discrimination[None, :] * (theta[:, None] - threshold2[None, :])
    first_cumulative = 1.0 / (1.0 + np.exp(-logits))
    second_cumulative = 1.0 / (1.0 + np.exp(-upper_logits))
    uniforms = rng.random((n_items, n_persons)).T
    data = (uniforms >= 1.0 - first_cumulative).astype(np.int_) + (
        uniforms >= 1.0 - second_cumulative
    ).astype(np.int_)

    behaviors = ["Curse", "Scold", "Shout", "Curse", "Scold", "Shout"] * 4
    situations = ["Bus", "Bus", "Bus", "Train", "Train", "Train"] * 4
    modes = ["Want"] * 12 + ["Do"] * 12

    return {
        "data": data,
        "description": "Verbal Aggression: 24 polytomous items (3 categories) measuring verbal aggression",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_names": [f"VA{i + 1}" for i in range(n_items)],
        "item_behavior": behaviors,
        "item_situation": situations,
        "item_mode": modes,
        "response_labels": ["no", "perhaps", "yes"],
        "true_theta": theta,
        "source": "Based on De Boeck, P., & Wilson, M. (2004). Explanatory Item Response Models. Springer.",
    }


def _load_fraction_subtraction() -> dict[str, Any]:
    """Fraction subtraction data for cognitive diagnosis.

    20 items testing fraction subtraction skills.
    Includes Q-matrix for cognitive diagnosis models.
    """
    rng = np.random.default_rng(11111)
    n_persons = 536
    n_items = 20
    n_attributes = 5

    q_matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int_,
    )

    attr_prob = np.array([0.8, 0.6, 0.5, 0.4, 0.3])
    alpha = (rng.random((n_persons, n_attributes)) < attr_prob).astype(np.int_)

    slip = rng.uniform(0.05, 0.25, n_items)
    guess = rng.uniform(0.05, 0.25, n_items)

    data = np.zeros((n_persons, n_items), dtype=np.int_)
    for j in range(n_items):
        required = q_matrix[j]
        eta = np.all(alpha >= required, axis=1).astype(np.int_)

        prob = (1 - slip[j]) ** eta * guess[j] ** (1 - eta)
        data[:, j] = (rng.random(n_persons) < prob).astype(np.int_)

    return {
        "data": data,
        "description": "Fraction Subtraction: 20 binary items for cognitive diagnosis",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_attributes": n_attributes,
        "item_names": [f"FS{i + 1}" for i in range(n_items)],
        "attribute_names": [
            "basic_subtraction",
            "reduce",
            "separate",
            "borrow",
            "convert",
        ],
        "q_matrix": q_matrix,
        "true_alpha": alpha,
        "true_slip": slip,
        "true_guess": guess,
        "source": "Based on Tatsuoka, K. K. (1984). Analysis of errors in fraction addition and subtraction problems.",
    }


def _load_asvab() -> dict[str, Any]:
    """Armed Services Vocational Aptitude Battery data.

    25 binary items measuring general aptitude.
    Classic multidimensional IRT dataset.
    """
    rng = np.random.default_rng(77777)
    n_persons = 1000
    n_items = 25
    n_factors = 4

    theta = rng.multivariate_normal(
        np.zeros(n_factors),
        np.array(
            [
                [1.0, 0.3, 0.2, 0.1],
                [0.3, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.3],
                [0.1, 0.2, 0.3, 1.0],
            ]
        ),
        n_persons,
    )

    discrimination = np.zeros((n_items, n_factors))
    items_per_factor = n_items // n_factors
    for f in range(n_factors):
        start = f * items_per_factor
        end = start + items_per_factor if f < n_factors - 1 else n_items
        for j in range(start, end):
            discrimination[j, f] = rng.uniform(0.8, 2.0)
            for f2 in range(n_factors):
                if f2 != f:
                    discrimination[j, f2] = rng.uniform(0.0, 0.3)

    difficulty = rng.uniform(-2.0, 2.0, n_items)

    z = theta @ discrimination.T - difficulty[None, :]
    prob = 1 / (1 + np.exp(-z))
    data = (rng.random((n_persons, n_items)) < prob).astype(np.int_)

    subtests = (
        ["General Science"] * 6
        + ["Arithmetic Reasoning"] * 6
        + ["Word Knowledge"] * 7
        + ["Paragraph Comprehension"] * 6
    )

    return {
        "data": data,
        "description": "ASVAB: 25 binary items from Armed Services Vocational Aptitude Battery",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_factors": n_factors,
        "item_names": [f"ASVAB{i + 1}" for i in range(n_items)],
        "subtests": subtests,
        "true_discrimination": discrimination,
        "true_difficulty": difficulty,
        "true_theta": theta,
        "source": "Simulated based on ASVAB characteristics. See Mislevy (1986).",
    }


def _load_attitude() -> dict[str, Any]:
    """Attitude scale data.

    10 items measuring general attitude with 5 response categories.
    """
    rng = np.random.default_rng(88888)
    n_persons = 500
    n_items = 10
    n_categories = 5

    theta = rng.standard_normal(n_persons)

    discrimination = rng.uniform(0.8, 1.8, n_items)
    threshold_draws = rng.random((n_items, n_categories))
    bases = -2.0 + 2.0 * threshold_draws[:, 0]
    increments = 0.5 + threshold_draws[:, 1:]
    thresholds = bases[:, None] + np.arange(n_categories - 1)[None, :] * increments

    logits = discrimination[None, :, None] * (
        theta[:, None, None] - thresholds[None, :, :]
    )
    cumulative = 1.0 / (1.0 + np.exp(-logits))
    probabilities = np.concatenate(
        (
            1.0 - cumulative[..., :1],
            cumulative[..., :-1] - cumulative[..., 1:],
            cumulative[..., -1:],
        ),
        axis=-1,
    )
    probabilities = np.clip(probabilities, PROB_EPSILON, 1.0)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    uniforms = rng.random((n_items, n_persons)).T
    data = _inverse_cdf_sample(probabilities, uniforms)

    return {
        "data": data,
        "description": "Attitude: 10 polytomous items (5 categories) measuring attitude",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_names": [f"Att{i + 1}" for i in range(n_items)],
        "response_labels": [
            "Strongly Disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly Agree",
        ],
        "true_discrimination": discrimination,
        "true_thresholds": thresholds,
        "true_theta": theta,
        "source": "Simulated Likert-type attitude scale data.",
    }


def _load_bock1997() -> dict[str, Any]:
    """Bock (1997) nominal response model data.

    5 items with 4 nominal response categories.
    Classic example for NRM.
    """
    rng = np.random.default_rng(19970)
    n_persons = 1000
    n_items = 5
    n_categories = 4

    theta = rng.standard_normal(n_persons)

    a = np.array(
        [
            [-0.5, 0.0, 0.5, 1.0],
            [-0.3, 0.2, 0.8, 1.2],
            [-0.8, -0.2, 0.4, 0.9],
            [-0.4, 0.1, 0.6, 1.1],
            [-0.6, 0.0, 0.5, 0.8],
        ]
    )

    c = np.array(
        [
            [0.5, 0.3, -0.2, -0.6],
            [0.4, 0.2, -0.1, -0.5],
            [0.6, 0.4, 0.0, -0.4],
            [0.3, 0.1, -0.3, -0.7],
            [0.5, 0.2, -0.1, -0.5],
        ]
    )

    logits = theta[:, None, None] * a[None, :, :] + c[None, :, :]
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    uniforms = rng.random((n_items, n_persons)).T
    data = _inverse_cdf_sample(probabilities, uniforms)

    return {
        "data": data,
        "description": "Bock1997: 5 nominal response items (4 categories)",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_names": [f"NRM{i + 1}" for i in range(n_items)],
        "true_a": a,
        "true_c": c,
        "true_theta": theta,
        "source": "Based on Bock, R.D. (1997). The nominal categories model. In W.J. van der Linden & R.K. Hambleton (Eds.), Handbook of modern item response theory.",
    }


def _load_deayala() -> dict[str, Any]:
    """de Ayala GPCM example data.

    10 items with varying numbers of categories (3-5).
    Example for Generalized Partial Credit Model.
    """
    rng = np.random.default_rng(99999)
    n_persons = 500
    n_items = 10

    n_cats = [4, 4, 5, 3, 4, 5, 3, 4, 4, 5]

    theta = rng.standard_normal(n_persons)

    discrimination = rng.uniform(0.8, 1.8, n_items)

    data = np.zeros((n_persons, n_items), dtype=np.int_)
    step_params = []

    for j in range(n_items):
        k = n_cats[j]
        steps = np.sort(rng.uniform(-2, 2, k - 1))
        step_params.append(steps)

        increments = discrimination[j] * (theta[:, None] - steps[None, :])
        logits = np.concatenate(
            (np.zeros((n_persons, 1)), np.cumsum(increments, axis=1)),
            axis=1,
        )
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        data[:, j] = _inverse_cdf_sample(probabilities, rng.random(n_persons))

    return {
        "data": data,
        "description": "deAyala: 10 polytomous items for GPCM (3-5 categories)",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories_per_item": n_cats,
        "item_names": [f"GPCM{i + 1}" for i in range(n_items)],
        "true_discrimination": discrimination,
        "true_step_params": step_params,
        "true_theta": theta,
        "source": "Simulated based on de Ayala, R.J. (2009). The Theory and Practice of Item Response Theory. Guilford Press.",
    }


def _load_slf() -> dict[str, Any]:
    """Science Literacy Foundation data.

    15 binary items measuring scientific literacy.
    Bifactor structure with general factor and specific factors.
    """
    rng = np.random.default_rng(55555)
    n_persons = 500
    n_items = 15
    n_specific = 3

    general = rng.standard_normal(n_persons)
    specific = rng.standard_normal((n_persons, n_specific)) * 0.7

    a_general = rng.uniform(0.5, 1.5, n_items)
    a_specific = np.zeros((n_items, n_specific))

    items_per_specific = n_items // n_specific
    for s in range(n_specific):
        start = s * items_per_specific
        end = start + items_per_specific if s < n_specific - 1 else n_items
        for j in range(start, end):
            a_specific[j, s] = rng.uniform(0.5, 1.2)

    difficulty = rng.uniform(-2.0, 2.0, n_items)

    z = (
        general[:, None] * a_general[None, :]
        + np.sum(specific[:, :, None] * a_specific.T[None, :, :], axis=1)
        - difficulty[None, :]
    )
    prob = 1 / (1 + np.exp(-z))
    data = (rng.random((n_persons, n_items)) < prob).astype(np.int_)

    content_areas = ["Physics"] * 5 + ["Chemistry"] * 5 + ["Biology"] * 5

    return {
        "data": data,
        "description": "SLF: 15 binary items measuring science literacy (bifactor structure)",
        "n_persons": n_persons,
        "n_items": n_items,
        "n_specific_factors": n_specific,
        "item_names": [f"SLF{i + 1}" for i in range(n_items)],
        "content_areas": content_areas,
        "true_a_general": a_general,
        "true_a_specific": a_specific,
        "true_difficulty": difficulty,
        "true_general_theta": general,
        "true_specific_theta": specific,
        "source": "Simulated bifactor data for science literacy assessment.",
    }


def __getattr__(name: str) -> dict[str, Any]:
    if name in _DATASET_LOADERS:
        dataset = load_dataset(name)
        globals()[name] = dataset
        return dataset
    raise AttributeError(f"module 'mirt.utils.datasets' has no attribute '{name}'")
