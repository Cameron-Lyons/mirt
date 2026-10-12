"""Sample datasets for IRT analysis.

``LSAT6`` and ``LSAT7`` are the published Law School Admission Test pattern
tables of Bock and Lieberman (1970), identical to the datasets of R's ltm and
mirt packages. Every other dataset is a simulated stand-in, not the R mirt
data or the original study data of the same name: its size and coding differ
(see each loader and ``describe_dataset``), and it carries ``"simulated":
True`` and its generating parameters. Code ported from R, such as
``key2binary(SAT12, key)``, does not apply to these stand-ins.

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
        Name of the dataset. Two datasets are published data:

        - 'LSAT6': Law School Admission Test, Section 6 (1000 x 5, binary)
        - 'LSAT7': Law School Admission Test, Section 7 (1000 x 5, binary)

        The others are simulated stand-ins, not the R mirt data (or the
        original study data) of the same name, with different sizes and
        coding:

        - 'SAT12': 500 x 12 binary 2PL responses; R mirt's SAT12 is
          600 x 32 unscored multiple-choice responses (1-5, 8 = missing)
        - 'Science': 500 x 25 binary 2PL responses; R mirt's Science is
          392 x 4 Likert responses (1-4)
        - 'verbal_aggression': 316 x 24 graded responses (0-2)
        - 'fraction_subtraction': 536 x 20 binary DINA responses with a
          5-attribute Q-matrix
        - 'ASVAB': 1000 x 25 binary responses from a 4-factor model
        - 'Attitude': 500 x 10 graded responses (0-4)
        - 'Bock1997': 1000 x 5 nominal responses (0-3); R mirt's Bock1997
          is a 3-item nominal table (1-4) of 2000 respondents
        - 'deAyala': 500 x 10 GPCM responses (0-2 to 0-4); R mirt's deAyala
          is a 5-item binary table of 19601 respondents
        - 'SLF': 500 x 15 binary bifactor responses; R mirt's SLF (Social
          Life Feelings) is a 5-item binary table of 1490 respondents
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
        - 'simulated': True for the simulated stand-ins, whose
          'description' starts with "Simulated stand-in"
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
    """List available dataset names in stable display order.

    Only ``LSAT6`` and ``LSAT7`` are published data. The other names are
    simulated stand-ins, not the R mirt data of the same name: their sizes
    and coding differ, as ``describe_dataset(name)["description"]`` states,
    and their metadata carry ``"simulated": True``.
    """
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
    """LSAT Section 6 data (Bock & Lieberman, 1970; Thissen, 1982).

    5 binary items from the Law School Admission Test, 1000 examinees.
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

    # Bock & Lieberman (1970) LSAT Section 6 pattern counts (1000 examinees),
    # as distributed with R's ltm and mirt packages. Patterns 01010 and 01100
    # were not observed.
    frequencies = np.array(
        [3, 6, 2, 11, 1, 1, 3, 4, 1, 8, 0, 16, 0, 3, 2, 15]
        + [10, 29, 14, 81, 3, 28, 15, 80, 16, 56, 21, 173, 11, 61, 28, 298]
    )

    data = np.repeat(patterns, frequencies, axis=0)

    return {
        "data": data,
        "description": "LSAT Section 6: 5 binary items from Law School Admission Test",
        "n_persons": data.shape[0],
        "n_items": data.shape[1],
        "item_names": [f"Item{i + 1}" for i in range(5)],
        "simulated": False,
        "source": "Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. Psychometrika, 35, 179-197; Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model. Psychometrika, 47, 175-186.",
    }


def _load_lsat7() -> dict[str, Any]:
    """LSAT Section 7 data (Bock & Lieberman, 1970; Bock & Aitkin, 1981).

    5 binary items from the Law School Admission Test, 1000 examinees.
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

    # Bock & Lieberman (1970) LSAT Section 7 pattern counts (1000 examinees),
    # as distributed with R's ltm and mirt packages.
    frequencies = np.array(
        [12, 19, 1, 7, 3, 19, 3, 17, 10, 5, 3, 7, 7, 23, 8, 28]
        + [7, 39, 11, 34, 14, 51, 15, 90, 6, 25, 7, 35, 18, 136, 32, 308]
    )

    data = np.repeat(patterns, frequencies, axis=0)

    return {
        "data": data,
        "description": "LSAT Section 7: 5 binary items from Law School Admission Test",
        "n_persons": data.shape[0],
        "n_items": data.shape[1],
        "item_names": [f"Item{i + 1}" for i in range(5)],
        "simulated": False,
        "source": "Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. Psychometrika, 46, 443-459.",
    }


def _load_sat12() -> dict[str, Any]:
    """Simulated stand-in, not the R mirt SAT12 data.

    500 respondents by 12 binary (0/1) items drawn from a unidimensional 2PL
    with varying difficulty and discrimination. R mirt's SAT12 holds 600
    unscored multiple-choice responses to 32 items (options 1-5, 8 for
    missing) that ``key2binary`` scores.
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
        "description": (
            "Simulated stand-in, not the R mirt data: 500 x 12 binary (0/1) "
            "responses from a 2PL. R mirt's SAT12 is 600 x 32 unscored "
            "multiple-choice responses (1-5, 8 = missing)."
        ),
        "simulated": True,
        "n_persons": n_persons,
        "n_items": n_items,
        "item_names": [f"Item{i + 1}" for i in range(n_items)],
        "true_discrimination": discrimination,
        "true_difficulty": difficulty,
        "true_theta": theta,
        "source": "Simulated 2PL responses; not the TESTFACT SAT12 data of R mirt.",
    }


def _load_science() -> dict[str, Any]:
    """Simulated stand-in, not the R mirt Science data.

    500 respondents by 25 binary (0/1) items drawn from a unidimensional 2PL.
    R mirt's Science holds 392 Likert responses (1-4) to 4 items from the
    Euro-Barometer survey (the ltm ``Science`` data).
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
        "description": (
            "Simulated stand-in, not the R mirt data: 500 x 25 binary (0/1) "
            "responses from a 2PL. R mirt's Science is 392 x 4 Likert "
            "responses (1-4)."
        ),
        "simulated": True,
        "n_persons": n_persons,
        "n_items": n_items,
        "item_names": [f"Sci{i + 1}" for i in range(n_items)],
        "true_discrimination": discrimination,
        "true_difficulty": difficulty,
        "true_theta": theta,
        "source": "Simulated 2PL responses; not the Euro-Barometer Science data of R mirt.",
    }


def _load_verbal_aggression() -> dict[str, Any]:
    """Simulated stand-in for the verbal aggression data of De Boeck & Wilson (2004).

    316 respondents by 24 graded items with responses 0 = no, 1 = perhaps,
    2 = yes, drawn from a unidimensional graded response model. The sizes and
    coding follow the original study, but the responses and the item design
    labels are simulated and illustrative, not the original data.
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
        "description": (
            "Simulated stand-in, not the original verbal aggression data: "
            "316 x 24 graded responses (0 = no, 1 = perhaps, 2 = yes) from a "
            "graded response model."
        ),
        "simulated": True,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_names": [f"VA{i + 1}" for i in range(n_items)],
        "item_behavior": behaviors,
        "item_situation": situations,
        "item_mode": modes,
        "response_labels": ["no", "perhaps", "yes"],
        "true_theta": theta,
        "source": "Simulated after De Boeck, P., & Wilson, M. (2004). Explanatory Item Response Models. Springer.",
    }


def _load_fraction_subtraction() -> dict[str, Any]:
    """Simulated stand-in for Tatsuoka's fraction subtraction data.

    536 respondents by 20 binary items drawn from a DINA model with an
    illustrative 5-attribute Q-matrix. The sizes follow Tatsuoka (1984), but
    the responses and the Q-matrix are simulated, not the original data or
    its 8-attribute Q-matrix.
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
        "description": (
            "Simulated stand-in, not the original fraction subtraction data: "
            "536 x 20 binary (0/1) responses from a DINA model with a "
            "5-attribute Q-matrix."
        ),
        "simulated": True,
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
        "source": "Simulated after Tatsuoka, K. K. (1984). Analysis of errors in fraction addition and subtraction problems.",
    }


def _load_asvab() -> dict[str, Any]:
    """Simulated stand-in for Armed Services Vocational Aptitude Battery data.

    1000 respondents by 25 binary items drawn from a compensatory 4-factor
    model with correlated factors. These are not ASVAB responses.
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
        "description": (
            "Simulated stand-in, not ASVAB data: 1000 x 25 binary (0/1) "
            "responses from a compensatory 4-factor model."
        ),
        "simulated": True,
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
    """Simulated attitude scale data.

    500 respondents by 10 graded items with 5 categories (0-4), drawn from a
    graded response model.
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
        "description": (
            "Simulated stand-in: 500 x 10 graded responses (0-4) from a "
            "graded response model."
        ),
        "simulated": True,
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
    """Simulated stand-in, not the R mirt Bock1997 data.

    1000 respondents by 5 items with 4 nominal categories (0-3), drawn from a
    nominal response model. R mirt's Bock1997 is the 3-item table (categories
    1-4, 2000 respondents) of Bock (1997, Table 3).
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
        "description": (
            "Simulated stand-in, not the R mirt data: 1000 x 5 nominal "
            "responses (0-3). R mirt's Bock1997 is a 3-item nominal table "
            "(1-4) of 2000 respondents."
        ),
        "simulated": True,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_names": [f"NRM{i + 1}" for i in range(n_items)],
        "true_a": a,
        "true_c": c,
        "true_theta": theta,
        "source": "Simulated after Bock, R.D. (1997). The nominal categories model. In W.J. van der Linden & R.K. Hambleton (Eds.), Handbook of modern item response theory.",
    }


def _load_deayala() -> dict[str, Any]:
    """Simulated stand-in, not the R mirt deAyala data.

    500 respondents by 10 items with 3 to 5 categories (coded from 0), drawn
    from a generalized partial credit model. R mirt's deAyala is the 5-item
    binary mathematics table of de Ayala (2009, p. 14), 19601 respondents.
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
        "description": (
            "Simulated stand-in, not the R mirt data: 500 x 10 GPCM responses "
            "with 3-5 categories (0-2 to 0-4). R mirt's deAyala is a 5-item "
            "binary table of 19601 respondents."
        ),
        "simulated": True,
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
    """Simulated stand-in, not the R mirt SLF data.

    500 respondents by 15 binary items drawn from a bifactor model with one
    general and three specific factors; the content-area labels are
    illustrative. R mirt's SLF is the Social Life Feelings table of
    Bartholomew (1998): 5 binary items, 1490 respondents.
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
        "description": (
            "Simulated stand-in, not the R mirt data: 500 x 15 binary (0/1) "
            "responses from a bifactor model. R mirt's SLF (Social Life "
            "Feelings) is a 5-item binary table of 1490 respondents."
        ),
        "simulated": True,
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
        "source": "Simulated bifactor responses; not the Social Life Feelings data of R mirt.",
    }


def __getattr__(name: str) -> dict[str, Any]:
    if name in _DATASET_LOADERS:
        dataset = load_dataset(name)
        globals()[name] = dataset
        return dataset
    raise AttributeError(f"module 'mirt.utils.datasets' has no attribute '{name}'")
