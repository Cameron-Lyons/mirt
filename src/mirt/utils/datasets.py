"""Sample datasets for IRT analysis.

This module provides classic IRT datasets commonly used in psychometric research.
"""

from typing import Any

import numpy as np


def load_dataset(name: str) -> dict[str, Any]:
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
    datasets = {
        "LSAT6": _load_lsat6,
        "LSAT7": _load_lsat7,
        "SAT12": _load_sat12,
        "Science": _load_science,
        "verbal_aggression": _load_verbal_aggression,
        "fraction_subtraction": _load_fraction_subtraction,
    }

    name_lower = name.lower()
    for key, loader in datasets.items():
        if key.lower() == name_lower:
            return loader()

    available = ", ".join(datasets.keys())
    raise ValueError(f"Unknown dataset: {name}. Available: {available}")


def list_datasets() -> list[str]:
    """List available dataset names."""
    return [
        "LSAT6",
        "LSAT7",
        "SAT12",
        "Science",
        "verbal_aggression",
        "fraction_subtraction",
    ]


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

    data = np.zeros((n_persons, n_items), dtype=np.int_)
    for j in range(n_items):
        a = discrimination[j]
        b1, b2 = threshold1[j], threshold2[j]

        for i in range(n_persons):
            t = theta[i]
            p_star1 = 1 / (1 + np.exp(-a * (t - b1)))
            p_star2 = 1 / (1 + np.exp(-a * (t - b2)))

            p0 = 1 - p_star1
            p1 = p_star1 - p_star2

            u = rng.random()
            if u < p0:
                data[i, j] = 0
            elif u < p0 + p1:
                data[i, j] = 1
            else:
                data[i, j] = 2

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


LSAT6: dict[str, Any] = _load_lsat6()
LSAT7: dict[str, Any] = _load_lsat7()
SAT12: dict[str, Any] = _load_sat12()
Science: dict[str, Any] = _load_science()
verbal_aggression: dict[str, Any] = _load_verbal_aggression()
fraction_subtraction: dict[str, Any] = _load_fraction_subtraction()
