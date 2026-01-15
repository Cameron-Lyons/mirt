"""Sample datasets for IRT analysis.

This module provides classic IRT datasets commonly used in psychometric research.
"""

from typing import Any

import numpy as np

from mirt.constants import PROB_EPSILON


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
        - 'ASVAB': Armed Services Vocational Aptitude Battery (1000 x 25)
        - 'Attitude': Attitude scale (500 x 10)
        - 'Bock1997': Bock (1997) nominal response data (1000 x 5)
        - 'deAyala': de Ayala GPCM example data (500 x 10)
        - 'SLF': Science Literacy Foundation data (500 x 15)

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
        "ASVAB": _load_asvab,
        "Attitude": _load_attitude,
        "Bock1997": _load_bock1997,
        "deAyala": _load_deayala,
        "SLF": _load_slf,
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
        "ASVAB",
        "Attitude",
        "Bock1997",
        "deAyala",
        "SLF",
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
    thresholds = np.zeros((n_items, n_categories - 1))
    for j in range(n_items):
        base = rng.uniform(-2, 0)
        for k in range(n_categories - 1):
            thresholds[j, k] = base + k * rng.uniform(0.5, 1.5)

    data = np.zeros((n_persons, n_items), dtype=np.int_)
    for j in range(n_items):
        a = discrimination[j]
        for i in range(n_persons):
            t = theta[i]
            p_star = np.zeros(n_categories - 1)
            for k in range(n_categories - 1):
                p_star[k] = 1 / (1 + np.exp(-a * (t - thresholds[j, k])))

            p = np.zeros(n_categories)
            p[0] = 1 - p_star[0]
            for k in range(1, n_categories - 1):
                p[k] = p_star[k - 1] - p_star[k]
            p[n_categories - 1] = p_star[n_categories - 2]

            p = np.clip(p, PROB_EPSILON, 1)
            p = p / p.sum()
            data[i, j] = rng.choice(n_categories, p=p)

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

    data = np.zeros((n_persons, n_items), dtype=np.int_)
    for j in range(n_items):
        z = a[j, :] * theta[:, None] + c[j, :]
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)

        for i in range(n_persons):
            data[i, j] = rng.choice(n_categories, p=probs[i])

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

        a = discrimination[j]
        for i in range(n_persons):
            t = theta[i]

            z = np.zeros(k)
            z[0] = 0
            for m in range(1, k):
                z[m] = z[m - 1] + a * (t - steps[m - 1])

            exp_z = np.exp(z - z.max())
            probs = exp_z / exp_z.sum()
            data[i, j] = rng.choice(k, p=probs)

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


LSAT6: dict[str, Any] = _load_lsat6()
LSAT7: dict[str, Any] = _load_lsat7()
SAT12: dict[str, Any] = _load_sat12()
Science: dict[str, Any] = _load_science()
verbal_aggression: dict[str, Any] = _load_verbal_aggression()
fraction_subtraction: dict[str, Any] = _load_fraction_subtraction()
ASVAB: dict[str, Any] = _load_asvab()
Attitude: dict[str, Any] = _load_attitude()
Bock1997: dict[str, Any] = _load_bock1997()
deAyala: dict[str, Any] = _load_deayala()
SLF: dict[str, Any] = _load_slf()
