"""
mirt: Multidimensional Item Response Theory for Python
======================================================

A comprehensive Python implementation of IRT models inspired by R's mirt package.

Main Functions
--------------
fit_mirt : Fit unidimensional or multidimensional IRT models
fscores : Compute person ability estimates (theta scores)
simdata : Simulate response data from IRT models
itemfit : Compute item fit statistics
personfit : Compute person fit statistics

Models
------
Dichotomous (binary) models:
- OneParameterLogistic (1PL/Rasch)
- TwoParameterLogistic (2PL)
- ThreeParameterLogistic (3PL)
- FourParameterLogistic (4PL)

Polytomous (ordinal) models:
- GradedResponseModel (GRM)
- GeneralizedPartialCredit (GPCM)
- PartialCreditModel (PCM)
- NominalResponseModel (NRM)

Examples
--------
>>> import mirt
>>> import numpy as np
>>>
>>> # Simulate data
>>> responses = mirt.simdata(model='2PL', n_persons=500, n_items=20, seed=42)
>>>
>>> # Fit model
>>> result = mirt.fit_mirt(responses, model='2PL')
>>> print(result.summary())
>>>
>>> # Score respondents
>>> scores = mirt.fscores(result, responses, method='EAP')
>>> print(scores.to_dataframe().head())
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mirt._version import __version__

# Model classes
from mirt.models.dichotomous import (
    OneParameterLogistic,
    TwoParameterLogistic,
    ThreeParameterLogistic,
    FourParameterLogistic,
    Rasch,
)
from mirt.models.polytomous import (
    GradedResponseModel,
    GeneralizedPartialCredit,
    PartialCreditModel,
    NominalResponseModel,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.bifactor import BifactorModel
from mirt.models.base import BaseItemModel

# Estimation
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature

# Results
from mirt.results.fit_result import FitResult
from mirt.results.score_result import ScoreResult

# Scoring
from mirt.scoring import fscores

# Utilities
from mirt.utils.simulation import simdata, generate_item_parameters
from mirt.utils.data import validate_responses


def fit_mirt(
    data: NDArray[np.int_],
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_factors: int = 1,
    n_categories: Optional[int] = None,
    estimation: Literal["EM"] = "EM",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: Optional[list[str]] = None,
) -> FitResult:
    """Fit a (multidimensional) IRT model.

    This is the main function for fitting IRT models. It supports both
    dichotomous and polytomous models with unidimensional or multidimensional
    latent structures.

    Parameters
    ----------
    data : ndarray of shape (n_persons, n_items)
        Response matrix. For dichotomous models, responses should be 0/1.
        For polytomous models, responses should be 0, 1, ..., K-1.
        Missing values should be coded as -1.
    model : {'1PL', '2PL', '3PL', '4PL', 'GRM', 'GPCM', 'PCM', 'NRM'}, default='2PL'
        IRT model type:
        - '1PL': One-parameter logistic (Rasch) model
        - '2PL': Two-parameter logistic model
        - '3PL': Three-parameter logistic model (with guessing)
        - '4PL': Four-parameter logistic model
        - 'GRM': Graded Response Model (for ordered polytomous)
        - 'GPCM': Generalized Partial Credit Model
        - 'PCM': Partial Credit Model
        - 'NRM': Nominal Response Model
    n_factors : int, default=1
        Number of latent dimensions.
    n_categories : int, optional
        Number of response categories for polytomous models.
        If None, inferred from data.
    estimation : {'EM'}, default='EM'
        Estimation method. Currently only EM is supported.
    n_quadpts : int, default=21
        Number of quadrature points per dimension.
    max_iter : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Whether to print progress information.
    item_names : list of str, optional
        Names for items.

    Returns
    -------
    FitResult
        Object containing fitted model, parameters, standard errors,
        and fit statistics.

    Examples
    --------
    >>> # Fit a 2PL model
    >>> result = mirt.fit_mirt(responses, model='2PL')
    >>> print(result.summary())

    >>> # Fit a 3-factor exploratory model
    >>> result = mirt.fit_mirt(responses, model='2PL', n_factors=3)

    >>> # Fit a graded response model for Likert data
    >>> result = mirt.fit_mirt(likert_data, model='GRM', n_categories=5)
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i+1}" for i in range(n_items)]

    # Determine if polytomous
    is_polytomous = model in ("GRM", "GPCM", "PCM", "NRM")

    if is_polytomous:
        if n_categories is None:
            # Infer from data
            n_categories = int(data[data >= 0].max()) + 1
        if n_categories < 2:
            raise ValueError("n_categories must be at least 2")

    # Create model
    if model == "1PL":
        irt_model = OneParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "2PL":
        irt_model = TwoParameterLogistic(
            n_items=n_items, n_factors=n_factors, item_names=item_names
        )
    elif model == "3PL":
        irt_model = ThreeParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "4PL":
        irt_model = FourParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "GRM":
        irt_model = GradedResponseModel(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    elif model == "GPCM":
        irt_model = GeneralizedPartialCredit(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    elif model == "PCM":
        irt_model = PartialCreditModel(
            n_items=n_items,
            n_categories=n_categories,
            item_names=item_names,
        )
    elif model == "NRM":
        irt_model = NominalResponseModel(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Create estimator
    if estimation == "EM":
        estimator = EMEstimator(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown estimation method: {estimation}")

    # Fit model
    result = estimator.fit(irt_model, data)

    return result


def itemfit(
    result: FitResult,
    responses: Optional[NDArray[np.int_]] = None,
    statistics: Optional[list[str]] = None,
) -> "pd.DataFrame":
    """Compute item fit statistics.

    Parameters
    ----------
    result : FitResult
        Fitted model result.
    responses : ndarray, optional
        Response matrix. Required for some fit statistics.
    statistics : list of str, optional
        Which statistics to compute. Default: ['infit', 'outfit'].

    Returns
    -------
    pandas.DataFrame
        DataFrame with fit statistics for each item.
    """
    import pandas as pd
    from mirt.diagnostics.itemfit import compute_itemfit

    if statistics is None:
        statistics = ["infit", "outfit"]

    fit_stats = compute_itemfit(result.model, responses, statistics)

    df = pd.DataFrame(fit_stats)
    df.index = result.model.item_names
    df.index.name = "item"

    return df


def personfit(
    result: FitResult,
    responses: NDArray[np.int_],
    theta: Optional[NDArray[np.float64]] = None,
    statistics: Optional[list[str]] = None,
) -> "pd.DataFrame":
    """Compute person fit statistics.

    Parameters
    ----------
    result : FitResult
        Fitted model result.
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    theta : ndarray, optional
        Person ability values. If None, computed via EAP.
    statistics : list of str, optional
        Which statistics to compute. Default: ['infit', 'outfit', 'Zh'].

    Returns
    -------
    pandas.DataFrame
        DataFrame with fit statistics for each person.
    """
    import pandas as pd
    from mirt.diagnostics.personfit import compute_personfit

    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    if theta is None:
        # Compute theta via EAP
        score_result = fscores(result, responses, method="EAP")
        theta = score_result.theta

    fit_stats = compute_personfit(result.model, responses, theta, statistics)

    df = pd.DataFrame(fit_stats)
    df.index.name = "person"

    return df


__all__ = [
    # Version
    "__version__",
    # High-level functions
    "fit_mirt",
    "fscores",
    "simdata",
    "itemfit",
    "personfit",
    # Model classes
    "OneParameterLogistic",
    "TwoParameterLogistic",
    "ThreeParameterLogistic",
    "FourParameterLogistic",
    "Rasch",
    "GradedResponseModel",
    "GeneralizedPartialCredit",
    "PartialCreditModel",
    "NominalResponseModel",
    "MultidimensionalModel",
    "BifactorModel",
    "BaseItemModel",
    # Estimation
    "EMEstimator",
    "GaussHermiteQuadrature",
    # Results
    "FitResult",
    "ScoreResult",
    # Utilities
    "generate_item_parameters",
    "validate_responses",
]
