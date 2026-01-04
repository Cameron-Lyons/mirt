from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

from mirt._version import __version__
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    Rasch,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.results.fit_result import FitResult
from mirt.results.score_result import ScoreResult
from mirt.scoring import fscores
from mirt.utils.data import validate_responses
from mirt.utils.simulation import generate_item_parameters, simdata


def fit_mirt(
    data: NDArray[np.int_],
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_factors: int = 1,
    n_categories: int | None = None,
    estimation: Literal["EM"] = "EM",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
) -> FitResult:
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    is_polytomous = model in ("GRM", "GPCM", "PCM", "NRM")

    if is_polytomous:
        if n_categories is None:
            n_categories = int(data[data >= 0].max()) + 1
        if n_categories < 2:
            raise ValueError("n_categories must be at least 2")

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

    if estimation == "EM":
        estimator = EMEstimator(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown estimation method: {estimation}")

    result = estimator.fit(irt_model, data)

    return result


def itemfit(
    result: FitResult,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
) -> "pd.DataFrame":
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
    theta: NDArray[np.float64] | None = None,
    statistics: list[str] | None = None,
) -> "pd.DataFrame":
    import pandas as pd

    from mirt.diagnostics.personfit import compute_personfit

    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    if theta is None:
        score_result = fscores(result, responses, method="EAP")
        theta = score_result.theta

    fit_stats = compute_personfit(result.model, responses, theta, statistics)

    df = pd.DataFrame(fit_stats)
    df.index.name = "person"

    return df


__all__ = [
    "__version__",
    "fit_mirt",
    "fscores",
    "simdata",
    "itemfit",
    "personfit",
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
    "EMEstimator",
    "GaussHermiteQuadrature",
    "FitResult",
    "ScoreResult",
    "generate_item_parameters",
    "validate_responses",
]
