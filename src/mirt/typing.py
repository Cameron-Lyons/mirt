"""Type definitions for the mirt package."""

from typing import Literal, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# Array types
ResponseMatrix = NDArray[np.int_]  # Shape: (n_persons, n_items)
ThetaArray = NDArray[np.float64]  # Shape: (n_persons,) or (n_persons, n_factors)
ThetaMatrix = NDArray[np.float64]  # Shape: (n_persons, n_factors)
ParameterArray = NDArray[np.float64]  # Shape varies by parameter type
WeightArray = NDArray[np.float64]  # Shape: (n_quadpts,) or (n_quadpts^n_factors,)

# Parameter dictionary type
ParameterDict = dict[str, NDArray[np.float64]]

# Model type literals
DichotomousModelType = Literal["1PL", "2PL", "3PL", "4PL", "Rasch"]
PolytomousModelType = Literal["GRM", "GPCM", "PCM", "NRM", "RSM"]
ModelType = Union[DichotomousModelType, PolytomousModelType]

# Estimation method literals
EstimationMethod = Literal["EM", "MHRM", "MCEM"]

# Scoring method literals
ScoringMethod = Literal["EAP", "MAP", "ML", "WLE", "EAPsum"]

# Invariance levels for multiple group analysis
InvarianceLevel = Literal["configural", "metric", "scalar", "strict"]

# Item fit statistics
ItemFitStatistic = Literal["S_X2", "X2", "G2", "Zh", "infit", "outfit"]

# Person fit statistics
PersonFitStatistic = Literal["Zh", "lz", "infit", "outfit"]

# Rotation methods
RotationMethod = Literal[
    "varimax", "promax", "oblimin", "quartimin", "equamax", "none"
]

# Generic type variable for item models
ItemModelT = TypeVar("ItemModelT", bound="BaseItemModel")  # type: ignore[name-defined]
