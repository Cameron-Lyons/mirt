from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

ResponseMatrix = NDArray[np.int_]
ThetaArray = NDArray[np.float64]
ThetaMatrix = NDArray[np.float64]
ParameterArray = NDArray[np.float64]
WeightArray = NDArray[np.float64]

ParameterDict = dict[str, NDArray[np.float64]]

DichotomousModelType = Literal["1PL", "2PL", "3PL", "4PL", "Rasch"]
PolytomousModelType = Literal["GRM", "GPCM", "PCM", "NRM", "RSM"]
ModelType = DichotomousModelType | PolytomousModelType

EstimationMethod = Literal["EM", "MHRM", "MCEM"]
ScoringMethod = Literal["EAP", "MAP", "ML", "WLE", "EAPsum"]
InvarianceLevel = Literal["configural", "metric", "scalar", "strict"]
ItemFitStatistic = Literal["S_X2", "X2", "G2", "Zh", "infit", "outfit"]
PersonFitStatistic = Literal["Zh", "lz", "infit", "outfit"]
RotationMethod = Literal["varimax", "promax", "oblimin", "quartimin", "equamax", "none"]

ItemModelT = TypeVar("ItemModelT", bound="BaseItemModel")
