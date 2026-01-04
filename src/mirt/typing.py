from typing import Literal

DichotomousModelType = Literal["1PL", "2PL", "3PL", "4PL", "Rasch"]
PolytomousModelType = Literal["GRM", "GPCM", "PCM", "NRM"]
ModelType = DichotomousModelType | PolytomousModelType

EstimationMethod = Literal["EM"]
ScoringMethod = Literal["EAP", "MAP", "ML"]
InvarianceLevel = Literal["configural", "metric", "scalar", "strict"]
ItemFitStatistic = Literal["infit", "outfit"]
PersonFitStatistic = Literal["Zh", "infit", "outfit"]
