from typing import Literal

DichotomousModelType = Literal["1PL", "2PL", "3PL", "4PL", "Rasch"]
PolytomousModelType = Literal["GRM", "GPCM", "PCM", "NRM"]
ModelType = DichotomousModelType | PolytomousModelType

CDMModelType = Literal["DINA", "DINO", "GDINA"]

AdvancedModelType = Literal[
    "Testlet", "MixtureIRT", "ZI-2PL", "ZI-3PL", "GGUM", "IdealPoint", "HCM"
]

AllModelType = ModelType | CDMModelType | AdvancedModelType

EstimationMethod = Literal["EM", "MHRM", "MCMC", "Gibbs"]

ScoringMethod = Literal["EAP", "MAP", "ML"]

InvarianceLevel = Literal["configural", "metric", "scalar", "strict"]

ItemFitStatistic = Literal["infit", "outfit", "S_X2"]
PersonFitStatistic = Literal["Zh", "infit", "outfit"]
ModelFitIndex = Literal["M2", "RMSEA", "CFI", "TLI", "SRMSR"]

DIFMethod = Literal["likelihood_ratio", "wald", "lord", "raju"]
DTFMethod = Literal["signed", "unsigned", "expected_score"]
SIBTESTMethod = Literal["original", "crossing"]

ImputationMethod = Literal["mean", "mode", "random", "EM", "multiple"]

BootstrapCIMethod = Literal["percentile", "BCa", "basic"]

ItemSelectionMethod = Literal["MFI", "MEI", "KL", "Urry", "random", "a-stratified"]
StoppingMethod = Literal[
    "SE", "max_items", "min_items", "theta_change", "classification"
]
ExposureMethod = Literal["sympson-hetter", "randomesque", "progressive", "none"]
