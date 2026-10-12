from typing import Literal

DichotomousModelType = Literal["1PL", "2PL", "3PL", "4PL", "Rasch"]
PolytomousModelType = Literal["GRM", "GPCM", "PCM", "NRM"]
ModelType = DichotomousModelType | PolytomousModelType

CDMModelType = Literal["DINA", "DINO", "GDINA", "ACDM", "LLM", "RRUM"]

# ``model_name`` values of advanced item-model classes. No fitting function
# accepts these names: construct the class (for example
# ``mirt.models.GeneralizedGradedUnfolding``) and fit it with an estimator.
AdvancedModelType = Literal[
    "Testlet", "MixtureIRT", "ZI-2PL", "ZI-3PL", "GGUM", "IdealPoint", "HCM"
]

# Every model name that a name-based fitting function accepts: the item
# families of ``fit_mirt(model=...)`` and the models of ``fit_cdm(model=...)``.
AllModelType = (
    Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] | CDMModelType
)

EstimationMethod = Literal["EM", "MHRM", "MCMC", "Gibbs"]

ScoringMethod = Literal["EAP", "MAP", "ML"]

InvarianceLevel = Literal["configural", "metric", "scalar", "strict"]

ItemFitStatistic = Literal[
    "infit", "outfit", "z_infit", "z_outfit", "S_X2", "X2", "G2", "PV_Q1", "Zh"
]
PersonFitStatistic = Literal["Zh", "infit", "outfit", "lz", "z_infit", "z_outfit"]
ModelFitIndex = Literal["M2", "RMSEA", "CFI", "TLI", "SRMSR"]
M2Type = Literal["M2", "M2*", "C2"]

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
