API Reference
=============

Core Functions
--------------

.. currentmodule:: mirt

.. autosummary::
   :toctree: generated
   :nosignatures:

   fit_mirt
   itemfit
   personfit
   dif
   load_dataset
   list_datasets
   describe_dataset

Models
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   OneParameterLogistic
   TwoParameterLogistic
   ThreeParameterLogistic
   FourParameterLogistic
   GradedResponseModel
   GeneralizedPartialCredit
   PartialCreditModel
   NominalResponseModel
   TwoPLNestedLogit
   ThreePLNestedLogit
   FourPLNestedLogit
   BifactorModel
   MixtureIRT
   TestletModel
   ResponseTimeModel
   DINA
   DINO
   IRTreeModel

Custom Models
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   create_item_type
   CustomItemModel
   ItemTypeSpec
   create_group
   createGroup
   CustomGroupModel
   GroupSpec

Network Models
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   models.IsingModel
   models.GaussianGraphicalModel
   models.fit_ising
   models.fit_ggm
   models.compare_networks

Estimation
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   EMEstimator
   MHRMEstimator
   GibbsSampler
   MCMCResult
   BLEstimator
   ResponseTimeGibbsSampler
   IRTreeEMEstimator
   IRTreeResult

Diagnostics
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_fit_indices
   compare_models
   anova_irt
   vuong_test
   compute_dtf
   compute_drf
   sibtest
   diagnostics.adjust_p_values
   diagnostics.psis_loo
   diagnostics.waic

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_icc
   plot_category_curves
   plot_information
   plot_expected_score
   plot_se
   plot_ability_distribution
   plot_person_item_map
   plot_itemfit
   plot_dif

Scoring
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   fscores
   ability_posterior

Cross-validation
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cross_validate
   KFold
   StratifiedKFold
   GroupKFold
   StratifiedGroupKFold
   LeaveOneOut

CAT
---

.. autosummary::
   :toctree: generated
   :nosignatures:

   CATEngine
   CATResult
   CATState

Equating
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   link
   true_score_equating
   observed_score_equating
   equipercentile_equating
   chain_link
   link_mirt
   transform_mirt_theta
   link_grm
   link_gpcm

Multigroup
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   fit_multigroup
   compare_invariance

Reports
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   generate_report
   ItemAnalysisReport
   ModelFitReport
   FullDiagnosticReport
   DIFAnalysisReport

Utils
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   simdata
   validate_responses
   itemstats
   missing_patterns
   MissingPatternResult
   traditional
   generate_plausible_values
   combine_plausible_values
   plausible_value_regression
   plausible_value_statistics
   draw_parameters
   posterior_summary
   sample_expected_scores
   ParameterSamples

Exceptions
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MirtError
   MirtValidationError
   MirtDataError
   MirtConvergenceError

Results
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   FitResult
   ScoreResult
   AbilityPosteriorResult
   ResponseTimeResult
