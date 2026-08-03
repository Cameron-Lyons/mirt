API Reference
=============

Core Functions
--------------

.. currentmodule:: mirt

.. autosummary::
   :toctree: generated
   :nosignatures:

   fit_mirt
   fscores
   itemfit
   personfit
   dif
   load_dataset
   list_datasets

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
   BifactorModel
   MixtureIRT
   TestletModel
   DINA
   DINO

Estimation
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   EMEstimator
   MHRMEstimator
   GibbsSampler
   BLEstimator

Diagnostics
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_fit_indices
   compare_models
   anova_irt
   compute_dtf
   compute_drf
   sibtest

Scoring
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   fscores

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
   traditional

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
