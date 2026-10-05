MIRT: Multidimensional Item Response Theory
===========================================

MIRT is a high-performance Python library for Item Response Theory (IRT) analysis,
powered by a Rust backend for computational efficiency.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   guides/estimation
   guides/model_syntax
   guides/mixed_format
   guides/cat
   guides/dynamic
   guides/response_time
   guides/explanatory
   guides/multilevel
   guides/irtree
   guides/mfrm
   guides/information
   guides/dif
   guides/plotting
   guides/multigroup
   guides/equating
   guides/uncertainty
   guides/bayesian
   guides/results
   guides/reports
   guides/custom_models
   guides/network_models
   api/index

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install mirt

Quick Start
-----------

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]

   result = mirt.fit_mirt(responses, model="2PL")
   print(result.summary())

   scores = mirt.fscores(result, responses, method="EAP")
   print(scores.theta[:5])
   print(scores.standard_error[:5])

Features
--------

* **Multiple IRT Models**: 1PL-5PL, graded response, (generalized) partial credit,
  rating scale, nominal, and mixed-format tests (:doc:`guides/mixed_format`)
* **Estimation Methods**: EM with SQUAREM acceleration, bifactor dimension
  reduction, MHRM, MCMC, Monte Carlo EM, and mixed-effects estimation
* **Estimation Controls**: Starting values, fixed parameters, item priors, and
  R-style model syntax with factor correlations (:doc:`guides/estimation`,
  :doc:`guides/model_syntax`)
* **Standard Errors**: Observed-information errors and parameter covariance by
  default for unidimensional 1PL-4PL, graded, partial credit, rating scale, and
  mixed-format EM fits (:doc:`guides/uncertainty`)
* **Diagnostics**: Item fit, model fit, multiple-group DIF, SIBTEST
* **Equating**: Linking, true-score, observed-score, and kernel equating, and
  fixed-item calibration (:doc:`guides/equating`)
* **Reports**: Standalone HTML summaries with optional embedded visualizations
* **Scoring**: EAP, MAP, ML, WLE, sum-score tables, and plausible values
* **Results**: Validated inference, confidence intervals, and portable exports
* **Computerized Adaptive Testing**: CAT and MCAT, shadow tests, and simulation
  reports
* **Rater and Learning Models**: Many-facet Rasch models (:doc:`guides/mfrm`) and
  Bayesian knowledge tracing (:doc:`guides/dynamic`)
* **Custom Models**: Validated dichotomous, polytomous, multidimensional, and group callbacks
* **Multilevel Models**: Integrated two- and three-level response-pattern likelihoods
* **High Performance**: Rust-powered backend for fast computation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
