MIRT: Multidimensional Item Response Theory
===========================================

MIRT is a high-performance Python library for Item Response Theory (IRT) analysis,
powered by a Rust backend for computational efficiency.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   guides/cat
   guides/dif
   guides/multigroup
   guides/equating
   guides/custom_models
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

* **Multiple IRT Models**: 1PL, 2PL, 3PL, Graded Response, Partial Credit, and more
* **Estimation Methods**: EM, MCMC, MHRM, Mixed estimation
* **Diagnostics**: Item fit, model fit, DIF analysis, SIBTEST
* **Scoring**: EAP, MAP, ML scoring methods
* **Computerized Adaptive Testing**: CAT algorithms and simulations
* **Custom Models**: Validated dichotomous, polytomous, multidimensional, and group callbacks
* **High Performance**: Rust-powered backend for fast computation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
