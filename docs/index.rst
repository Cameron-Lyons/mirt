MIRT: Multidimensional Item Response Theory
===========================================

MIRT is a high-performance Python library for Item Response Theory (IRT) analysis,
powered by a Rust backend for computational efficiency.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install mirt

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from mirt import Model2PL, EMEstimator

   # Load your response data (items x respondents)
   responses = np.random.randint(0, 2, size=(100, 20))

   # Create a 2PL model
   model = Model2PL(n_items=20, n_factors=1)

   # Fit the model using EM estimation
   estimator = EMEstimator()
   results = estimator.fit(model, responses)

   # View results
   print(results.item_parameters)
   print(results.person_parameters)

Features
--------

* **Multiple IRT Models**: 1PL, 2PL, 3PL, Graded Response, Partial Credit, and more
* **Estimation Methods**: EM, MCMC, MHRM, Mixed estimation
* **Diagnostics**: Item fit, model fit, DIF analysis, SIBTEST
* **Scoring**: EAP, MAP, ML scoring methods
* **Computerized Adaptive Testing**: CAT algorithms and simulations
* **High Performance**: Rust-powered backend for fast computation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
