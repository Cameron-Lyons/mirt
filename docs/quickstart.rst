Quick Start Guide
=================

This guide will walk you through the basics of using MIRT for IRT analysis.

Basic Usage
-----------

1. **Prepare your data**: Response data should be a 2D NumPy array where rows
   are respondents and columns are items.

2. **Choose a model**: Select an IRT model appropriate for your data.

3. **Estimate parameters**: Use an estimation method to fit the model.

4. **Analyze results**: Examine item and person parameters, fit statistics, etc.

Example: 2PL Model
------------------

.. code-block:: python

   import numpy as np
   from mirt import Model2PL, EMEstimator

   # Simulated response data: 500 respondents, 20 items
   np.random.seed(42)
   responses = np.random.randint(0, 2, size=(500, 20))

   # Create model
   model = Model2PL(n_items=20, n_factors=1)

   # Fit with EM algorithm
   estimator = EMEstimator(max_iter=100, tol=1e-4)
   results = estimator.fit(model, responses)

   # Access parameters
   print("Discrimination parameters:", results.item_parameters["a"])
   print("Difficulty parameters:", results.item_parameters["b"])

Example: Graded Response Model
------------------------------

For polytomous items (Likert scales, etc.):

.. code-block:: python

   from mirt import GradedResponseModel, EMEstimator

   # Responses with 5 categories (0-4)
   responses = np.random.randint(0, 5, size=(500, 15))

   model = GradedResponseModel(n_items=15, n_categories=5)
   results = EMEstimator().fit(model, responses)

Scoring
-------

After fitting a model, you can score new respondents:

.. code-block:: python

   from mirt.scoring import EAPScorer

   scorer = EAPScorer()
   theta_estimates = scorer.score(results, new_responses)

Model Fit
---------

Evaluate model fit with various diagnostics:

.. code-block:: python

   from mirt.diagnostics import ItemFit

   fit = ItemFit(results)
   print(fit.infit)
   print(fit.outfit)
