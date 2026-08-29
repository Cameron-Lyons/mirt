IRTree Models
=============

Item-response trees decompose an ordinal response into a sequence of binary
decisions. This makes it possible to estimate substantive and response-style
traits separately while retaining the original response categories.

Fitting a built-in tree
-----------------------

The built-in specifications are ``"bockenholt"``, ``"extreme_midpoint"``,
and ``"direction_intensity"``. Each specification defines its own decision
nodes and latent traits.

.. code-block:: python

   import numpy as np

   from mirt import IRTreeEMEstimator, IRTreeModel

   responses = np.array(
       [
           [0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 2, 1, 0],
           [4, 3, 2, 1],
       ]
   )

   model = IRTreeModel(
       n_items=responses.shape[1],
       tree_spec="bockenholt",
       n_categories=5,
   )
   estimator = IRTreeEMEstimator(n_quadpts=7, max_iter=300, tol=1e-4)
   result = estimator.fit(model, responses)

   print(result.summary())
   print(result.trait_summary())

The number of multidimensional quadrature points is
``n_quadpts ** model.n_traits``. Smaller values such as 5 or 7 are useful for
initial model development; increase the value when checking the stability of
the final estimates.

Inspecting estimates and uncertainty
------------------------------------

The fitted result keeps item estimates on ``result.model`` and exposes the
estimated latent distribution and person scores directly:

.. code-block:: python

   item_parameters = result.model.parameters
   discrimination = item_parameters["discrimination"]
   difficulty = item_parameters["difficulty"]

   trait_means = result.trait_means
   trait_covariance = result.trait_covariance
   trait_correlations = result.trait_correlations

   person_eap = result.theta_estimates
   person_standard_error = result.theta_se
   item_standard_error = result.standard_errors

Item standard errors use the expected complete-data information matrix. They
are useful as local uncertainty approximations; bootstrap inference remains a
better choice when sample sizes are small or estimates approach parameter
bounds. An entry is ``NaN`` when a decision node has no observed responses or
its local information matrix is singular.

Missing responses
-----------------

``NaN`` and negative response values are treated as missing. Observed values
must be integer category codes between zero and ``n_categories - 1``. Invalid
fractional or out-of-range values raise a data-validation error instead of
being silently converted.

Simulating responses
--------------------

Use :meth:`mirt.IRTreeModel.simulate` to draw ordinal responses conditional
on person trait values. The method evaluates every item and category in
vectorized batches, selects a memory-bounded batch size automatically, and
returns a person-by-item matrix of integer category codes:

.. code-block:: python

   rng = np.random.default_rng(42)
   theta = rng.multivariate_normal(
       result.trait_means,
       result.trait_covariance,
       size=10_000,
   )
   replicated_responses = result.model.simulate(theta, seed=42)

Pass ``chunk_size`` to cap the number of persons evaluated at once. A fixed
``seed`` yields the same responses regardless of the chosen chunk size, so
memory tuning does not change a reproducible simulation.

Fixing the trait distribution
-----------------------------

Set ``estimate_correlations=False`` on the estimator to hold the latent mean
at zero and covariance at the identity matrix. A model created with
``correlated_traits=False`` also keeps that fixed distribution and does not
store a trait-correlation matrix:

.. code-block:: python

   model = IRTreeModel(
       n_items=responses.shape[1],
       tree_spec="direction_intensity",
       correlated_traits=False,
   )
   result = IRTreeEMEstimator(n_quadpts=7).fit(model, responses)

Custom trees
------------

Use :class:`mirt.IRTreeSpec` and :class:`mirt.TreeNode` to describe a custom
binary decision tree. The model validates that every category is reachable,
node traits are in range, and the structure contains neither cycles nor shared
subtrees before estimation begins.
