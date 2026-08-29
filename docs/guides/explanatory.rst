Explanatory IRT Predictions
===========================

Explanatory IRT combines item features with person covariates. Item features
explain difficulty, while a latent regression explains the expected ability of
each respondent.

Model Setup
-----------

Create the model with one row of item features per item and one column per
feature:

.. code-block:: python

   import numpy as np
   from mirt.models import ExplanatoryIRT

   item_features = np.array(
       [
           [1.0, 0.0],
           [0.0, 1.0],
           [1.0, 1.0],
       ]
   )
   model = ExplanatoryIRT(
       n_items=3,
       item_features=item_features,
       n_person_covariates=2,
       feature_names=["operations", "reading"],
       covariate_names=["instruction", "prior_score"],
   )
   model.set_feature_weights(np.array([0.4, -0.2]))
   model.set_regression_weights(np.array([0.1, 0.5, 0.8]))
   model.set_residual_variance(0.7)

Conditional and Marginal Predictions
------------------------------------

``probability_given_covariates`` evaluates responses at the latent-regression
mean. Use ``marginal_probability_given_covariates`` when predictions should
include unexplained ability variation:

.. code-block:: python

   covariates = np.array(
       [
           [0.0, -1.0],
           [1.0, 0.0],
           [1.0, 1.0],
       ]
   )

   conditional = model.probability_given_covariates(covariates)
   marginal = model.marginal_probability_given_covariates(
       covariates,
       n_quadpts=21,
   )

The marginal method uses Gauss-Hermite quadrature and returns one probability
per respondent and item. Supply ``item_idx`` to evaluate a single item.

Joint Response Likelihoods
--------------------------

Use ``marginal_log_likelihood_given_covariates`` to evaluate complete response
patterns. It integrates the shared residual ability once per respondent, so the
dependence among their item responses is retained:

.. code-block:: python

   responses = np.array(
       [
           [1, 0, 1],
           [0, 1, -1],
           [1, 1, 1],
       ]
   )

   log_likelihood = model.marginal_log_likelihood_given_covariates(
       responses,
       covariates,
       n_quadpts=21,
   )

Negative response values are treated as missing. The returned array contains
one joint log likelihood per respondent.

The likelihood integral is reduced one quadrature point at a time, so temporary
memory does not grow with ``n_persons * n_quadpts``. When the compiled backend
is available, independent respondent patterns are evaluated in parallel and
only the final vector is materialized. The NumPy fallback uses the same clipped
probability and missing-response semantics and is selected automatically when
the compiled backend is unavailable or disabled with
``mirt.set_backend("numpy")``.
