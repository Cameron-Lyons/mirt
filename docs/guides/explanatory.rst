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

Estimating Item Features
------------------------

``EMEstimator`` fits the item side of :class:`~mirt.models.LLTM`,
:class:`~mirt.models.RaschLLTM` and ``ExplanatoryIRT`` by marginal maximum
likelihood with a standard-normal ability. The feature weights, and the
common discrimination when ``constrain_discrimination=True`` (the default),
are shared by every item, so EM's shared-parameter step estimates them from
all items' expected counts:

.. code-block:: python

   from mirt.estimation.em import EMEstimator
   from mirt.models import LLTM

   lltm = LLTM(n_items=3, item_features=item_features)
   result = EMEstimator().fit(lltm, responses)
   print(lltm.feature_weights, lltm.discrimination[0], result.log_likelihood)

A common discrimination ``a`` with a standard-normal ability is R mirt's Rasch
LLTM (``itemtype="Rasch"`` with ``item.formula``), whose latent variance is
``a ** 2``. ``mirt.LLTM(q_matrix).fit(responses)`` fits the same model and
adds the likelihood-ratio test against the Rasch model. EM ignores the person
covariates of ``ExplanatoryIRT``, whose latent regression keeps its values;
:class:`mirt.MixedEffectsIRT` estimates item and person covariate effects
together. Estimators without a shared-parameter step raise
``MirtModelError`` for these models.

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
