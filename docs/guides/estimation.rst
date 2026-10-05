Estimation Controls
===================

``fit_mirt`` and the estimators accept starting values, fixed parameters and
item priors, much like the ``value`` and ``est`` columns of R's ``mirt``
``pars`` table and its ``PRIOR`` syntax.

Starting values and fixed parameters
------------------------------------

``start_values`` maps stored parameter names to starting arrays; parameters
left out start from the family defaults. ``fixed`` maps parameter names to
Boolean masks, where ``True`` holds a coordinate at its starting value. A
scalar mask applies to the whole parameter:

.. code-block:: python

   import numpy as np
   import mirt

   responses = mirt.simdata("3PL", n_persons=1000, n_items=10, seed=1)
   discrimination = np.ones(10)
   discrimination[0] = 1.8
   anchor = np.zeros(10, dtype=bool)
   anchor[0] = True

   result = mirt.fit_mirt(
       responses,
       model="3PL",
       start_values={"discrimination": discrimination, "guessing": np.full(10, 0.2)},
       fixed={"discrimination": anchor, "guessing": True},
   )

Fixed coordinates do not count as parameters and have zero standard errors.
MHRM and Gibbs sampling cannot hold coordinates fixed and raise
``MirtValidationError`` for such models.

Estimators take the same controls on a model instance. Fix coordinates with
``model.set_free_parameter_masks`` and choose the starting point with
``start``: ``"default"`` resets an unfitted model's free coordinates and
warm-starts a fitted one, ``"model"`` keeps the current values, and a mapping
overrides the defaults:

.. code-block:: python

   model = mirt.TwoParameterLogistic(n_items=10)
   model.set_parameters(discrimination=np.full(10, 1.5))
   result = mirt.EMEstimator().fit(model, responses, start="model")

``EMEstimator``, ``WeightedEMEstimator``, the Monte Carlo EM estimators and
``GVEMEstimator`` accept ``start``. All of them except ``GVEMEstimator`` honor
fixed coordinates.

Bayes modal estimation with item priors
---------------------------------------

Unregularized 3PL guessing estimates often reach the edges of their range.
``priors`` adds log-prior terms for item parameters to every EM M-step, so EM
maximizes the posterior density (Bayes modal or MAP estimation):

.. code-block:: python

   from mirt.estimation.priors import BetaPrior, LogNormalPrior

   result = mirt.fit_mirt(
       responses,
       model="3PL",
       priors={
           "guessing": BetaPrior(5, 17),
           "discrimination": LogNormalPrior(0.0, 0.5),
       },
   )
   print(result.log_likelihood, result.log_posterior)

Keys name stored per-item parameters. A
:class:`~mirt.estimation.priors.PriorSpecification` applies its
discrimination, difficulty, guessing and upper priors to whichever of those
parameters the model has. The same specification is available as
``EMEstimator(item_priors=...)``. Priors apply to free coordinates only, and
convergence is judged on the log-posterior. ``log_likelihood``, AIC and BIC
stay likelihood-based, and standard errors ignore the prior curvature.

Monte Carlo EM convergence
--------------------------

``MCEMEstimator`` stops when a confidence interval for the change in marginal
log-likelihood between iterates lies within ``(-tol, tol)``. Each change is
estimated on the fresh draws of the next iteration, which are shared by both
parameter sets. When a change cannot be told apart from Monte Carlo error, the
sample grows by half, up to ``max_samples`` per person:

.. code-block:: python

   from mirt.estimation.mcem import MCEMEstimator

   estimator = MCEMEstimator(n_samples=200, max_samples=4000, tol=1e-3, seed=1)
   result = estimator.fit(mirt.TwoParameterLogistic(n_items=10), responses)
   print(result.converged, estimator.sample_size_history[-1])

If changes stay within Monte Carlo error once ``max_samples`` is reached, the
fit stops with ``converged=False`` and a ``RuntimeWarning``. Memory grows with
the sample, so lower ``max_samples`` for large data sets. ``QMCEMEstimator``
uses a fixed quasi-random grid and the plain change rule.
