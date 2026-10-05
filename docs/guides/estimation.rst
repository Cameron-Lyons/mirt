Estimation Controls
===================

``fit_mirt`` and the estimators accept starting values, fixed parameters and
item priors, much like the ``value`` and ``est`` columns of R's ``mirt``
``pars`` table and its ``PRIOR`` syntax. The same controls can be written as
model syntax (:doc:`model_syntax`), mixed-format tests name parameters by
family (:doc:`mixed_format`), and :doc:`uncertainty` describes the standard
errors that EM reports.

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
parameters the model has. A specification that reaches none of them, as for
a nominal model, raises an error, and an explicit discrimination or
difficulty prior for a parameter the model lacks, such as ``difficulty`` for
a graded model, is ignored with a warning; pass a mapping for other
parameters. The same specification is available as
``EMEstimator(item_priors=...)``. Priors apply to free coordinates only, and
convergence is judged on the log-posterior. ``log_likelihood``, AIC and BIC
stay likelihood-based. Standard errors include the prior: the negative second
derivative of each log-prior (``Prior.hess_log_pdf``) is added to the
information, so the errors describe the curvature of the log-posterior at the
mode rather than of the likelihood alone.

Accelerating EM
---------------

EM converges slowly when much information is missing, as with few items or
weakly informative responses. ``accelerate="squarem"`` extrapolates
consecutive EM steps by SQUAREM (Varadhan and Roland, 2008) and falls back to
the plain EM step whenever an extrapolation lowers the likelihood:

.. code-block:: python

   result = mirt.fit_mirt(responses, model="3PL", accelerate="squarem")

SQUAREM runs the generic EM loop, so a unidimensional 2PL fit with it skips
the native full-EM fast path; the extrapolation usually more than makes up
for that on slowly converging fits. It requires a fixed Gaussian latent
density and is also available as ``EMEstimator(accelerate="squarem")``.

Parameters shared by all items
------------------------------

The rating scale models give every item the same category thresholds
(:class:`~mirt.models.polytomous.RatingScaleModel`), and the graded rating
scale model also one common slope
(:class:`~mirt.models.polytomous.GradedRatingScaleModel`). ``EMEstimator``
and ``WeightedEMEstimator`` update these shared parameters after the itemwise
M-step by maximizing the expected complete-data log-likelihood of all items,
with the item locations held fixed. Every iteration still increases the
marginal likelihood:

.. code-block:: python

   from mirt.estimation.em import EMEstimator
   from mirt.models.polytomous import RatingScaleModel

   model = RatingScaleModel(n_items=10, n_categories=4)
   # likert: responses to 10 items coded 0, 1, 2 or 3
   result = EMEstimator().fit(model, likert)
   print(result.model.thresholds, result.standard_errors["thresholds"])

The exact observed information covers the shared coordinates, so the default
``se_method="auto"`` reports observed-information standard errors and the
full covariance in ``FitResult.vcov``. ``BLEstimator`` also estimates shared
parameters. The Monte Carlo EM estimators update items one at a time and
raise ``MirtModelError`` for a model with free shared parameters; hold them
fixed with ``model.set_free_parameter_masks`` to use those estimators.

Equality constraints across items
---------------------------------

``constraints`` holds a parameter equal across a set of items, like
``CONSTRAIN`` in R's ``mirt.model``. Each entry names one stored parameter
and its items, as a mapping or a tuple; items are zero-based positions or
item names, and omitting them ties every item:

.. code-block:: python

   responses = mirt.simdata("2PL", n_persons=1000, n_items=10, seed=1)

   # One common slope: the Rasch structure with an estimated slope.
   rasch_like = mirt.fit_mirt(
       responses, model="2PL", constraints=[{"parameter": "discrimination"}]
   )
   # Equal slopes within two blocks of items.
   blocks = mirt.fit_mirt(
       responses,
       model="2PL",
       constraints=[("discrimination", range(5)), ("discrimination", range(5, 10))],
   )

For an array parameter, ``"column"`` (or a third tuple element) ties one
coordinate of each item's row, such as one threshold or the slope on one
factor; without it whole rows are tied column by column:

.. code-block:: python

   likert = mirt.simdata("GRM", n_persons=1000, n_items=6, n_categories=4, seed=2)
   graded = mirt.fit_mirt(
       likert,
       model="GRM",
       constraints=[
           ("discrimination", [0, 1, 2]),
           {"parameter": "thresholds", "items": [3, 4], "column": 0},
       ],
   )

Tied coordinates start from the mean of their starting values and stay
exactly equal. In each M-step the items they link are optimized jointly, with
every tied group as one coordinate, so EM still increases the likelihood at
every iteration; ``accelerate="squarem"`` also extrapolates a group as one
coordinate. Each group counts as one parameter in ``n_parameters``, AIC and
BIC, so likelihood-ratio tests against the unconstrained model have the right
degrees of freedom. Standard errors come from the information of the
constrained parameters, ``J' I J`` for the 0/1 matrix ``J`` that maps each
group to its coordinates, by every ``se_method``. Tied coordinates report the
group's estimate and standard error, and ``FitResult.vcov`` repeats the
group's row and column for each of them, so the covariance is singular in
those directions.

Tied coordinates must be free, and one coordinate may belong to one group
only; parameters shared by all items, such as rating-scale thresholds, are
already common. Item priors apply to every stored coordinate, so a group's
value receives its prior once per tied item. Constraints are also available as
``EMEstimator(constraints=...)`` and accept
:class:`~mirt.estimation.constraints.EqualityConstraint` objects. The native
full-EM paths are skipped, while the batched 2PL and native polytomous M-steps
still update the untied items. Constraints are not available for mixed-format
models or for estimation methods other than EM. ``FitResult.refit_recipe``
records them with the other estimator settings, so the bootstrap utilities
refit with the constraints, and ``bootstrap_lr`` counts each group as one
parameter.

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

Bifactor models
---------------

In a bifactor model every item loads on one general factor and on one
specific factor. ``EMEstimator`` integrates such a model over the full
product grid of ``n_quadpts ** (1 + S)`` nodes for ``S`` specific factors,
which at the default 21 points is already 194,481 nodes for three specific
factors. ``mirt.bfactor``, like R's ``mirt::bfactor``, fits the model with
:class:`~mirt.BifactorEMEstimator` instead. Because the specific factors are
independent given the general factor, it integrates each specific factor
jointly with the general factor on a two-dimensional grid of
``n_quadpts ** 2`` nodes (Gibbons and Hedeker, 1992). The log-likelihood
equals the product-grid quadrature exactly, while the work grows linearly in
``S``:

.. code-block:: python

   import numpy as np
   import mirt

   rng = np.random.default_rng(1)
   specific_factors = np.repeat([0, 1, 2, 3], 5)
   theta = rng.standard_normal((2000, 5))
   logits = 1.4 * theta[:, :1] + 1.0 * theta[:, 1 + specific_factors]
   responses = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)

   result = mirt.bfactor(responses, specific_factors)
   print(result.model.general_loadings, result.se_method)

Specific-factor labels may be any non-negative integers. Standard errors
invert the observed information, which Louis's identity gives exactly from
the reduced posteriors (``se_method="oakes"``); ``"crossprod"``,
``"sandwich"`` and ``"complete_data"`` are also available. To let an item
load on the general factor only, fix its specific loading at zero:

.. code-block:: python

   general_only = np.zeros(20, dtype=bool)
   general_only[[4, 9]] = True
   result = mirt.bfactor(
       responses,
       specific_factors,
       start_values={"specific_loadings": np.where(general_only, 0.0, 0.5)},
       fixed={"specific_loadings": general_only},
   )

The reduced estimator assumes dichotomous items and independent
standard-normal factors, and does not take item priors. ``EMEstimator``
warns when a product grid would exceed a million nodes.

Metropolis-Hastings Robbins-Monro
---------------------------------

``estimation="MHRM"`` and ``MHRMEstimator`` follow Cai (2010). Each cycle
draws abilities by random-walk Metropolis and moves every item along its
complete-data score, preconditioned by a running average of its complete-data
information. The first ``burnin`` cycles use unit gains, which are Newton
steps on the imputed data; afterwards gains decrease as ``1 / (t + 1)``
(``gain_sequence="standard"``) or ``min(1, 10 / (t + 10))`` (``"adaptive"``),
and the estimates average the post-burn-in iterates. Every built-in item
family is supported except the rating scale models (RSM and GRSM), whose
thresholds are shared by all items; fit those with EM or Bock-Lieberman
estimation. Unidimensional 2PL fits use the native kernel:

.. code-block:: python

   estimator = mirt.MHRMEstimator(n_cycles=1000, burnin=250, seed=1)
   result = estimator.fit(mirt.GradedResponseModel(n_items=10, n_categories=4), data)

Standard errors of unidimensional 1PL-4PL, GRM, GPCM and PCM fits come from
the exact observed information of the marginal likelihood at the MH-RM
estimates, integrated on ``n_quadpts`` Gauss-Hermite points, as EM computes
them with ``se_method="oakes"``; ``FitResult.vcov`` holds their covariance.
For other models ``se_method="mhrm_iterate_sd"`` marks the spread of the
post-burn-in iterates, which reflects the Robbins-Monro noise rather than
sampling variability; fit with EM and ``se_method="oakes"`` for
observed-information errors.

``GibbsSampler(n_chains=k)`` stacks the draws of ``k`` chains seeded
``seed + 1000 * i``. With ``parallel_chains=True`` NumPy chains run in spawned
worker processes that use the current backend and reproduce the serial draws;
the multithreaded native 2PL kernel runs its chains in turn.
