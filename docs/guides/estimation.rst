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
``MHRMEstimator`` and ``GibbsSampler`` hold them too; the native MHRM burn-in
ignores masks, so masked models run the NumPy MHRM cycles throughout.

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
fixed coordinates. ``MHRMEstimator`` and ``GibbsSampler`` start from the
model's current values.

Every estimator checks the response codes against the model as ``fit_mirt``
does: a dichotomous model accepts only 0, 1 and missing codes, and a
polytomous model codes below each item's category count. Other codes raise
``MirtDataError``, as do ``bootstrap_se``, ``bootstrap_ci`` and
``bootstrap_lr``.

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

The fused native 3PL iteration applies ``NormalPrior``, ``LogNormalPrior``
and ``BetaPrior`` priors on 3PL items, and its M-steps reach the same
posterior mode as the NumPy steps. The native polytomous M-steps cannot apply
priors, so a fit with priors solves its other built-in items in NumPy batches
instead: Newton steps for 1PL, 2PL and MIRT items, whose difficulty priors
enter through the chain rule of ``b = -d / sum(a)``, and Fisher scoring for
unidimensional 3PL items under other priors, graded, generalized partial
credit and partial credit items. Nominal items,
which have no native M-step, take a batched Newton step with or without
priors. The 1PL and PCM common slope keeps its conditional step, with its
prior counted once per item. These steps reach the posterior mode of the
itemwise optimizer at about the cost of a fit without priors. Items whose
mode lies on an optimizer bound, such as a guessing parameter that a flat
prior lets reach zero, keep the bounded itemwise optimizer, as do 3PL and
polytomous models with fixed coordinates.

Accelerating EM
---------------

EM converges slowly when much information is missing, as with few items or
weakly informative responses, and above all along the scale of the latent
trait: with the trait held standard normal, plain EM moves all slopes and
locations toward their common scale by a small step per iteration. By
default ``EMEstimator`` therefore takes parameter-expanded steps (PX-EM; Liu,
Rubin and Wu, 1998). After each M-step, the mean and standard deviation of
the latent trait that the E-step posterior implies on the quadrature grid (in
several dimensions, the Cholesky factor of its covariance) are absorbed into
the item slopes and locations, so the trait stays standard normal and every
item keeps its family. Default fits of 5000 persons and 30 items then take
7 iterations instead of 60 for a 2PL, 7 instead of 74 for a GRM and 9 instead
of 141 for a GPCM, and reach the same optimum, with standard errors that
agree to 0.1%. A step whose next E-step lowers the log-likelihood is undone,
and the fit continues with plain EM, as it also does once the log-likelihood
changes by less than ``100 * tol``, so the log-likelihood never decreases and
the fit ends at the fixed point of plain EM on its grid.

The steps apply to the 1PL and PCM with their estimated common slope, the
2PL, GRM, GPCM and nominal models and the slope-intercept
``MultidimensionalModel``, with the default standard normal latent density,
slopes that are free or fixed at zero by a loading pattern, and no other
fixed coordinates, item priors (which the rescaling would change) or user
equality constraints. They also need a rectangular grid with nodes at most
0.4 apart, as the default grids for one and two factors are: on coarser and
Gauss-Hermite grids, the EM fixed point of the grid drifts from the standard
normal scale that the steps restore. Other fits run plain EM, and
``EMEstimator(px_em=False)`` always does:

.. code-block:: python

   from mirt.estimation.em import EMEstimator

   plain = EMEstimator(px_em=False).fit(model, responses)

``accelerate="squarem"`` instead extrapolates consecutive plain EM steps by
SQUAREM (Varadhan and Roland, 2008) and falls back to the plain EM step
whenever an extrapolation lowers the likelihood. It helps most where the
expansion steps do not apply, as for the 3PL, whose fused native EM
iteration it extrapolates:

.. code-block:: python

   result = mirt.fit_mirt(responses, model="3PL", accelerate="squarem")

SQUAREM takes no expansion steps. It requires a fixed Gaussian latent
density and is also available as ``EMEstimator(accelerate="squarem")``.

Estimated latent densities
--------------------------

``EMEstimator(latent_density=...)`` estimates the shape of a univariate
latent distribution together with the items: ``"ehw"`` is the empirical
histogram of Woods (2007), ``"davidian"`` a Davidian curve of degree 4 (Woods
and Lin, 2009) and ``"mixture"`` a mixture of two normals. With free item
parameters the location and scale of such a density are not identified, so
each EM cycle holds it at mean 0 and variance 1, as R's ``mirt`` does for
``dentype="EHW"`` and ``"Davidian-k"``. The item parameters therefore stay on
the scale of a standard-normal fit, and ``FitResult.latent_mean`` and
``latent_covariance`` are ``None``:

.. code-block:: python

   from mirt.estimation.em import EMEstimator
   from mirt.estimation.latent_density import DavidianCurve

   density = DavidianCurve(degree=4)
   result = EMEstimator(latent_density=density).fit(model, responses)
   print(density.mean, density.variance)  # 0.0 and 1.0

The histogram is rescaled by Woods' interpolation-extrapolation, the Davidian
curve is fitted under the two moment constraints, and the mixture components
are shifted and scaled together. Every update fits the posterior counts at
least as well as the current density, so the log-likelihood never decreases.
The constraints remove two parameters from ``n_parameters``. Empirical
histograms converge slowly when perfect or zero scores pull mass toward the
outermost nodes and may need a larger ``max_iter``; R's ``mirt`` stops such
fits earlier because it judges convergence on parameter changes.
``DavidianCurve(standardize=False)`` and ``MixtureDensity(standardize=False)``
leave the location and scale free, for item parameters held fixed, and then
report the density's mean and variance as ``latent_mean`` and
``latent_covariance``.

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

   # One common slope: the Rasch structure with an estimated slope, which
   # is also what fit_mirt(responses, model="1PL") fits.
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
degrees of freedom. Given the ``FitResult``, :func:`mirt.itemfit`,
``compute_m2`` and ``compute_fit_indices`` count each group once in their
degrees of freedom too. Standard errors come from the information of the
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

Rasch models and the common slope
---------------------------------

``fit_mirt(model="1PL")`` and ``model="PCM"`` estimate one discrimination
common to every item, through the same tie as above, so a 1PL with ``J``
items has ``J + 1`` parameters. The latent trait stays standard normal. R
mirt's ``itemtype="Rasch"`` instead fixes the slopes at 1 and estimates the
latent variance; the two models have the same likelihood, the common slope
is R's latent standard deviation, and R's difficulties are the common slope
times these, so likelihood-ratio tests against a 2PL agree with R's. The
classes take the choice as ``OneParameterLogistic(...,
estimate_discrimination=True)`` and ``PartialCreditModel(...,
estimate_discrimination=True)``; by default they hold the slope at 1, which
``fixed={"discrimination": True}`` also requests from ``fit_mirt``.
:class:`~mirt.EMEstimator` estimates the common slope, and estimators
without equality constraints raise ``MirtModelError`` for it, and
``fit_mirt`` accepts "1PL" and "PCM" only with ``estimation="EM"``. ``spec``
fits and mixed-format tests hold the slope at 1; with ``spec``,
``COV = F*F`` estimates the latent variance as R does.

Quadrature and parameter boxes
------------------------------

EM integrates over the latent trait on a product grid of ``n_quadpts ** F``
nodes for ``F`` factors. By default ``n_quadpts`` depends on ``F`` as in R
mirt: 61 points for one factor, 31 for two, 15 for three, 9 for four, 7 for
five and 5 for six. Seven or more factors need an explicit ``n_quadpts``, or
``estimation="MHRM"`` or :class:`~mirt.estimation.mcem.QMCEMEstimator`, which
integrate by sampling. Grids of over a million nodes, or whose E-step arrays
would need over a gigabyte, warn before they are built.

``quadrature="auto"`` (the default) places grids of 15 or more points per
dimension on R mirt's rectangular rule, equally spaced nodes on ``[-6, 6]``
weighted by the normal density, and coarser grids on Gauss-Hermite nodes.
Long or highly discriminating tests have sharply peaked person likelihoods,
which the even grid resolves with far fewer nodes: at the generating
parameters of a 40-item graded test with slopes between 2 and 4, 61
rectangular nodes integrate the marginal log-likelihood to within 0.2,
whereas 61 Gauss-Hermite nodes miss it by about 130 and the former default of
21 Gauss-Hermite nodes by about 1500, which shrank the fitted slopes to 0.72
of their true values. With a few points per dimension Gauss-Hermite nodes
are the more accurate rule. ``quadrature="rectangular"`` or
``"gauss-hermite"`` selects a rule explicitly.

The item M-steps keep parameters in boxes. Logistic locations lie in
``[-30, 30]``, slopes of the 1PL-4PL families in ``[0.1, 20]`` (3PL and 4PL
slopes in ``[0.1, 5]``), and the intercepts and slopes of multidimensional
logistic items in ``[-30, 30]`` and ``[-20, 20]``, so unrotated exploratory
loadings may be negative. Unidimensional slopes stay positive, as linking,
DIF and other consumers of a fitted model assume. A reverse-keyed item then
ends on the lower bound, where R mirt would report a negative slope;
``fit_mirt`` warns and the item should be recoded. Graded and partial-credit
items keep slopes in ``[0.1, 5]`` and thresholds or steps in ``[-6, 6]``,
the box of the native polytomous M-step.

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
the sample, so lower ``max_samples`` for large data sets. Every person has
draws of their own. With ``importance_sampling=False`` they come from
Metropolis chains that continue from one iteration's draws to the next, so
that they track the posteriors as the parameters move; chains restarted at the
prior in every iteration used to leave the slopes of a 300-person test about
0.18 from EM's.

``QMCEMEstimator`` uses one scrambled quasi-random grid for every iteration of
a fit, drawn from ``seed`` or, without one, once per fit, and the plain change
rule. A grid scrambled afresh in every iteration kept the log-likelihood
estimate moving by more than ``tol``, so unseeded fits never converged.

``StochasticEMEstimator`` imputes ``n_chains`` posterior draws per person and
maximizes their complete-data likelihood, so its iterates fluctuate around the
maximum likelihood estimate instead of converging to it. It returns the
average of the iterates after ``burnin`` iterations (100 by default) and
converges once the batch-means Monte Carlo standard error of every averaged
free parameter is below ``tol`` (0.01 by default):

.. code-block:: python

   from mirt.estimation.mcem import StochasticEMEstimator

   result = StochasticEMEstimator(seed=1).fit(
       mirt.TwoParameterLogistic(n_items=20), responses
   )

On a 2PL of 1000 persons and 20 items this takes about 150 iterations, and the
slopes are within 0.01 of EM's. Reaching ``max_iter`` first, or ending within
the burn-in, gives ``converged=False`` and a ``RuntimeWarning``.

Graded response items of all three estimators keep their thresholds ordered
in the M-step, as in EM. Without that, a middle category used by 1% of the
persons let its thresholds cross to ``[6, -6]``, with category probabilities
near -1.

The log-likelihood, AIC and BIC of these fits are the marginal log-likelihood
at the final estimates, integrated over the ability prior on EM's default
quadrature grid for up to six factors and on 4,096 scrambled Sobol points
beyond, so they can be compared with an EM fit; ``convergence_history`` keeps
the Monte Carlo estimates of the iterations. Stochastic EM and posterior-draw
MCEM formerly reported harmonic-mean estimates, such as -13853 against a
marginal log-likelihood of -9164. With ``compute_standard_errors=True``,
unidimensional 1PL-4PL, GRM, GPCM and PCM fits report the observed
information at the estimates, as EM does (see :doc:`results`).

Regularized exploratory MIRT
----------------------------

:class:`~mirt.estimation.regularized.RegularizedMIRTEstimator` fits an
exploratory multidimensional 2PL whose loadings carry a LASSO, ridge or
elastic-net penalty, so that a sparse loading pattern emerges without a
rotation. The penalized likelihood is not concave. Zero loadings are a
stationary point at every penalty, because the posterior then equals the
prior, and equal starting loadings on every factor stay equal, which traps a
fit in one factor duplicated. Every fit therefore starts from an unpenalized
EM solution rotated to varimax simple structure; on a two-factor test of 16
items this start reaches a penalized log-likelihood 388 higher at
``lambda=20`` than the former equal start, with every item on its own factor.
``fit(..., start=previous_result)`` warm-starts from another fit:

.. code-block:: python

   from mirt.estimation.regularized import RegularizedMIRTEstimator

   estimator = RegularizedMIRTEstimator(n_factors=2, penalty="lasso")
   path = estimator.fit_path(responses, n_lambda=20)
   for result in path:
       print(result.lambda_val, result.n_nonzero, result.bic, result.ebic)

``fit_path`` spaces its penalties geometrically from ``lambda_max`` down to
``lambda_max * lambda_min_ratio``. ``lambda_max`` is the smallest L1 penalty
at which the first M-step from the rotated unpenalized solution sets every
loading to zero, ``max |sum_i (y_ij - p_j) E[theta_ik | y_i]|`` over items and
factors, divided by the adaptive weight and the elastic-net ``alpha``. A
converged fit can still fall back to zero loadings below it, so the largest
penalties of a path may fit the null model; on the test above that happens
down to about a third of ``lambda_max``. The penalties are fitted from the
smallest up, each from the previous solution, because a factor whose
loadings are all zero can never regain them. ``EBIC`` is BIC plus
``log C(n_items * n_factors, n_nonzero)`` (Chen and Chen, 2008), so unlike
BIC it rewards sparsity among the many possible loading patterns.
:func:`mirt.information_criteria_path` and :func:`mirt.cv_select_lambda`
select a penalty on the same default grid; cross-validation fits each fold's
path the same way on its training persons.

Variational and sparse Bayesian MIRT
------------------------------------

:class:`~mirt.estimation.gvem.GVEMEstimator` (Cho et al., 2021) replaces the
E-step by Gaussian variational posteriors and a quadratic bound on the
logistic function, so each iteration is closed-form at any number of
factors. It maximizes the evidence lower bound rather than the likelihood,
and its slopes are biased toward zero: on a simulated 2PL of 1000 persons
and 20 items the GVEM slopes were 0.79 to 1.01 times the EM estimates and
the difficulties of the most extreme items moved by up to 0.34, and on a
two-factor test of 16 items the communalities ``diag(A A')`` averaged 0.73 of
EM's. Tighter tolerances do not change this. Use GVEM for fast approximate
fits of many factors, and EM, ``QMCEMEstimator`` or MH-RM when unbiased
estimates matter.

``GVEMResult.log_likelihood``, AIC and BIC use the marginal log-likelihood at
the GVEM estimates, which is comparable with an EM fit, and
``GVEMResult.elbo`` keeps the lower bound; on the test above they are -9175
and -9342. Up to three factors are integrated on EM's default grid, more by
importance sampling from the variational posteriors. A unidimensional fit
reports the observed information of the marginal likelihood at its estimates
(``se_method="oakes"``, with ``vcov``). A multidimensional fit reports the
curvature of the bound with the variational posteriors held fixed
(``se_method="variational_curvature"``), which ignores the latent
uncertainty and understates the errors severalfold; fit with
``EMEstimator`` for usable multidimensional uncertainty. A fixed
``prior_mean`` or ``prior_cov`` is recorded as the result's latent
population.

:class:`~mirt.estimation.sparse_bayesian.SparseBayesianEstimator` places a
spike-and-slab LASSO prior on the loadings of the same variational model.
``SparseBayesianResult.log_likelihood`` is the marginal log-likelihood at the
sparse loadings, and its BIC and EBIC (BIC plus
``log C(n_items * k_max, n_selected)``) compare numbers of factors on that
scale. ``result.model`` is a slope-intercept
:class:`~mirt.models.multidimensional.MultidimensionalModel`, so an item whose
loadings are all zero keeps its intercept; the 2PL passed in receives the
loadings and the difficulties ``-d / sum(a)``, which cannot represent such an
item.

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

Graded (Likert) items are fitted the same way, as R's
``bfactor(data, model, itemtype = "graded")`` fits them (Gibbons et al.,
2007). ``model="graded"`` (or ``"GRM"``) builds a
:class:`~mirt.models.GradedBifactorModel` whose item ``j`` has cumulative
curves ``logistic(a_g theta_0 + a_s theta_s + d_jc)`` with R's decreasing
intercepts ``d_j1 > d_j2 > ...``, stored as ``intercepts[j, c - 1]``. Each
item has ``max(response) + 1`` categories; two-category items are 2PL
items, so dichotomous and Likert items can be mixed. By default ``bfactor``
fits dichotomous data as 2PL and data with a code above 1 as graded, as R
does:

.. code-block:: python

   from mirt.models import GradedBifactorModel

   graded = GradedBifactorModel(20, specific_factors, n_categories=5)
   graded.set_parameters(
       general_loadings=np.full(20, 1.4), specific_loadings=np.ones(20)
   )
   likert = graded.simulate(rng.standard_normal((2000, 5)), seed=2)
   result = mirt.bfactor(likert, specific_factors, model="graded")
   print(result.coef())  # general_loadings, specific_loadings, intercepts_1-4

The E-step accumulates expected category counts on each (general,
specific) grid, and each M-step solves the items by Newton's method, with
every item's intercepts kept decreasing. Standard errors again invert the
exact observed information of the reduced posteriors. ``fscores`` integrates
EAP scores by the same dimension reduction, and ``FitResult.to_dict()``
records the model as ``"GradedBifactor"`` with its specific factors and
category counts. ``EMEstimator`` fits a ``GradedBifactorModel`` with
:class:`~mirt.BifactorEMEstimator` too, unless a latent density, item prior,
equality constraint, acceleration or rectangular quadrature needs the
product grid, whose M-steps also keep the intercepts decreasing.

R's ``bfactor`` integrates over 31 equally spaced points on ``[-6, 6]`` per
dimension; ``quadrature="rectangular", n_quadpts=31`` uses the same grid and
reproduces its log-likelihood, estimates and Oakes standard errors. On a
simulated 12-item, five-category, three-specific-factor data set
(``N = 1000``), R mirt 1.48 and ``bfactor`` both reach a log-likelihood of
-15499.03795, with estimates within ``1.4e-5`` and standard errors within a
relative ``3.1e-5``.

The reduced estimator assumes independent standard-normal factors and does
not take item priors. ``EMEstimator`` warns when a product grid would exceed
a million nodes.

``result.model.get_factor_loadings()`` converts the logistic slopes to the
standardized loadings of R's ``summary()``: each slope is divided by
``D = 1.702`` and by the latent response standard deviation
``sqrt(1 + (a_g / D)**2 + (a_s / D)**2)``, and ``communalities()`` gives
``h2``. ``omega_hierarchical()``, ``omega_total()``, ``omega_subscale(label)``
and ``explained_common_variance()`` are computed from these loadings, with
the uniquenesses ``1 - h2`` as error variances (McDonald's omega).

Testlet models
--------------

A testlet model,
``logistic(a_j * theta + d_j * gamma_t - b_j)`` with testlet effects
``gamma_t ~ N(0, sigma_t^2)``, is a bifactor model with one specific factor
per testlet and specific slopes ``d_j * sigma_t``. ``EMEstimator`` therefore
fits :class:`~mirt.models.TestletModel`,
:class:`~mirt.models.BifactorTestletModel` and
:class:`~mirt.models.RandomTestletEffectsModel` with the dimension reduction
of :class:`~mirt.BifactorEMEstimator`, which also takes them directly, on one
``n_quadpts ** 2`` grid per testlet (21 points by default). It falls back to
the full product grid only when a latent density, item prior, equality
constraint, acceleration or rectangular quadrature needs it:

.. code-block:: python

   from mirt.estimation.em import EMEstimator
   from mirt.models import RandomTestletEffectsModel, create_testlet_structure

   membership = create_testlet_structure(20, [5, 5, 5, 5])
   result = EMEstimator().fit(RandomTestletEffectsModel(20, membership), responses)

Only the products ``d_j * sigma_t`` are identified. A testlet's variance is
therefore held fixed, at 1 unless set otherwise, while any of its loadings is
free, as ``mirt::bfactor`` fixes specific-factor variances, and it counts
neither in ``n_parameters`` nor in the standard errors. Fixing every loading
of a testlet frees its variance; with unit loadings this is the random
testlet-effect model of Wang and Wilson (2005):

.. code-block:: python

   import numpy as np

   model = RandomTestletEffectsModel(20, membership)
   model.set_parameters(testlet_loadings=(membership >= 0).astype(float))
   model.set_free_parameter_masks({"testlet_loadings": np.zeros(20, dtype=bool)})
   result = EMEstimator().fit(model, responses)
   print(result.model.testlet_variances, result.standard_errors["testlet_variances"])

``BifactorTestletModel(constrain_testlet_loadings=True)`` estimates one common
loading per testlet, which fits the same curves as unit loadings with a free
variance. The common loadings and free variances are shared by their
testlets' items: each M-step updates the items with them held, then each
testlet's common value from all its items' expected counts. Full-factor
``probability`` takes the standardized effects ``z_t = gamma_t / sigma_t``,
and ``omega_hierarchical()``, ``explained_variance()`` and
``testlet_reliability()`` use the standardized loadings of the equivalent
bifactor model, as ``BifactorModel`` does.

Metropolis-Hastings Robbins-Monro
---------------------------------

``estimation="MHRM"`` and ``MHRMEstimator`` follow Cai (2010). Each cycle
draws abilities by random-walk Metropolis and moves every item along its
complete-data score, preconditioned by a running average of its complete-data
information. The first ``burnin`` cycles use unit gains, which are Newton
steps on the imputed data; afterwards gains decrease as ``1 / (t + 1)``
(``gain_sequence="standard"``) or ``min(1, 10 / (t + 10))`` (``"adaptive"``),
and the estimates are the final iterate, which under decreasing gains is
closer to the maximum-likelihood estimates than the average of the
post-burn-in iterates. Every built-in item
family is supported except the rating scale models (RSM and GRSM), whose
thresholds are shared by all items; fit those with EM or Bock-Lieberman
estimation, or fix the shared thresholds.

As in R mirt, the fit stops once no free coordinate has changed by ``tol``
(default 0.001) or more in three consecutive post-burn-in cycles, and reports
``converged=True`` with the number of cycles run as ``n_iterations``. A fit
that reaches ``n_cycles`` first reports ``converged=False`` and warns. A
smaller ``tol`` runs longer under smaller gains, which lowers the Monte Carlo
error of the estimates: on a 1000-person, 20-item 2PL the default stops after
about 650 cycles with slopes within about 0.05 of the EM estimates, close to
R mirt's MH-RM.
``n_chains=k`` imputes ``k`` independent ability chains in every cycle and
averages their complete-data scores and information, which lowers the noise
of each step. The native kernel runs the burn-in cycles of unidimensional 2PL
fits with one chain; the decreasing-gain cycles run in NumPy:

.. code-block:: python

   estimator = mirt.MHRMEstimator(n_cycles=1000, burnin=250, seed=1)
   result = estimator.fit(mirt.GradedResponseModel(n_items=10, n_categories=4), data)

Coordinates fixed with ``set_free_parameter_masks`` keep their values, so
confirmatory loading patterns, Cai's (2010) main application, can be fitted;
the factor covariance is not estimated. ``fit(model, data,
prior_mean=..., prior_cov=...)`` fixes another normal ability density, as
:meth:`~mirt.estimation.em.EMEstimator.fit` does, and records it as the
result's ``latent_mean`` and ``latent_covariance``.

``FitResult.log_likelihood``, and the AIC and BIC built from it, is the
marginal log-likelihood at the MH-RM estimates. Up to three factors it is
integrated on EM's default grid, so it equals EM's log-likelihood at the same
parameters and likelihood-ratio tests against EM fits are valid; beyond three
it is estimated by importance sampling from normal approximations of each
person's ability posterior.

Standard errors of unidimensional 1PL-4PL, GRM, GPCM and PCM fits come from
the exact observed information of the marginal likelihood at the MH-RM
estimates, integrated on ``n_quadpts`` Gauss-Hermite points, as EM computes
them with ``se_method="oakes"``; ``FitResult.vcov`` holds their covariance.
For other models ``se_method="mhrm_iterate_sd"`` marks the spread of the
post-burn-in iterates, which reflects the Robbins-Monro noise rather than
sampling variability; fit with EM and ``se_method="oakes"`` for
observed-information errors.

``GibbsSampler(n_chains=k)`` stacks the draws of ``k`` chains seeded
``seed + 1000 * i``; NumPy chains start from jittered values so that R-hat can
compare them (see :doc:`bayesian`). With ``parallel_chains=True`` the chains
run in spawned worker processes that use the current backend and reproduce the
serial draws.
