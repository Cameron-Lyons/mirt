Confirmatory Models and Model Syntax
====================================

:func:`mirt.mirt_model` reads a subset of the model syntax of R's
``mirt.model``. The resulting :class:`mirt.ModelSpec` assigns items to
factors, frees factor correlations and sets starting values, fixed
parameters, priors and equality constraints. ``fit_mirt(data, spec=...)``
then fits the confirmatory model and estimates the factor correlations.

Fitting correlated factors
--------------------------

.. code-block:: python

   import numpy as np
   import mirt

   rng = np.random.default_rng(0)
   loadings = np.zeros((10, 2))
   loadings[:5, 0] = rng.uniform(1, 2, 5)
   loadings[5:, 1] = rng.uniform(1, 2, 5)
   theta = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 2000)
   logits = theta @ loadings.T + rng.normal(size=10)
   responses = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)

   spec = mirt.mirt_model(
       """
       F1 = 1-5
       F2 = 6-10
       COV = F1*F2
       """
   )
   result = mirt.fit_mirt(responses, model="2PL", spec=spec, n_quadpts=15)
   print(result.latent_covariance)       # estimated factor correlation near 0.5
   print(result.model.parameters["slopes"])

   scores = mirt.fscores(result, responses)  # prior covariance from the fit

``spec`` also accepts the syntax string itself, which is then parsed against
the item names of the data. Items load only on the factors listed for them;
the other slopes are held at zero. A "2PL" model with several factors is
fitted as a slope-intercept :class:`mirt.MultidimensionalModel` (``slopes``
and ``intercepts``, as in R's ``a1``, ``a2`` and ``d``). "GRM" and "GPCM"
models keep their parameterization and hold the unused discriminations at
zero. With one factor every family can be used, so the syntax also works for
unidimensional models with fixed parameters or priors.

Syntax
------

Each statement has the form ``NAME = value``. ``#`` starts a comment, blank
lines and extra whitespace are ignored, and a value that ends with a comma or
an open parenthesis continues on the next line. Items are numbered from one,
ranges use ``-`` or ``:``, and with ``item_names`` items can be named, for
example ``Q1-Q5``. Numbers always refer to item positions, even when items
are named by numbers.

=====================================  =============================================
Statement                              Meaning
=====================================  =============================================
``F1 = 1-5, 7``                        Factor ``F1`` with items 1 to 5 and 7. Any
                                       name other than a keyword is a factor.
``COV = F1*F2, F2*F3``                 Free correlations; ``F1*F2*F3`` frees every
                                       pair. Unlisted pairs are uncorrelated.
``COV = F1*F1``                        Free variance of ``F1`` (see below).
``FIXED = (1, a1), (2-3, d)``          Hold parameters at their starting values.
``START = (1-5, a1, 1.5)``             Starting values.
``PRIOR = (1-10, d, norm, 0, 2)``      Priors: ``norm`` (mean, sd), ``lnorm``
                                       (log mean, log sd) or ``beta`` (shapes).
``CONSTRAIN = (1-3, a1), (4-6, d)``    Equality constraints: one parameter held
                                       equal across the listed items.
=====================================  =============================================

Parameter names are resolved against the fitted model: ``a`` selects all
slopes and ``a1``, ``a2``, ... the slopes on one factor, ``d`` the intercepts
of a slope-intercept model, ``g`` guessing and ``u`` upper asymptotes. Stored
parameter names such as ``difficulty`` or ``thresholds`` select whole item
rows. Errors name the offending line, for example
``line 3: unknown factor 'F4' in COV``; the line number is also available as
``error.context["line"]``.

``FIXED``, ``START`` and ``PRIOR`` act like the ``fixed``, ``start_values``
and ``priors`` arguments of :func:`mirt.fit_mirt`, and can be combined with
them: masks are joined, and ``START`` values override the matching
coordinates of ``start_values``. Item priors apply to every free coordinate of
a stored parameter, so all ``PRIOR`` groups for one parameter must use the
same distribution and together cover its free coordinates; priors come either
from the syntax or from ``priors=``.

Equality constraints
--------------------

``CONSTRAIN`` holds one parameter equal across the items of each group and
is fitted through the ``constraints`` argument of :func:`mirt.fit_mirt` (see
:doc:`estimation`). ``(1-5, a1)`` gives items 1 to 5 one slope on ``F1``;
``a`` ties the slopes on every factor, factor by factor; a stored name such
as ``thresholds`` ties whole item rows column by column. Coordinates that the
loading pattern or the family fixes, such as unused slopes and category
padding, are left out:

.. code-block:: python

   result = mirt.fit_mirt(
       responses,
       model="2PL",
       spec="F1 = 1-5\nF2 = 6-10\nCOV = F1*F2\nCONSTRAIN = (1-5, a1), (6-10, a2)",
   )

Each group counts as one parameter, tied coordinates share their estimate
and standard error, and the groups combine with ``constraints=``. A group
must tie free coordinates of at least two items, so ``FIXED`` and
``CONSTRAIN`` cannot name the same coordinate. R's form that equates
different parameters, such as ``CONSTRAIN = (1, 3, a1, a2)``, raises
``NotImplementedError``.

Factor correlations and variances
---------------------------------

Factor variances are fixed at one, so the latent covariance is a correlation
matrix. After each item M-step, EM maximizes the expected complete-data
likelihood over the free correlations with the item parameters held fixed
(an ECM step), so every iteration increases the likelihood, or the posterior
when priors are used. The estimate is stored in ``FitResult.latent_covariance``
(with ``factor_correlation`` as its correlation matrix), counted in
``n_parameters`` and printed by ``summary()``. Given the ``FitResult``, these
functions use it as the latent population: :func:`mirt.fscores`,
:func:`mirt.ability_posterior`, the EAPsum functions, plausible values,
imputation and :func:`mirt.simdata`, and the diagnostics that integrate over
the population or score abilities by EAP: :func:`mirt.itemfit` (S-X2, X2, G2,
PV-Q1, infit and outfit), ``compute_m2`` and ``compute_fit_indices``, the
residual and local-dependence statistics, ``vuong_test`` and the reliability
summaries. Pass the item model instead (``result.model``) for the standard
normal, or ``prior_cov`` to choose another population. M2 projects out the
estimated correlations like the free item parameters, so each costs one degree
of freedom. Standard errors treat the latent covariance as fixed. The
bootstrap utilities and ``bootstrap_lr`` refit the same structure, priors and
constraints and re-estimate the covariance each time, and parametric
replicates draw abilities from the estimated population.

``COV = F1*F1`` frees the variance of ``F1``. The variance is identified only
when ``FIXED`` holds a nonzero slope on ``F1``, as in a marker-item model,
or when the family fixes the slopes, as for the Rasch model:

.. code-block:: python

   rasch = mirt.fit_mirt(responses[:, :5], model="1PL", spec="F = 1-5\nCOV = F*F")
   marker = mirt.fit_mirt(
       responses[:, :5],
       model="2PL",
       spec="F = 1-5\nCOV = F*F\nFIXED = (1, a1)\nSTART = (1, a1, 1.0)",
   )

Using the estimator directly
----------------------------

:class:`~mirt.estimation.FactorCovarianceDensity` provides the same
correlation estimation for :class:`~mirt.EMEstimator` and any model whose
slopes follow a loading pattern:

.. code-block:: python

   from mirt.estimation import FactorCovarianceDensity

   pattern = spec.loading_pattern().astype(float)
   model = mirt.MultidimensionalModel(
       10, 2, model_type="confirmatory", loading_pattern=pattern
   )
   density = FactorCovarianceDensity(2)
   result = mirt.EMEstimator(n_quadpts=15, latent_density=density).fit(
       model, responses
   )
   print(density.correlation)  # result.latent_covariance holds the same matrix

``EMEstimator`` reports the final mean and covariance of any Gaussian latent
density as ``FitResult.latent_mean`` and ``FitResult.latent_covariance``, so
the consumers above treat both routes alike.

Limitations
-----------

* Only EM estimation is supported, and ``MEAN``, ``LBOUND``, ``UBOUND``,
  ``CONSTRAINB`` and other multiple-group keywords are not; ``CONSTRAIN``
  groups must each equate a single parameter.
* The quadrature grid has ``n_quadpts ** n_factors`` nodes. Lower
  ``n_quadpts`` (for example to 9) for three or more factors.
* Several factors are available for "2PL", "GRM" and "GPCM" models.
* A per-item ``model`` sequence must name one family, which is then fitted
  as that family; mixed item formats raise ``MirtValidationError``.
* EM keeps free slopes between 0.1 and 5, so loadings must be positive;
  reverse-score items that load negatively before fitting.
