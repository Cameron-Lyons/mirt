Mixed-Format Tests
==================

Many tests combine item formats: multiple-choice items scored right or wrong
and constructed-response items scored in ordered categories. A
:class:`~mirt.models.mixed_format.MixedItemModel` gives each item its own
family while all items measure one latent trait, like the per-item
``itemtype`` vector of R's ``mirt``.

Fitting a mixed-format test
---------------------------

Pass one family name per item as ``model``. Items of one family form one
component, in the order the families first appear:

.. code-block:: python

   import mirt

   itemtypes = ["3PL"] * 20 + ["GRM"] * 5
   result = mirt.fit_mirt(responses, model=itemtypes, n_categories=4)

   result.model.item_types[18:22]      # ['3PL', '3PL', 'GRM', 'GRM']
   result.coef()                        # one row per item

A scalar ``n_categories`` applies to the polytomous items; a sequence gives
one count per item, with 2 for dichotomous items. When it is omitted the
counts are inferred from the data. A sequence that names a single family fits
that family's ordinary model, also with ``spec`` model syntax and in
``fit_multigroup``, which raise ``MirtValidationError`` for mixed families.
Mixed formats are estimated by EM; MHRM and Gibbs sampling raise
``MirtValidationError``.

Parameters are stored by the components and named with their family, for
example ``"3PL.guessing"`` or ``"GRM.thresholds"``. Each array follows its
component's own item order. These qualified names are used by
``start_values``, ``fixed``, standard errors and ``FitResult.vcov``:

.. code-block:: python

   import numpy as np

   result = mirt.fit_mirt(
       responses,
       model=itemtypes,
       start_values={"3PL.guessing": np.full(20, 0.2)},
       fixed={"3PL.guessing": True},
   )

Item priors given as a mapping may name a parameter with its family prefix
for one component, or without one for every component that has it. As for a
single family, standard errors and ``FitResult.vcov`` add each prior's
curvature to the information.

``FitResult.coef()`` lists every item with ``NaN`` for the parameters its
family lacks; :meth:`MixedItemModel.item_parameter_arrays` returns the same
layout as arrays. Parameters that a component shares across its items, such
as the thresholds of a rating-scale component, have no per-item value, so
both raise ``MirtModelError`` for such components; use
``FitResult.parameter_statistics()`` or the component's own parameters
instead.

``FitResult.to_json()`` records each component's family and items, so
``FitResult.from_json()`` rebuilds a mixed-format result of the ``fit_mirt``
families (rating-scale components excluded) for scoring.

Estimation details
------------------

:class:`~mirt.estimation.mixed_format_em.MixedFormatEMEstimator` evaluates the
whole test in each E-step and then updates every component through its own
family's M-step, so graded and partial-credit components keep the native
optimizer and 1PL and 2PL components the batched Newton step. Standard errors
default to the observed information (``se_method="oakes"``) when every
component is a unidimensional built-in 1PL-4PL, GRM, GPCM, PCM, RSM or GRSM
model, and to itemwise complete-data curvature otherwise. The observed
information treats all components jointly, so ``FitResult.vcov`` includes
covariances between items of different families. SQUAREM acceleration falls
back to plain EM.

The estimator also accepts a model built directly from components, which may
be any item models that share ``n_factors``:

.. code-block:: python

   from mirt import GradedResponseModel, MixedItemModel, ThreeParameterLogistic

   model = MixedItemModel(
       [
           (ThreeParameterLogistic(20), range(20)),
           (GradedResponseModel(5, n_categories=4), range(20, 25)),
       ]
   )
   result = mirt.MixedFormatEMEstimator().fit(model, responses)

A rating-scale component (:class:`~mirt.models.polytomous.RatingScaleModel`
or :class:`~mirt.models.polytomous.GradedRatingScaleModel`) shares its
thresholds among its own items (the graded rating-scale model also its
slope). Its M-step ends with the same joint update of those shared parameters
as a single-family fit, so they are estimated with the rest of the test, and
results label them once per component, for example ``"RSM.thresholds[1]"``.
The exact observed information covers the shared parameters as well.
Equality constraints across items (``constraints``) are not available for
mixed-format models.

Calibrated item pools
---------------------

Components calibrated on a common scale can be combined for scoring,
simulation and adaptive testing without refitting:

.. code-block:: python

   pool = MixedItemModel(
       [(mc_result.model, mc_positions), (cr_result.model, cr_positions)]
   )
   scores = mirt.fscores(pool, responses, method="EAP")
   simulated = pool.simulate(theta, seed=1)
   engine = mirt.CATEngine(pool)

Binary items are simulated by the rule of their dichotomous family (a
response is 1 when its uniform draw is below the success probability), so a
pool of one dichotomous component reproduces that component's ``simulate``
for the same seed.

Item fit, person fit, information functions, plausible values,
``bootstrap_se``, ``multi_start_fit``, ``BLEstimator`` and ``compute_se`` work
with mixed-format models. Tools that need one family's parameter layout, such
as ``mod2values``, linking constants and multiple-group models, raise an
error; apply them to the components in ``model.components``.

``fixed_item_calibration`` also raises. To calibrate new items against fixed
anchors, hold the anchor coordinates with ``set_free_parameter_masks`` and
estimate the population with the mixed-format estimator:

.. code-block:: python

   from mirt.estimation.latent_density import GaussianDensity

   # The model holds the anchor values; the first ten 3PL items are anchors.
   anchors = np.isin(model.parameter_items("3PL.guessing"), range(10))
   model.set_free_parameter_masks(
       {f"3PL.{name}": ~anchors for name in ("discrimination", "difficulty", "guessing")}
   )
   density = GaussianDensity(n_dimensions=1, estimate_mean=True, estimate_cov=True)
   estimator = mirt.MixedFormatEMEstimator(latent_density=density)
   result = estimator.fit(model, responses, start="model")

A mixed-format model is not a latent-class mixture; see
:class:`~mirt.models.mixture.MixtureIRT` for that.
