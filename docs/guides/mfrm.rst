Many-Facet Rasch Models
=======================

Many-facet Rasch models (Linacre, 1994) extend the Rasch model with further
facets such as raters, tasks or criteria. The log-odds of each adjacent
category step are

.. math::

   \theta_n - b_i - \sum_f d_{f l_f} - \tau_{(i) k},

where ``b_i`` is the item difficulty, ``d_{f l}`` the measure of level ``l``
of facet ``f`` (for example rater severity) and ``tau`` the category
thresholds. ``ManyFacetRaschModel`` covers binary ratings and
``PolytomousMFRM`` covers ordinal ratings with shared (rating scale) or
item-specific (partial credit) thresholds.

Fitting a rater-by-item design
------------------------------

``fit_mfrm`` estimates every item, facet level and threshold by marginal
maximum likelihood, with person measures ``theta ~ N(0, sigma^2)``
integrated out on an equally spaced grid over ``+-6 sigma``. Facet
assignments may be a scalar, one level per person, or one level per person
and item:

.. code-block:: python

   import numpy as np

   import mirt
   from mirt.models import Facet, PolytomousMFRM

   rng = np.random.default_rng(1)
   n_persons, n_items = 500, 6
   raters = rng.integers(0, 4, size=(n_persons, n_items))
   tasks = rng.integers(0, 2, size=n_persons)
   theta = rng.normal(size=n_persons)

   truth = PolytomousMFRM(n_items, 5, [Facet("rater", 4), Facet("task", 2)])
   truth.set_facet_parameters("rater", np.array([-0.5, -0.1, 0.2, 0.4]))
   truth.set_thresholds(np.array([-1.0, -0.2, 0.3, 0.9]))
   assignments = {"rater": raters, "task": tasks}
   responses = truth.simulate(theta, assignments, seed=2)

   model = PolytomousMFRM(n_items, 5, [Facet("rater", 4), Facet("task", 2)])
   result = mirt.fit_mfrm(model, responses, assignments)

   print(result.summary())
   severity = result.facet_parameters["rater"]
   severity_se = result.facet_se["rater"]

The fitted parameters are also stored on ``model``. ``result.thresholds``
holds the category thresholds, ``result.sigma`` the person standard
deviation and ``result.theta`` the EAP person measures with posterior
standard deviations in ``result.theta_se``. Pass
``category_structure="partial_credit"`` to ``PolytomousMFRM`` for
item-specific thresholds; missing ratings are coded as negative values or
``NaN``.

Repeated ratings
----------------

When several raters score the same person on the same item, pass the
ratings in a long layout: each column of ``responses`` is one rating and
``item_indices`` names the item it belongs to. Facet assignments then refer
to the columns of ``responses``:

.. code-block:: python

   first = rng.integers(0, 4, size=(n_persons, n_items))
   second = (first + rng.integers(1, 4, size=first.shape)) % 4
   pairs = np.stack((first, second), axis=2).reshape(n_persons, -1)
   item_indices = np.repeat(np.arange(n_items), 2)

   long_responses = np.empty(pairs.shape, dtype=int)
   for column, item in enumerate(item_indices):
       simulated = truth.simulate(
           theta, {"rater": pairs[:, column], "task": tasks}, seed=column
       )
       long_responses[:, column] = simulated[:, item]

   result = mirt.fit_mfrm(
       PolytomousMFRM(n_items, 5, [Facet("rater", 4), Facet("task", 2)]),
       long_responses,
       {"rater": pairs, "task": tasks},
       item_indices=item_indices,
   )

``item_indices`` may have one entry per column or one per person and column.
Persons with fewer ratings pad their rows with missing responses; item and
facet indices at missing ratings are ignored.

Identification and data requirements
------------------------------------

Person measures have mean zero, so item difficulties are free and every
facet must be anchored: its levels average to ``Facet.anchor_value``
(zero by default). Threshold rows sum to zero. ``fit_mfrm`` raises
``MirtValidationError`` for an unanchored facet and for a design in which a
facet is confounded with items or other facets, for example when each rater
scores a disjoint set of items. It raises ``MirtDataError`` when an item or
facet level has no ratings or only extreme ratings, when a response
category is never used (per item for partial credit), and when no person
has two ratings while ``sigma`` is estimated. Pass ``estimate_sd=False`` to
fix ``sigma`` at 1.

Person grid
-----------

Many ratings per person make the person posteriors narrow, and a grid whose
spacing exceeds their width biases ``sigma`` and the item difficulties. By
default ``fit_mfrm`` starts with 41 nodes and refines the grid, up to 401
nodes, until its spacing is at most 1.5 times the 5th percentile of the
posterior standard deviations; ``result.n_quadpts`` reports the size used.
An explicit ``n_quadpts`` fixes the grid and emits a ``RuntimeWarning``
when it is too coarse.

Standard errors and fit statistics
----------------------------------

Standard errors come from the inverse of the exact observed information
(Louis' identity) and cover every level of an anchored facet.
``result.infit`` and ``result.outfit`` hold mean-square fit
statistics for every facet level and ``result.item_infit`` and
``result.item_outfit`` for every item. They average the squared residuals
over each person's posterior distribution, so both have expectation one
when the model holds. Values well above one flag erratic raters; values
well below one flag overly predictable ratings such as central tendency.
