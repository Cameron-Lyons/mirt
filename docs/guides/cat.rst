Computerized Adaptive Testing
=============================

``mirt`` includes a unidimensional CAT engine and multidimensional MCAT support.

Quick start
-----------

.. code-block:: python

   import mirt
   from mirt.cat import CATEngine

   data = mirt.load_dataset("LSAT7")["data"]
   fit = mirt.fit_mirt(data, model="2PL")

   engine = CATEngine(
       fit.model,
       item_selection="MFI",
       stopping_rule="SE",
       se_threshold=0.3,
       max_items=12,
   )
   result = engine.run_simulation(true_theta=0.5)
   print(result.theta, result.standard_error, result.n_items_administered)

Interactive responses
---------------------

For a live test, disclose ``engine.select_next_item()`` and pass the observed
category to ``engine.administer_item(response)``. Repeated selection returns the
same pending item until a valid response is submitted.

``administer_item`` accepts integer-valued numeric responses in the selected
item's category range: 0 or 1 for a binary item, and 0 through
``n_categories[item] - 1`` for an ordinal item. Invalid responses raise
``ValueError`` and preserve the pending item, response history, ability estimate,
and administration/exposure counts, so the caller can submit a corrected answer.
Both CAT and MCAT enforce this contract.

Batch simulation
----------------

Use ``run_batch_simulation`` for recovery studies across ability levels and
replications:

.. code-block:: python

   import numpy as np

   ability_grid = np.linspace(-3.0, 3.0, 25)
   simulations = engine.run_batch_simulation(
       ability_grid,
       n_replications=100,
   )

The native parallel path is selected automatically for unidimensional 1PL/2PL
models using EAP scoring, MFI selection, SE stopping with an optional maximum
length, and no content or exposure constraints. Native results include the exact
administered item and response sequences, making them fully compatible with
``CATResult`` serialization and downstream audits.

Otherwise, unmodified engines with MFI selection, EAP scoring, SE and/or
maximum-length stopping, no content or exposure constraints, and a built-in
unidimensional 1PL-4PL, GRM, GPCM, or PCM model simulate all examinees in lock
step, with one vectorized selection and scoring step per item position. Given the
same responses, lock step reproduces the item paths, estimates, standard errors,
stopping reasons, and histories of independent sessions. It reserves
``max_items`` (capped at the pool size) uniform draws per examinee, so a seeded
engine gives identical results to independent sessions for fixed-length tests and
other, equally distributed, responses when tests end early. Pass
``vectorized=False`` to run independent sessions, the reference implementation
used for every other configuration.

Simulation methods validate finite abilities, array dimensions, and positive
integer replication counts before changing the active session. Conditional
diagnostics accumulate bias, MSE, and average test length as sessions finish,
without storing every replicated estimate. Their working storage depends on the
number of factors rather than the number of replications.

Simulation reports
~~~~~~~~~~~~~~~~~~

:func:`~mirt.cat.simulation_report.summarize_cat_simulation` summarizes a
simulation study in the style of catR's ``simulateRespondents``. Pass the same
abilities and replication count given to ``run_batch_simulation``, and the pool
size so unused items are counted:

.. code-block:: python

   from mirt.cat import summarize_cat_simulation

   report = summarize_cat_simulation(
       simulations,
       ability_grid,
       n_items=fit.model.n_items,
       n_replications=100,
       theta_bins=[-3.0, -1.0, 0.0, 1.0, 3.0],
   )
   print(report.summary())
   exposure_table = report.to_dataframe("items")
   conditional_table = report.to_dataframe("conditional")

The report contains bias, RMSE, mean absolute error, the correlation of true and
estimated abilities, the mean reported SE, test-length statistics, stopping
reason counts, item selection counts and exposure rates with Wilson confidence
bounds, unused items, Chang and Ying's chi-square exposure index, and the test
overlap rate. The overlap rate is the mean number of items two examinees share
divided by the mean test length, ``sum(c * (c - 1)) / (N * (N - 1)) / mean_length``
for item selection counts ``c`` and ``N`` sessions. ``theta_bins`` adds
conditional bias, RMSE, test length, and mean SE per true-ability bin, given as
bin edges or as a number of equal-count bins. MCAT results report accuracy per
factor. Results from every simulation path work, including native results
without histories, and ``to_dict()`` exports JSON-compatible values.

Ability updates
---------------

For independent built-in binary models, EAP updates reuse the Gauss-Hermite grid
and evaluate only administered item curves in bounded blocks. The posterior is
recomputed against current item parameters after every response, so parameter
changes and resets do not reuse old likelihood evidence. Customized model
probability or likelihood hooks retain the general scoring path. MCAT continues
to retain the full posterior covariance.

Portable results
----------------

``CATResult`` and ``MCATResult`` can be stored or sent without a dataframe
package. Their dictionary exports contain only JSON-compatible Python values,
including the complete response, estimate, uncertainty, covariance, and item
information histories:

.. code-block:: python

   from mirt.cat import CATResult

   payload = result.to_dict()
   json_text = result.to_json(indent=2)

   restored = CATResult.from_json(json_text)
   assert restored.to_dict() == payload

The reconstruction methods validate required and unknown fields, administered
item and response counts, history lengths, and multidimensional array shapes.
Result objects also copy caller-owned arrays when they are created, so later
changes to the original arrays do not alter the stored administration.

Item selection
--------------

Common strategies (pass a string or strategy object):

* ``"MFI"`` — maximum Fisher information
* ``"MEI"`` — maximum expected information under the bounded posterior
* ``"KL"`` — Kullback–Leibler
* ``"random"`` / ``"a_stratified"`` / ``"urry"``

Names ignore case and surrounding whitespace, and ``_`` may replace ``-``.

a-stratified selection (:class:`~mirt.cat.selection.AStratified`) spends an
equal share of the test in each discrimination stratum, from the least to the
most discriminating. The engine schedules the strata over ``max_items`` (or the
pool size) unless the strategy sets its own ``test_length``. Items are ranked
within the current stratum by Fisher information, or by difficulty matching with
``within="b-matching"``.

MEI uses the complete response history to evaluate the posterior ability after
every possible response to each candidate item. The engine's ``n_quadpts`` and
``theta_bounds`` settings control this integration. This makes MEI sensitive to
posterior uncertainty and allows it to choose differently from point-estimate
MFI selection.

Shadow tests
~~~~~~~~~~~~

:class:`~mirt.cat.shadow.ShadowTestSelection` implements van der Linden's
shadow-test approach for constrained adaptive tests. Before every selection it
assembles a full-length test with :func:`~mirt.cat.assembly.assemble_form`. That
test contains every administered item, satisfies all constraints, and maximizes
information at the current ability estimate. Its most informative unadministered
item is administered next. A test that reaches the shadow-test length therefore
satisfies content minima and maxima, enemy pairs, all-or-none item bundles, and
a cost budget exactly, which greedy content balancing cannot guarantee:

.. code-block:: python

   import numpy as np
   from mirt.cat import CATEngine, ContentArea, ContentBlueprint, ShadowTestSelection

   expected_seconds = np.full(fit.model.n_items, 90.0)
   blueprint = ContentBlueprint([
       ContentArea("Algebra", items=set(range(60)), min_items=10, max_items=10),
       ContentArea("Geometry", items=set(range(60, 120)), min_items=10, max_items=10),
   ])
   selection = ShadowTestSelection(
       blueprint=blueprint,
       enemy_pairs={(3, 4), (70, 71)},
       item_bundles=[{10, 11, 12}],
       item_costs=expected_seconds,
       max_cost=1800.0,
   )
   engine = CATEngine(fit.model, item_selection=selection, max_items=20, min_items=20)

The shadow test spans the engine's ``max_items`` (or the pool size) unless
``test_length`` is given, and the engine must stop by that length. Tests that
stop earlier still respect maxima, enemy pairs, and the budget. Without
constraints, selection equals MFI. Randomesque exposure control draws among the
shadow test's free items. Items that Sympson-Hetter control makes ineligible are
left out of the shadow test unless that leaves it infeasible; the exclusions are
then relaxed, so constraints take precedence over exposure control. Progressive
exposure control bypasses the selection strategy and is rejected. Put content
constraints in the shadow test rather than in the engine's
``content_constraint``. Shadow tests need a unidimensional model, and batch
simulation runs their sessions independently. ``solver_options`` such as
``{"time_limit": 0.5}`` bound the per-item optimization time, and
``selection.last_shadow_test`` holds the most recent shadow test for audits.

Stopping rules
--------------

* ``"SE"`` — stop when standard error falls below ``se_threshold``
* ``"max_items"`` / ``MaxItemsStop`` — fixed test length
* ``"theta_change"`` / ``ThetaChangeStop`` — the estimate changes by at most a
  threshold for ``n_stable`` consecutive items
* ``"se_change"`` / ``SEChangeStop`` — the standard error changes by at most a
  threshold for ``n_stable`` consecutive items, so further items no longer buy
  precision
* ``MinInformationStop(model, threshold)`` — catR's ``minInfo`` rule: no
  remaining item has at least ``threshold`` information at the current estimate
* ``PredictedSEReductionStop(model, min_reduction)`` — the most informative
  remaining item would reduce the SE by less than ``min_reduction``, using the
  predicted SE ``1 / sqrt(SE**-2 + I(theta))`` (Choi, Grady, & Dodd, 2011)
* Combined rules via ``max_items`` plus an SE rule on ``CATEngine``, or
  ``CombinedStop`` for any combination

Variable-length tests often reach the length cap for examinees the pool cannot
measure precisely, typically at extreme abilities. Pairing the SE rule with a
precision-change rule ends those tests early:

.. code-block:: python

   from mirt.cat import CombinedStop, PredictedSEReductionStop, StandardErrorStop

   stopping = CombinedStop([
       StandardErrorStop(0.3),
       PredictedSEReductionStop(fit.model, min_reduction=0.01),
   ])
   engine = CATEngine(fit.model, stopping_rule=stopping, max_items=40)

The information-based rules hold the item model, so construct them with the
engine's model. They count every unadministered item as remaining, regardless
of exposure or content control.

CAT and MCAT rules share validation and composition. Thresholds must be finite
and positive and counts must be integers, so a NaN threshold that could never
stop a test is rejected. ``CombinedStop`` and ``CombinedMCATStop`` with
``operator="or"`` stop at the first rule that stops without evaluating later
rules, and with ``"and"`` evaluate every rule so stateful rules see each state.
``get_reason()`` reports the rule that stopped the most recent state.

For multidimensional classification, project the ability vector and its full
covariance onto a policy-relevant composite:

.. code-block:: python

   from mirt.cat import CompositeClassificationStop, MCATEngine

   classification = CompositeClassificationStop(
       weights=[0.7, 0.3],
       cut_score=0.0,
       confidence=0.95,
   )
   engine = MCATEngine(
       fit.model,
       stopping_rule=classification,
       min_items=8,
       max_items=30,
   )

The rule evaluates ``weights @ theta`` with standard error
``sqrt(weights @ covariance @ weights)``. It therefore incorporates factor
correlations and stops only after a one-sided decision is sufficiently
confident. Use :func:`~mirt.cat.mcat_stopping.create_mcat_stopping_rule` with
``"classification"`` to construct the same rule from configuration.

With ``scoring_method="EAP"`` (the default), ``MCATEngine`` retains the full
posterior covariance after each response, including cross-factor correlations.
Selection, stopping, state snapshots, and result histories all use this matrix.
``scoring_method="MAP"`` currently uses a diagonal approximation from marginal
standard errors. The D-, A-, C-optimality and Bayesian selection strategies share
the current precision matrix across candidates and batch covariance updates to
limit temporary memory for large item pools. Bayesian selection uses the
A-optimality criterion, the expected posterior variance under this covariance
update, and KL selection ranks items by ``trace(I_j(theta) @ covariance)``.

MCAT selection needs each item's Fisher information matrix. Models that define
``item_information_matrix`` use it, and compensatory dichotomous models without
it use ``p q a a^T``. Polytomous models without the method raise
:class:`~mirt.exceptions.MirtModelError` instead of approximating it.

Content balancing and exposure
------------------------------

Use :class:`~mirt.cat.content.ContentConstraint` / blueprints and exposure
controllers (Sympson–Hetter, randomesque, progressive) for operational CAT.
``CATEngine`` and ``MCATEngine`` apply them identically. Randomesque control
draws among the ``k`` items ranked highest by the selection strategy's
criterion, breaking ties by item index; random selection ranks by seeded uniform
scores. Progressive control randomizes early selections within an information
window and increasingly favors item information as the configured test limit
nears; ordinal models are evaluated item by item.
The first interactive session is included in exposure reports automatically.
Resetting before any item selection reuses the unused session; resetting after
selection starts a new examinee.

Fixed-form assembly
-------------------

Use :func:`~mirt.cat.assembly.assemble_form` to build a fixed form from a
calibrated item pool. The mixed-integer optimizer can maximize weighted test
information or match a target information curve while enforcing content-area
limits, required and excluded items, enemy pairs, and a cost budget.
Both fixed and parallel assembly accept ``item_bundles`` for all-or-none
selection of items sharing a passage, stimulus, or other common material.
Overlapping bundles form one connected bundle. Requiring any member requires
all members; excluding a member, or leaving it outside ``candidate_items``,
makes the whole bundle unavailable. A required bundle with an unavailable
member is rejected before optimization.

.. code-block:: python

   import numpy as np
   from mirt.cat import ContentArea, ContentBlueprint, assemble_form

   blueprint = ContentBlueprint([
       ContentArea("Algebra", items=set(range(10)), min_items=4, max_items=6),
       ContentArea("Geometry", items=set(range(10, 20)), min_items=4, max_items=6),
   ])
   assembly = assemble_form(
       fit.model,
       form_size=10,
       theta=np.linspace(-2.0, 2.0, 21),
       blueprint=blueprint,
       enemy_pairs={(1, 2), (11, 12)},
       item_bundles=[{3, 4}, {13, 14}],
   )
   print(assembly.selected_items)
   print(assembly.summary())

Parallel forms
~~~~~~~~~~~~~~

Use :func:`~mirt.cat.parallel_assembly.assemble_parallel_forms` to assemble
multiple forms simultaneously. The default max-min objective balances weighted
information across forms and assigns each non-anchor item to at most one form.
Common required items act as shared anchors; item reuse and pairwise overlap can
be relaxed explicitly when the pool is too small for disjoint forms.
Required bundles are shared anchors in their entirety, exempt from
``max_item_usage``; every member counts toward ``max_pairwise_overlap``.

.. code-block:: python

   from mirt.cat import assemble_parallel_forms

   parallel = assemble_parallel_forms(
       fit.model,
       n_forms=3,
       form_size=10,
       theta=np.linspace(-2.0, 2.0, 21),
       blueprint=blueprint,
       required_items={0, 10},
       max_item_usage=2,
       max_pairwise_overlap=4,
   )
   for form in parallel.forms:
       print(form.selected_items)
   print(parallel.overlap_matrix)

``target_information`` accepts a scalar, one common curve, or a
form-by-ability matrix for form-specific targets. Content, enemy-pair, and cost
constraints are applied independently to every form, while reuse and overlap
limits are enforced jointly across the full set.

Solver limits
~~~~~~~~~~~~~

Large problems can be bounded with ``solver_options`` such as
``{"time_limit": 30}`` or ``{"node_limit": 1000}``. A looser
``{"mip_rel_gap": 0.01}`` also shortens the search; its solutions still count as
optimal within that gap. When a limit stops the optimizer, both assembly
functions return the best feasible solution found, after checking that it
satisfies every constraint, with ``is_optimal=False`` and the relative
``mip_gap``; ``summary()`` reports both.
They raise only when no feasible solution was found or the constraints are
infeasible. Pass ``require_optimal=True`` to raise whenever optimality was not
proven.

See ``examples/fit_score_itemfit_cat.py`` for a complete script.
