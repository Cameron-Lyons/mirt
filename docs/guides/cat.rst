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

MEI uses the complete response history to evaluate the posterior ability after
every possible response to each candidate item. The engine's ``n_quadpts`` and
``theta_bounds`` settings control this integration. This makes MEI sensitive to
posterior uncertainty and allows it to choose differently from point-estimate
MFI selection.

Stopping rules
--------------

* ``"SE"`` — stop when standard error falls below ``se_threshold``
* ``"max_items"`` / ``MaxItemsStop`` — fixed test length
* Combined rules via ``max_items`` plus an SE rule on ``CATEngine``

Content balancing and exposure
------------------------------

Use :class:`~mirt.cat.content.ContentConstraint` / blueprints and exposure
controllers (Sympson–Hetter, randomesque, progressive) for operational CAT.
Progressive control randomizes early selections within an information window
and increasingly favors item information as the configured test limit nears.

Fixed-form assembly
-------------------

Use :func:`~mirt.cat.assembly.assemble_form` to build a fixed form from a
calibrated item pool. The mixed-integer optimizer can maximize weighted test
information or match a target information curve while enforcing content-area
limits, required and excluded items, enemy pairs, and a cost budget.

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
   )
   print(assembly.selected_items)
   print(assembly.summary())

See ``examples/fit_score_itemfit_cat.py`` for a complete script.
