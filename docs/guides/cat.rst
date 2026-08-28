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

See ``examples/fit_score_itemfit_cat.py`` for a complete script.
