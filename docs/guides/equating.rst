Test Equating and Linking
=========================

Link separately calibrated forms onto a common scale using anchor items.

Legacy helper
-------------

.. code-block:: python

   import mirt

   result = mirt.equate(
       model_old,
       model_new,
       anchor_items_old,
       anchor_items_new,
       method="stocking_lord",
   )

Preferred API
-------------

Prefer :func:`mirt.equating.link` for additional linking methods, diagnostics,
and polytomous support (Stocking–Lord, Haebara, mean/sigma, mean/mean, and more).

Fixed-item calibration
----------------------

:func:`mirt.fixed_calib` calibrates new items onto an existing scale defined by
anchors.

See ``examples/equating.py``.
