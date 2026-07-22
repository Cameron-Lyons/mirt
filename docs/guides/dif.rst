Differential Item Functioning
=============================

DIF detects whether items function differently across groups after matching
on ability.

.. code-block:: python

   import mirt
   import numpy as np

   # data: (n_persons, n_items); groups: length n_persons
   result = mirt.dif(data, groups, model="2PL", method="likelihood_ratio")
   print(result)

Methods
-------

* ``likelihood_ratio`` — nested model LR test
* ``wald`` — Wald DIF
* ``lord`` — Lord's chi-square
* ``raju`` — Raju area measures

Related utilities
-----------------

* :func:`mirt.sibtest` — SIBTEST
* :func:`mirt.compute_grdif` — multi-group GRDIF with robust scaling
* :func:`mirt.compute_dtf` / :func:`mirt.compute_drf` — test/response functioning

See ``examples/dif_analysis.py``.
