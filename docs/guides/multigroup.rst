Multiple Group Analysis
=======================

Multigroup IRT estimates item parameters across groups with optional
invariance constraints on item parameters and latent distributions.

.. code-block:: python

   import numpy as np
   from mirt.multigroup import fit_multigroup

   data = np.vstack([responses_g0, responses_g1])
   groups = np.array([0] * len(responses_g0) + [1] * len(responses_g1))
   result = fit_multigroup(data, groups, model="2PL", invariance="metric")
   print(result.summary())

Invariance
----------

Common ``invariance`` specifications include ``configural``, ``metric``,
``scalar``, and ``strict``. Use :func:`mirt.multigroup.compare_invariance`
to compare nested models.

See ``examples/multigroup_invariance.py``.
