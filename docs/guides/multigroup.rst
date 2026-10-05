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
to compare nested models. Item-level invariance tests, with anchor selection
and drop or add schemes, are described under
:func:`mirt.multigroup.multigroup_dif` in :doc:`dif`.

Configural fits standardize every group's latent mean to zero and covariance
to the identity. Metric fits keep means fixed and estimate nonreference
covariances when shared loading constraints identify the scale. Scalar and
strict fits can estimate nonreference means and covariances when location
and scale anchors remain. Partial-invariance specifications are checked
against their remaining anchors; the reference distribution always stays
standardized. Information criteria count only the estimated latent components.
These conventions prevent an arbitrary affine change of item and population
parameters from being reported as estimated group impact.

Fixed calibration and ordered means
-----------------------------------

The estimator supports known item parameter blocks and an optional ordering
of identified unidimensional population means:

.. code-block:: python

   from mirt.models import TwoParameterLogistic
   from mirt.multigroup import MultigroupEMEstimator, MultigroupModel

   bank = MultigroupModel(TwoParameterLogistic(n_items=10), n_groups=2)
   result = MultigroupEMEstimator(n_quadpts=31).fit(
       bank,
       [responses_g0, responses_g1],
       invariance="scalar",
       fixed_parameters={
           "discrimination": {0: 1.3, 1: 0.9},
           "difficulty": {0: -0.7, 1: 0.6},
       },
       mean_order=[0, 1],
   )

Fixed blocks use stored parameter names and zero-based item indices. They are
copied, validated, excluded from parameter counts, and preserved before the
first E-step and throughout fitting. Whole scale and location blocks can also
identify an external unidimensional metric under configural invariance.
``mean_order`` must contain every group index once. It constrains the Gaussian
population update on the same normalized quadrature grid as the likelihood;
it requires identified free nonreference means.

For restrictions within a parameter row, ``model.set_free_parameter_masks``
accepts Boolean arrays matching the stored parameter shapes. False entries
for additional restrictions retain the supplied independent coefficient values
during EM and Bock-Lieberman initialization and optimization. Masks cannot
free family constraints, such as Rasch slopes, ordinal padding, or nominal
reference categories. Copies preserve the masks, and model-fit diagnostics
use them to count nuisance parameters. A shared coordinate fixed in one group
is known in every linked group; conflicting fixed values are rejected.

Family masks also exclude metadata, padding, and dependent storage. For example,
GGUM estimates one half of its thresholds and reconstructs the reflected half;
common discriminations and mixture proportions also have dependent entries.
These entries follow their independent coefficients during updates rather than
adding fitted parameters.

For grade forms with different item positions or widths, use the physical-item
maps provided by :doc:`equating` anchor calibration.

See ``examples/multigroup_invariance.py``.
