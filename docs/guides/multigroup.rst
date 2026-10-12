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

Every group is integrated on one quadrature grid with its own normal prior
weights, as in R's ``multipleGroup``. The defaults are those of
:func:`mirt.fit_mirt`: 61 points equally spaced on ``[-6, 6]`` of the
reference group's scale for one factor, 31 per dimension for two, and so on
(``n_quadpts`` and ``quadrature`` change them). A focal group whose mean lies
several standard deviations above the reference group therefore keeps as many
effective nodes as the reference group, and a configural fit has the
log-likelihood of separate :func:`mirt.fit_mirt` fits of the groups. Fixed
standard-normal Gauss-Hermite grids, the ``quadrature="gauss-hermite"``
option, place only a few nodes under such a group and bias its mean and
standard deviation. When a group puts more than 0.01% of its normal mass
beyond the rectangular grid, about 3.7 standard deviations from an edge, as a
focal group at mean 4.5 does, EM extends the grid at the same node spacing
until every group's mean plus or minus six standard deviations is covered, up
to twice the number of points. R's fixed ``[-6, 6]`` grid truncates such a
group instead and shrinks its estimated mean and standard deviation; the fits
otherwise coincide. A group that the largest grid still truncates warns;
choose a reference group nearer the middle of the groups then.

Groups are ordered by their sorted labels. ``reference_group`` is an index
into that order or a label string, and defaults to the first group; an
integer that is also the label of another group raises ``ValueError``. Rows
whose group label is missing (``None`` or ``NaN``) are removed with a
``UserWarning``, as R does.

An item that a group never answered, such as an item missing from one form,
has no group-specific parameters in that group: they are neither estimated nor
counted in the degrees of freedom, a ``UserWarning`` names the items, and
``coef`` reports the parameters as ``NaN``. Its shared parameters are
estimated from the other groups. A group that never uses a response category
of an item whose category parameters are group-specific, such as the
thresholds of a configural or metric GRM, raises ``MirtValidationError``,
because those parameters are not identified; collapse the category or
constrain the parameters across groups.

Invariance
----------

Common ``invariance`` specifications include ``configural``, ``metric``,
``scalar``, and ``strict``. Use :func:`mirt.multigroup.compare_invariance`
to compare nested models, and
:func:`mirt.multigroup.test_invariance_hierarchy` for likelihood-ratio tests
between adjacent levels. Strict invariance also constrains the parameters
that are neither slopes nor intercepts; among the families of
:func:`~mirt.multigroup.fit_multigroup`, only the guessing and upper
asymptotes of the 3PL and 4PL are such parameters, so for the other families
the strict model is the scalar model. These functions then reuse the scalar fit for ``"strict"`` and report
the scalar-versus-strict comparison with ``df=0`` and a ``note``. Item-level
invariance tests, with anchor selection and drop or add schemes, are described
under :func:`mirt.multigroup.multigroup_dif` in :doc:`dif`.

Metric invariance constrains every slope parameter across groups. For nominal
(NRM) items these are all the category slopes, so a metric NRM fit has equal
category discriminations in every group. R's ``invariance = "slopes"``
constrains only the overall slope ``a1`` of its nominal parameterization and
keeps the category scoring ``ak`` group-specific, so R's metric model has more
parameters, and its configural-versus-metric test has fewer degrees of
freedom. Configural and scalar NRM fits are the same models in both.

Configural fits standardize every group's latent mean to zero and variance to
one. Metric fits keep means fixed and estimate nonreference variances when
shared loading constraints identify the scale. Scalar and strict fits can
estimate nonreference means and variances when location and scale anchors
remain. Partial-invariance specifications are checked against their remaining
anchors; the reference distribution always keeps mean zero and unit
variances. Information criteria count only the estimated latent components.
These conventions prevent an arbitrary affine change of item and population
parameters from being reported as estimated group impact.

``free_means`` and ``free_variances``, the ``free_means`` and ``free_var``
options of R's ``invariance``, set these latent moments independently of the
level. ``False`` fixes the nonreference moments at the reference values, and
``True`` estimates them, which requires invariant anchors: intercepts for
means, slopes for variances. Comparing scalar fits with and without free means
tests the group mean differences (latent impact):

.. code-block:: python

   from mirt.multigroup import fit_multigroup, invariance_lrt

   impact = fit_multigroup(data, groups, invariance="scalar")
   no_impact = fit_multigroup(data, groups, invariance="scalar", free_means=False)
   invariance_lrt(no_impact, impact)

``latent_pars()`` and ``summary()`` mark which moments were estimated.

Multidimensional models
-----------------------

``n_factors`` fits an exploratory model in which every item loads on every
factor; the reference factors stay uncorrelated, which identifies the
rotation, and ``F(F - 1) / 2`` rotational coordinates per group (once when the
slopes are shared) are not counted as parameters, as in R. ``spec`` takes the
model syntax of :func:`mirt.mirt_model` for a confirmatory structure:

.. code-block:: python

   result = fit_multigroup(
       data,
       groups,
       invariance="scalar",
       spec="""
           F1 = 1-6
           F2 = 7-12
           COV = F1*F2
       """,
   )

``COV`` pairs of distinct factors estimate those correlations in every group,
including the reference group, whose variances stay at one, as R's
``multipleGroup`` does; without ``COV`` the factors are uncorrelated. Group
variances follow the invariance level and ``free_variances``, so ``COV =
F*F`` terms are rejected, and so are ``START``, ``PRIOR`` and ``CONSTRAIN``
groups; ``FIXED`` holds coordinates at their starting values. Loading patterns
set directly with ``set_free_parameter_masks`` on the model passed to
:class:`~mirt.multigroup.MultigroupModel` estimate every factor correlation by
default when the pattern identifies them, that is when every factor has items
that do not load on it with full-rank loadings on the other factors.
``free_correlations`` of :func:`~mirt.multigroup.fit_multigroup` and
:meth:`MultigroupEMEstimator.fit <mirt.multigroup.MultigroupEMEstimator.fit>`
fixes them (``False``), frees them (``True``) or selects pairs (a Boolean
matrix). Nonreference groups
that estimate variances also estimate these covariances, so the configural,
metric and scalar models stay nested.

Built-in 1PL and 2PL items whose parameters are all shared or all
group-specific, or whose slopes are shared and difficulties group-specific as
in a metric fit, are updated together by batched Newton steps. Other built-in
binary items and GRM, GPCM, PCM and NRM items optimize all their coordinates
in every group at once with analytic gradients, shared coordinates as one
variable, and graded thresholds stay ordered in every group; custom models
and binary models restricted by ``set_free_parameter_masks`` update one
parameter at a time. Exploratory
multidimensional models start with slopes staggered across factors: equal
starting slopes would keep the factors identical, and EM would stop at that
saddle point.

Acceleration
------------

Plain EM converges slowly when the groups' latent means are weakly
identified, for example by a few anchor items, or when a focal group lies far
from the reference group: the means drift along a flat ridge of the
likelihood for hundreds or thousands of iterations. ``accelerate="squarem"``
applies SQUAREM extrapolation to every group's item parameters and latent
moments together, with a monotone safeguard that falls back to the plain EM
step, as :func:`mirt.fit_mirt` does for one group:

.. code-block:: python

   result = fit_multigroup(data, groups, invariance="scalar", accelerate="squarem")

It typically needs several times fewer E-steps and stops at an equal or higher
log-likelihood; ``n_iterations`` then counts E-steps. Anchor-based vertical
scales (:doc:`equating`) always use it.

Standard errors, scoring and saving
-----------------------------------

:func:`~mirt.multigroup.fit_multigroup` computes observed-information
standard errors by default, as R's ``multipleGroup(SE=TRUE)``: the exact Louis
information of each group's items under its own prior, with latent mean and
covariance columns from Oakes' identity, summed over the shared coordinates.
They match R's ``SE.type = "Oakes"`` errors. ``coef(include_standard_errors=True)``
and ``latent_pars(include_standard_errors=True)`` add ``_se`` columns, and
``vcov`` with ``vcov_labels`` holds the joint covariance of every free item
parameter and latent moment, for Wald tests of group impact. Pass
``compute_standard_errors=False`` to skip them; ``MultigroupEMEstimator``
computes them only when asked.

:func:`mirt.fscores` scores a multigroup result with each row's group label,
under that group's fitted items and latent distribution, as R's ``fscores``
does for a ``multipleGroup`` fit; the scores are on the reference group's
metric and come back in the order of the rows:

.. code-block:: python

   import mirt

   scores = mirt.fscores(result, data, groups=groups)  # or result.fscores(...)

``to_json`` and ``MultigroupFitResult.from_json`` (or ``to_dict`` and
``from_dict``) save and restore a result with its group models, invariance
links, fixed parameters, latent distributions and standard errors.

Fixed calibration and ordered means
-----------------------------------

The estimator supports known item parameter blocks and an optional ordering
of identified unidimensional population means:

.. code-block:: python

   from mirt.models import TwoParameterLogistic
   from mirt.multigroup import MultigroupEMEstimator, MultigroupModel

   bank = MultigroupModel(TwoParameterLogistic(n_items=10), n_groups=2)
   result = MultigroupEMEstimator().fit(
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
