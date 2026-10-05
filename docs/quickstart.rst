Quick Start Guide
=================

This guide walks through a minimal end-to-end IRT workflow with MIRT.

Basic Usage
-----------

1. **Prepare your data**: Response data should be a 2D NumPy array where rows
   are respondents and columns are items.

2. **Fit a model**: Use ``fit_mirt()`` with an appropriate model type.

3. **Score respondents**: Use ``fscores()`` to estimate person abilities.

4. **Analyze fit**: Use item and person fit diagnostics.

Explore the sample datasets without loading response-matrix copies:

.. code-block:: python

   import mirt

   print(mirt.list_datasets())
   print(mirt.describe_dataset("LSAT7"))

For repeated read-only workflows, ``load_dataset(name, copy=False)`` returns
arrays backed by the process-local cache. Those arrays are intentionally
read-only; omit the argument when writable, independent arrays are needed.

Example: 2PL Model
------------------

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]

   result = mirt.fit_mirt(responses, model="2PL", max_iter=200)
   print(result.summary())

   print("Discrimination:", result.model.parameters["discrimination"])
   print("Difficulty:", result.model.parameters["difficulty"])

   scores = mirt.fscores(result, responses, method="EAP")
   print("Theta estimates:", scores.theta[:5])
   print("SE estimates:", scores.standard_error[:5])

Example: Graded Response Model
------------------------------

For polytomous items (Likert scales, etc.):

.. code-block:: python

   import mirt

   data = mirt.load_dataset("verbal_aggression")
   responses = data["data"]

   result = mirt.fit_mirt(responses, model="GRM", n_categories=3)
   print(result.summary())

For items with different response scales, pass one category count per item,
such as ``n_categories=[2, 3, 4]`` for a three-item response matrix. Omitting
``n_categories`` infers each item's count separately from its largest observed
code, with a minimum of two. Explicit counts retain categories absent from the
sample and are required for wholly unobserved items. A scalar count applies
to every item. GRM EM calibration constrains adjacent thresholds to remain
ordered, including free thresholds next to fixed ones.

Scoring
-------

After fitting a model, you can score new respondents:

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]
   result = mirt.fit_mirt(responses, model="2PL")

   new_responses = responses[:10]
   score_result = mirt.fscores(result, new_responses, method="MAP")
   print(score_result.theta)

EAP scoring automatically evaluates large respondent sets in memory-bounded
batches. It detects repetition-heavy response matrices and reuses posterior
calculations when pattern compression is expected to improve throughput. Use
``batch_size`` when a deployment needs an explicit upper bound on the number of
response rows evaluated at once:

.. code-block:: python

   score_result = mirt.fscores(
       result,
       new_responses,
       method="EAP",
       batch_size=5_000,
   )

Smaller batches reduce peak working memory; larger batches can improve throughput
on systems with ample memory. Compressed rows are expanded back to their original
respondent order without changing estimates or standard errors.

Model Fit
---------

Evaluate model fit with various diagnostics:

.. note::

   ``itemfit()`` and ``personfit()`` return DataFrame objects. Install either
   pandas or polars (for example, ``pip install mirt[pandas]``) to use these
   outputs.

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]
   result = mirt.fit_mirt(responses, model="2PL")

   item_stats = mirt.itemfit(
       result,
       responses,
       statistics=["infit", "outfit", "S_X2"],
       p_adjust="fdr_bh",
   )
   person_stats = mirt.personfit(
       result,
       responses,
       statistics=["infit", "outfit", "Zh"],
       p_adjust="holm",
   )
   print(item_stats.head())
   print(person_stats.head())

Conditional S-X2 item fit
~~~~~~~~~~~~~~~~~~~~~~~~~

``S_X2`` compares item category counts within each exact total-score group.
Expected proportions follow the `Orlando--Thissen derivation
<https://doi.org/10.1177/01466216000241003>`_: for item :math:`j`, category
:math:`k` and total score :math:`s`,

.. math::

   E_{jsk} =
   \frac{\int P_j(k\mid\theta)\,P(S_{-j}=s-k\mid\theta)\,f(\theta)\,d\theta}
        {\int P(S=s\mid\theta)\,f(\theta)\,d\theta}.

Score recursion integrates these probabilities without scoring each respondent.
The Pearson statistic uses observed counts and :math:`N_s E_{jsk}`. The legacy
``n_groups`` argument is deprecated and ignored; exact scores define the groups.

Binary items pool sparse adjacent score rows. Ordinal items follow
`Kang and Chen's generalized S-X2
<https://files.eric.ed.gov/fulltext/ED510479.pdf>`_: exclude zero and perfect
total scores, pool incomplete tail score rows, then combine sparse adjacent
response categories within each row. ``min_expected=1.0`` is the default;
zero disables sparse-cell pooling. When two adjacent groups are eligible,
pooling chooses the one with the smaller expected count. Category counts may
differ across items; category scores must be consecutive integers starting at
zero.

The ``df`` column counts retained category contrasts minus the number of
estimated parameters for that item. Fixed parameters and padded ordinal
thresholds are excluded using the model's free parameter masks. Supply
``item_parameter_counts`` explicitly for models with shared parameters, including
RSM, GRSM, explanatory item-feature models, and testlet models. A global array
whose length happens to equal the item count still requires this explicit
allocation. For externally known item parameters, supply one zero per item.
Nonpositive degrees of freedom appear as ``df=0`` and ``p_value=NaN``; a
remaining expected cell below the requested minimum also gives ``NaN``.
An ordinal item whose maximum score exceeds the remaining test's maximum
has no full-category score group and returns ``S_X2=NaN``, ``df=0``.

By default, S-X2 integrates over independent standard-normal factors with 41
quadrature points per factor and assumes conditionally independent items.
``MixtureIRT`` and ``HigherOrderCDM`` need joint integration over their shared
latent classes or mastery patterns and are rejected. Testlet and bifactor
models remain conditionally independent when all their factors are specified;
the quadrature distribution must match those factors, including testlet
variances. Full-factor testlet curves use the supplied testlet-factor values;
they do not rescale those values using the model's stored testlet variances.
S-X2 therefore needs explicit quadrature masses for that full-factor
distribution when it differs from independent standard normals.
Fitted nonstandard factor distributions are not selected
automatically.
For a different fitted latent distribution,
supply both its grid and nonnegative probability masses:

.. code-block:: python

   item_stats = mirt.itemfit(
       result,
       responses,
       statistics=["S_X2"],
       quadrature_points=latent_grid,
       quadrature_weights=latent_masses,
       p_adjust="holm",
   )

Weights are normalized automatically. Person abilities supplied to
``compute_itemfit`` affect infit/outfit; S-X2 always integrates the latent
distribution. Complete responses are required for S-X2. ``na_rm=True``
explicitly removes persons with any negative or NaN response from S-X2.
Infit/outfit continue using available responses under the negative missing-code
convention. Complete-case inference pertains to the retained sample; its latent
distribution must remain appropriate. Out-of-range categories, noninteger codes,
invalid model probabilities, and observed scores with zero model probability
raise descriptive errors.

Standardized and ability-grouped item fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``z_infit`` and ``z_outfit`` (items and persons) are Wilson-Hilferty
standardized mean squares, approximately standard normal under the model.
Their variances use the second and fourth central moments of each modeled
score (Wright and Masters, 1982). ``compute_outfit_infit`` reports them with
``include_standardized=True`` and follows the same mean-square rules as
``itemfit()`` and ``personfit()``.

``X2`` (Bock/Yen Q1) and ``G2`` group respondents into ``n_groups`` quantiles
of their abilities (default 10) and compare category counts with the model at
each group's mean ability. Because the abilities are estimates, their p-values
are approximate and liberal, severely so on short tests; prefer ``S_X2`` or
``PV_Q1`` for inference there. ``PV_Q1`` (Chalmers and Ng, 2017)
recomputes ``X2`` on ``n_plausible`` posterior draws and reports the median,
with ``seed`` for reproducibility. These statistics require unidimensional
models. Unknown statistic names raise ``MirtValidationError``.

.. code-block:: python

   grouped = mirt.itemfit(
       result,
       responses,
       statistics=["z_infit", "z_outfit", "X2", "PV_Q1"],
       n_plausible=50,
       seed=1,
   )

Overall model fit
~~~~~~~~~~~~~~~~~

For overall fit, use the fitted item model directly:

.. code-block:: python

   overall = mirt.compute_fit_indices(result.model, responses, n_quadpts=31)
   print(overall)

``compute_m2()`` tests binary items with M2 and ordinal items with the collapsed
M2* statistic. Both use first-order scores and pairwise score products, their
model-implied sampling covariance, and a projection that removes estimated
parameter directions. This follows `Maydeu-Olivares and Joe (2006)
<https://doi.org/10.1007/s11336-005-1295-9>`_ and `Cai and Hansen (2013)
<https://doi.org/10.1111/j.2044-8317.2012.02050.x>`_. The ordinal statistic assumes
the category codes 0, 1, ... have meaningful order and spacing; it does not test
every category-specific univariate and bivariate margin.

The default null model has independent standard normal latent factors and
conditionally independent item responses. ``MixtureIRT`` and ``HigherOrderCDM``
are not supported: averaging their shared latent classes or mastery patterns
before forming joint moments would violate this assumption. For testlet and
bifactor models, integration conditions on all factors, and the stated
standard-normal distribution must match the intended null for every factor.
Full-factor testlet curves do not apply the stored testlet variances; those
variances enter the separate one-dimensional marginalized model methods.
Default M2 quadrature does not automatically select that marginalized
distribution.
Increase ``n_quadpts`` and check
stability for steep item curves. If ``theta`` is supplied, it defines a fixed
person ability design, with conditional expectations and covariance for each
person. Ability estimates obtained from the same responses do not satisfy that
fixed-design assumption, so their chi-square p-values are not calibrated by
this calculation.

Negative response codes and NaNs are missing. Available-case item and pair
moments retain their own observation counts; the covariance also retains
overlap between moments. With quadrature, validity requires missingness
independent of the responses and latent abilities. With a fixed ability
design, observation masks may depend on that design, but not on the random
responses after conditioning on it. Response-dependent missingness requires
an explicit missingness model.

Degrees of freedom use the numerical ranks of the covariance and free
parameter tangent. Too few observable moments, especially with short ordinal
tests, can leave zero testable dimensions: ``M2``, its p-value, and inferential
fit indices then return NaN. SRMSR remains available as a descriptive score
correlation residual. Deterministic moments are removed from the stochastic
rank; an observed response impossible under the model yields an infinite
statistic when testable dimensions remain. CFI and TLI use a covariance-weighted
independence baseline with estimated item means. Chi-square calibration is
asymptotic and assumes regular, consistent parameter estimation.

Probability and response calculations run in bounded blocks. The dense
moment covariance itself requires space proportional to the fourth power of
the item count; this is a computational consideration for very long tests.
