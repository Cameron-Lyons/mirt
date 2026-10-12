Plotting
========

Plotting support is optional, keeping the core installation small. Install it
with the dedicated extra:

.. code-block:: bash

   pip install "mirt[plot]"

Importing ``mirt.plotting`` or resolving a top-level plotting helper does not
load NumPy, SciPy, or Matplotlib. NumPy loads on the first plotting call,
Matplotlib loads only when an axes must be created, and SciPy loads only when a
kernel-density overlay is requested.

Item and test curves
--------------------

The same helpers support dichotomous and polytomous models. An ICC call plots
one correct-response curve per dichotomous item and every response category
for each polytomous item.

.. code-block:: python

   import mirt

   result = mirt.fit_mirt(responses, model="GRM", n_categories=5)
   mirt.plot_icc(result.model, item_idx=[0, 1])
   mirt.plot_category_curves(result.model, item_idx=0)
   mirt.plot_information(result.model)
   mirt.plot_expected_score(result.model)
   mirt.plot_se(result.model)

Pass ``item_idx`` to ``plot_information`` to combine selected item curves with
the test curve, or set ``test_info=False`` to show item curves alone.

Multidimensional slices
-----------------------

Curve helpers vary one factor while holding all other factors at zero. Select
the dimension with the keyword-only ``factor`` argument:

.. code-block:: python

   mirt.plot_icc(result.model, item_idx=0, factor=1)
   mirt.plot_information(result.model, factor=1)
   mirt.plot_se(result.model, factor=1)

``plot_information`` and ``plot_se`` use the information along the selected
factor, the diagonal element ``I_kk(theta)`` of the Fisher information matrix
(``a_k^2 P Q`` for a compensatory 2PL item), as R's ``testinfo(mod, Theta,
degrees = ...)`` with the angle pointing along that factor. The standard error
is ``1 / sqrt(I_kk)``, the error of that factor when the others are known.
Note that :func:`mirt.testinfo` and :func:`mirt.iteminfo` return the trace of
the information matrix, summed over all factors, for multidimensional models
by default; their ``degrees`` argument gives these per-factor curves (see
:doc:`information`).

Distributions and maps
----------------------

Ability estimates may be one- or two-dimensional. Standard errors are
optional, and a Wright map places item locations on the same scale.

Item locations are difficulties for dichotomous items, mean thresholds for
graded items, mean step parameters for GPCM and PCM items and ``-d / a`` for
slope-intercept items; mixed-format models locate each component's items.
For a multidimensional model the map of ``factor`` shows the items with a
slope on that factor, located where their (mean) logit is zero with the other
factors at zero; items without such a slope are left out and counted in a
note on the map. Nominal response categories are unordered and have no item
location, so ``plot_person_item_map`` raises ``ValueError`` for a nominal
model.

.. code-block:: python

   scores = mirt.fscores(result, responses)
   mirt.plot_ability_distribution(
       scores.theta,
       se=scores.standard_error,
       factor=0,
   )
   mirt.plot_person_item_map(result.model, scores.theta, factor=0)

Item fit and DIF
----------------

``plot_itemfit`` and ``plot_dif`` take the mappings of
:func:`mirt.diagnostics.itemfit.compute_itemfit` and
:func:`mirt.diagnostics.compute_dif` or the pandas or polars DataFrames of
:func:`mirt.itemfit` and :func:`mirt.dif`, whose item labels name the bars.

``plot_dif`` draws ``|ets_effect|``, the effect behind the ETS class, against
the class B and C thresholds of its scale: ``|MH D-DIF|`` of 1 and 1.5 for
binary items and ``|SMD/SD|`` of 0.17 and 0.25 for polytomous items (see
:doc:`dif`). The scale comes from the ``ets_scale`` entry of
``compute_dif``; DataFrames do not carry it, so pass ``ets_scale="delta"`` or
``"smd"`` to draw the threshold lines:

.. code-block:: python

   mirt.plot_itemfit(mirt.itemfit(result, responses), statistic="outfit")
   mirt.plot_dif(mirt.dif(responses, groups), ets_scale="delta")

Customization
-------------

Every helper returns its axes and accepts an existing axes through ``ax``.
Extra keyword arguments are forwarded to the primary line, bar, or histogram
operation.

.. code-block:: python

   import matplotlib.pyplot as plt

   figure, axes = plt.subplots()
   mirt.plot_expected_score(
       result.model,
       ax=axes,
       color="navy",
       linewidth=3,
   )
   figure.tight_layout()
