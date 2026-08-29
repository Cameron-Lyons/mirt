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

Distributions and maps
----------------------

Ability estimates may be one- or two-dimensional. Standard errors are
optional, and a Wright map places item locations on the same scale.

.. code-block:: python

   scores = mirt.fscores(result, responses)
   mirt.plot_ability_distribution(
       scores.theta,
       se=scores.standard_error,
       factor=0,
   )
   mirt.plot_person_item_map(result.model, scores.theta, factor=0)

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
