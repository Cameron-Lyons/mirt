HTML Reports
============

The reporting API produces standalone HTML documents with embedded styling. Report
titles, item names, group labels, table cells, captions, and image descriptions are
escaped automatically before they are inserted into the document.

Lightweight reports
-------------------

Tabular reports do not require the plotting extra. Pass ``include_plots=False`` to
generate model summaries and diagnostics using only the core dependencies:

.. code-block:: python

   from mirt import fit_mirt, generate_report

   result = fit_mirt(responses, model="2PL")
   html = generate_report(
       result,
       responses,
       report_type="full_diagnostic",
       include_plots=False,
       output_path="diagnostic.html",
   )

When ``output_path`` is provided, the same rendered document is returned and written
to disk. Diagnostics are computed only once.

Reports with visualizations
---------------------------

Install the plotting extra to embed PNG visualizations directly in the HTML file:

.. code-block:: bash

   pip install "mirt[plot]"

Plots are enabled by default. They can also be requested explicitly when using a
report builder:

.. code-block:: python

   from mirt.reports import ItemAnalysisReport

   report = ItemAnalysisReport(
       result,
       responses,
       theta=scores.theta,
       include_plots=True,
       title="Operational Item Review",
   )
   report.save("item-analysis.html")

Available builders
------------------

``ItemAnalysisReport``
   Item parameters, standard errors, confidence intervals, and item-fit statistics.

``ModelFitReport``
   Global fit indices, interpretation guidance, information, and measurement error.

``FullDiagnosticReport``
   Combined parameter, item-fit, model-fit, local-dependence, and ability sections.

``DIFAnalysisReport``
   Two-group DIF summaries, ETS classifications, and optional effect-size plots.

Responses may be arrays or pandas or polars DataFrames. As in
:func:`mirt.fscores`, DataFrame columns are matched to the model's item names,
so a reordered frame gives the same report, and ``NaN``, the nulls of nullable
columns (``pd.NA``) and negative codes are missing responses.

The model-fit sections report M2 for binary items and M2* for ordinal items.
When M2* has no degrees of freedom, as on short ordinal scales, they report the
hybrid C2 statistic instead, label it ``C2`` and base RMSEA, CFI and TLI on it.

Given ``theta``, the item-analysis and full diagnostic reports draw one Wright
map per factor. A map that does not apply, such as that of a nominal response
model, is replaced by a note rather than failing the report.

All builders accept ``include_plots=False``. Plot figures are closed even if a
visualization fails, so repeated report creation does not accumulate figure resources.
