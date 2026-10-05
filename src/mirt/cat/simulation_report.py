"""catR-style summaries of CAT and MCAT simulation studies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.cat.results import CATResult, MCATResult


@dataclass(frozen=True)
class CATSimulationReport:
    """Accuracy, test-length, and item-exposure summary of a CAT simulation.

    Accuracy statistics are floats for unidimensional results and arrays with
    one value per factor for MCAT results.

    Attributes
    ----------
    n_examinees : int
        Number of simulated sessions.
    n_items : int
        Size of the item pool.
    bias, rmse, mae : float or ndarray
        Mean, root-mean-square, and mean absolute estimation error.
    correlation : float or ndarray
        Pearson correlation of true and estimated abilities, NaN when either
        is constant.
    mean_standard_error : float or ndarray
        Mean of the reported standard errors.
    mean_length, sd_length : float
        Mean and sample standard deviation of the test lengths.
    min_length, max_length : int
        Shortest and longest test.
    stopping_reasons : dict[str, int]
        Session counts per stopping reason, most frequent first.
    selection_counts : ndarray
        Number of sessions that administered each item.
    exposure_rates : ndarray
        ``selection_counts / n_examinees``.
    exposure_lower, exposure_upper : ndarray
        Wilson confidence bounds for the exposure rates.
    confidence_level : float
        Two-sided confidence level of the exposure bounds.
    overlap_rate : float
        Expected proportion of items shared by two random examinees,
        ``sum(c * (c - 1)) / (N * (N - 1)) / mean_length`` for selection
        counts ``c`` and ``N`` sessions: the mean number of items two tests
        share, divided by the mean test length. NaN for fewer than two
        sessions.
    chi_square : float
        Chang and Ying's (1999) exposure index
        ``sum((r - L / n) ** 2) / (L / n)`` for exposure rates ``r``, mean
        test length ``L``, and pool size ``n``. Zero means uniform exposure.
    conditional : dict[str, ndarray] or None
        Unidimensional results grouped by true ability: bin ``lower`` and
        ``upper`` bounds, ``n``, ``bias``, ``rmse``, ``mean_length``, and
        ``mean_standard_error``. ``None`` unless ``theta_bins`` was given.
    """

    n_examinees: int
    n_items: int
    bias: float | NDArray[np.float64]
    rmse: float | NDArray[np.float64]
    mae: float | NDArray[np.float64]
    correlation: float | NDArray[np.float64]
    mean_standard_error: float | NDArray[np.float64]
    mean_length: float
    sd_length: float
    min_length: int
    max_length: int
    stopping_reasons: dict[str, int]
    selection_counts: NDArray[np.int64]
    exposure_rates: NDArray[np.float64]
    exposure_lower: NDArray[np.float64]
    exposure_upper: NDArray[np.float64]
    confidence_level: float
    overlap_rate: float
    chi_square: float
    conditional: dict[str, NDArray[Any]] | None = None

    @property
    def max_exposure(self) -> float:
        """Largest item exposure rate."""
        return float(np.max(self.exposure_rates))

    @property
    def unused_items(self) -> NDArray[np.intp]:
        """Indices of items no session administered."""
        return np.flatnonzero(self.selection_counts == 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the report."""

        def plain(value: Any) -> Any:
            return value.tolist() if isinstance(value, np.ndarray) else value

        payload: dict[str, Any] = {
            field.name: plain(getattr(self, field.name))
            for field in fields(self)
            if field.name != "conditional"
        }
        payload["stopping_reasons"] = dict(self.stopping_reasons)
        payload["conditional"] = (
            None
            if self.conditional is None
            else {name: plain(values) for name, values in self.conditional.items()}
        )
        return payload

    def to_dataframe(self, table: Literal["items", "conditional"] = "items") -> Any:
        """Return the item-exposure or conditional table as a DataFrame.

        Parameters
        ----------
        table : {"items", "conditional"}, default="items"
            ``"items"`` gives one row per pool item with its selection count,
            exposure rate, and confidence bounds. ``"conditional"`` gives one
            row per ability bin.
        """
        from mirt.utils.dataframe import create_dataframe

        if table == "items":
            return create_dataframe(
                {
                    "item": np.arange(self.n_items),
                    "selection_count": self.selection_counts,
                    "exposure_rate": self.exposure_rates,
                    "exposure_lower": self.exposure_lower,
                    "exposure_upper": self.exposure_upper,
                }
            )
        if table == "conditional":
            if self.conditional is None:
                raise ValueError("the report has no conditional results")
            return create_dataframe(dict(self.conditional))
        raise ValueError("table must be 'items' or 'conditional'")

    def summary(self) -> str:
        """Return a formatted multi-line summary."""
        lines = [
            "CAT simulation report",
            "=" * 40,
            f"Examinees:            {self.n_examinees}",
            f"Item pool:            {self.n_items}",
            "",
            "Accuracy",
            f"  Bias:               {_format(self.bias, sign=True)}",
            f"  RMSE:               {_format(self.rmse)}",
            f"  MAE:                {_format(self.mae)}",
            f"  Correlation:        {_format(self.correlation)}",
            f"  Mean reported SE:   {_format(self.mean_standard_error)}",
            "",
            "Test length",
            f"  Mean (SD):          {self.mean_length:.2f} ({self.sd_length:.2f})",
            f"  Range:              {self.min_length} - {self.max_length}",
            "",
            "Stopping reasons",
            *(
                f"  {count:6d}  {reason}"
                for reason, count in self.stopping_reasons.items()
            ),
            "",
            "Item exposure",
            f"  Maximum rate:       {self.max_exposure:.3f}",
            f"  Unused items:       {self.unused_items.size} / {self.n_items}",
            f"  Chi-square index:   {self.chi_square:.3f}",
            f"  Test overlap rate:  {self.overlap_rate:.3f}",
        ]
        if self.conditional is not None:
            lines.extend(
                [
                    "",
                    "Conditional results",
                    f"  {'theta range':>17s} {'n':>6s} {'bias':>8s} {'rmse':>7s} "
                    f"{'length':>7s}",
                ]
            )
            table = self.conditional
            for row in range(table["n"].size):
                lines.append(
                    f"  [{table['lower'][row]:6.2f}, {table['upper'][row]:6.2f}] "
                    f"{int(table['n'][row]):6d} {table['bias'][row]:+8.3f} "
                    f"{table['rmse'][row]:7.3f} {table['mean_length'][row]:7.2f}"
                )
        return "\n".join(lines)


def _format(value: float | NDArray[np.float64], *, sign: bool = False) -> str:
    pattern = "{:+.4f}" if sign else "{:.4f}"
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(pattern.format(item) for item in value) + "]"
    return pattern.format(value)


def _wilson_interval(
    counts: NDArray[np.int64],
    n_trials: int,
    confidence_level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Wilson score bounds for binomial proportions."""
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rates = counts / n_trials
    z_squared = z_value**2
    denominator = 1.0 + z_squared / n_trials
    center = (rates + z_squared / (2.0 * n_trials)) / denominator
    half_width = (
        z_value
        * np.sqrt(rates * (1.0 - rates) / n_trials + z_squared / (4.0 * n_trials**2))
        / denominator
    )
    return np.maximum(0.0, center - half_width), np.minimum(1.0, center + half_width)


def _correlation(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Columnwise Pearson correlation, NaN when a column is constant."""
    first_centered = first - first.mean(axis=0)
    second_centered = second - second.mean(axis=0)
    scale = np.sqrt(
        np.sum(first_centered**2, axis=0) * np.sum(second_centered**2, axis=0)
    )
    covariance = np.sum(first_centered * second_centered, axis=0)
    return np.divide(
        covariance,
        scale,
        out=np.full(covariance.shape, np.nan),
        where=scale > 0.0,
    )


def _bin_edges(
    theta_bins: int | ArrayLike,
    true_theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return bin edges from a quantile count or explicit increasing edges."""
    if isinstance(theta_bins, Integral) and not isinstance(
        theta_bins, (bool, np.bool_)
    ):
        if theta_bins < 1:
            raise ValueError("theta_bins must be a positive integer or bin edges")
        quantiles = np.quantile(true_theta, np.linspace(0.0, 1.0, int(theta_bins) + 1))
        edges = np.unique(quantiles)
        return edges if edges.size > 1 else np.repeat(edges, 2)
    try:
        edges = np.asarray(theta_bins, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "theta_bins must be a positive integer or bin edges"
        ) from error
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("theta_bins edges must be finite and strictly increasing")
    return edges


def _conditional_table(
    edges: NDArray[np.float64],
    true_theta: NDArray[np.float64],
    errors: NDArray[np.float64],
    lengths: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
) -> dict[str, NDArray[Any]]:
    """Summarize sessions in ``[lower, upper)`` bins; the last bin is closed."""
    n_bins = edges.size - 1
    bins = np.searchsorted(edges, true_theta, side="right") - 1
    bins[true_theta == edges[-1]] = n_bins - 1
    inside = (bins >= 0) & (bins < n_bins)
    bins, errors = bins[inside], errors[inside]
    lengths, standard_errors = lengths[inside], standard_errors[inside]

    counts = np.bincount(bins, minlength=n_bins)

    def bin_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
        totals = np.bincount(bins, weights=values, minlength=n_bins)
        return np.divide(totals, counts, out=np.full(n_bins, np.nan), where=counts > 0)

    return {
        "lower": edges[:-1].copy(),
        "upper": edges[1:].copy(),
        "n": counts,
        "bias": bin_mean(errors),
        "rmse": np.sqrt(bin_mean(errors**2)),
        "mean_length": bin_mean(lengths),
        "mean_standard_error": bin_mean(standard_errors),
    }


def summarize_cat_simulation(
    results: Sequence[CATResult] | Sequence[MCATResult],
    true_theta: ArrayLike,
    *,
    n_items: int,
    n_replications: int = 1,
    theta_bins: int | ArrayLike | None = None,
    confidence_level: float = 0.95,
) -> CATSimulationReport:
    """Summarize a CAT or MCAT simulation study, like catR's simulateRespondents.

    Parameters
    ----------
    results : sequence of CATResult or MCATResult
        Completed sessions, for example from ``run_batch_simulation``.
        Results from every simulation path are supported; ability and SE
        histories are not used.
    true_theta : array-like
        True abilities, shape ``(n_thetas,)`` for CAT or
        ``(n_thetas, n_factors)`` for MCAT. Each row is repeated
        ``n_replications`` times in a row, matching the order of
        ``run_batch_simulation(true_theta, n_replications)``.
    n_items : int
        Size of the item pool, so that unused items are counted.
    n_replications : int, default=1
        Sessions per true ability.
    theta_bins : int or array-like, optional
        Conditional results by true ability (unidimensional only): an
        integer number of equal-count quantile bins, or strictly increasing
        bin edges. Bins are ``[lower, upper)`` except the last, which is
        closed; sessions outside explicit edges are left out.
    confidence_level : float, default=0.95
        Two-sided confidence level of the Wilson exposure-rate bounds.

    Returns
    -------
    CATSimulationReport
        Accuracy, test-length, stopping, exposure, and overlap statistics.

    Examples
    --------
    >>> thetas = np.linspace(-2, 2, 9)
    >>> results = engine.run_batch_simulation(thetas, n_replications=50)
    >>> report = summarize_cat_simulation(
    ...     results, thetas, n_items=engine.model.n_items, n_replications=50,
    ...     theta_bins=4,
    ... )
    >>> print(report.summary())
    """
    sessions = list(results)
    if not sessions:
        raise ValueError("results must contain at least one session")
    multidimensional = isinstance(sessions[0], MCATResult)
    expected_type = MCATResult if multidimensional else CATResult
    if not all(isinstance(result, expected_type) for result in sessions):
        raise TypeError("results must all be CATResult or all be MCATResult")
    if (
        isinstance(n_items, (bool, np.bool_))
        or not isinstance(n_items, Integral)
        or n_items < 1
    ):
        raise ValueError("n_items must be a positive integer")
    if (
        isinstance(n_replications, (bool, np.bool_))
        or not isinstance(n_replications, Integral)
        or n_replications < 1
    ):
        raise ValueError("n_replications must be a positive integer")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, Real
    ):
        raise ValueError("confidence_level must be in (0, 1)")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0 or 0.5 + confidence / 2.0 >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    try:
        estimates = np.array(
            [np.atleast_1d(result.theta) for result in sessions], dtype=np.float64
        )
        standard_errors = np.array(
            [np.atleast_1d(result.standard_error) for result in sessions],
            dtype=np.float64,
        )
    except ValueError as error:
        raise ValueError(
            "results must all estimate the same number of factors"
        ) from error
    if estimates.ndim != 2 or standard_errors.shape != estimates.shape:
        raise ValueError("results must all estimate the same number of factors")
    n_factors = estimates.shape[1]
    try:
        truth = np.asarray(true_theta, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("true_theta must contain numeric values") from error
    if truth.ndim == 0 or (truth.ndim == 1 and n_factors == 1):
        truth = truth.reshape(-1, 1)
    elif truth.ndim == 1:
        truth = truth.reshape(1, -1)
    if truth.ndim != 2 or truth.shape[1] != n_factors:
        raise ValueError(f"true_theta must have {n_factors} value(s) per examinee")
    if not np.all(np.isfinite(truth)):
        raise ValueError("true_theta must contain only finite values")
    truth = np.repeat(truth, int(n_replications), axis=0)
    n_sessions = len(sessions)
    if truth.shape[0] != n_sessions:
        raise ValueError(
            f"true_theta with n_replications={n_replications} describes "
            f"{truth.shape[0]} sessions, but {n_sessions} results were given"
        )

    lengths = np.array([result.n_items_administered for result in sessions])
    administered = [np.asarray(result.items_administered) for result in sessions]
    for items in administered:
        if items.size and items.max() >= n_items:
            raise ValueError("results contain an item index outside n_items")
        if np.unique(items).size != items.size:
            raise ValueError("an item cannot be administered twice in one session")
    selection_counts = np.bincount(
        np.concatenate(administered).astype(np.intp), minlength=n_items
    ).astype(np.int64)
    exposure_rates = selection_counts / n_sessions
    exposure_lower, exposure_upper = _wilson_interval(
        selection_counts, n_sessions, confidence
    )

    mean_length = float(np.mean(lengths))
    if n_sessions > 1 and mean_length > 0.0:
        shared_pairs = float(np.sum(selection_counts * (selection_counts - 1)))
        overlap_rate = shared_pairs / (n_sessions * (n_sessions - 1)) / mean_length
    else:
        overlap_rate = float("nan")
    uniform_rate = mean_length / n_items
    chi_square = (
        float(np.sum((exposure_rates - uniform_rate) ** 2) / uniform_rate)
        if uniform_rate > 0.0
        else float("nan")
    )

    errors = estimates - truth

    def per_factor(values: NDArray[np.float64]) -> float | NDArray[np.float64]:
        return values if multidimensional else float(values[0])

    conditional = None
    if theta_bins is not None:
        if multidimensional:
            raise ValueError("theta_bins requires unidimensional results")
        conditional = _conditional_table(
            _bin_edges(theta_bins, truth[:, 0]),
            truth[:, 0],
            errors[:, 0],
            lengths.astype(np.float64),
            standard_errors[:, 0],
        )

    reasons = Counter(result.stopping_reason for result in sessions)
    return CATSimulationReport(
        n_examinees=n_sessions,
        n_items=int(n_items),
        bias=per_factor(np.mean(errors, axis=0)),
        rmse=per_factor(np.sqrt(np.mean(errors**2, axis=0))),
        mae=per_factor(np.mean(np.abs(errors), axis=0)),
        correlation=per_factor(_correlation(truth, estimates)),
        mean_standard_error=per_factor(np.mean(standard_errors, axis=0)),
        mean_length=mean_length,
        sd_length=float(np.std(lengths, ddof=1)) if n_sessions > 1 else 0.0,
        min_length=int(np.min(lengths)),
        max_length=int(np.max(lengths)),
        stopping_reasons=dict(
            sorted(reasons.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        selection_counts=selection_counts,
        exposure_rates=exposure_rates,
        exposure_lower=exposure_lower,
        exposure_upper=exposure_upper,
        confidence_level=confidence,
        overlap_rate=overlap_rate,
        chi_square=chi_square,
        conditional=conditional,
    )
