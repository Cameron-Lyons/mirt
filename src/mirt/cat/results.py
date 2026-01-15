"""CAT result classes for tracking adaptive testing state and outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class MCATState:
    """Current state during multidimensional CAT administration.

    This class tracks the evolving state of an MCAT session, including the
    current ability estimates across multiple dimensions, items administered,
    and whether the test has reached a stopping condition.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Current ability estimates, shape (n_factors,).
    covariance : NDArray[np.float64]
        Posterior covariance matrix, shape (n_factors, n_factors).
    standard_error : NDArray[np.float64]
        Standard errors for each dimension (sqrt of diagonal of covariance).
    items_administered : list[int]
        Indices of items that have been administered.
    responses : list[int]
        Responses to administered items.
    n_items : int
        Number of items administered so far.
    is_complete : bool
        Whether the CAT session has reached a stopping condition.
    next_item : int | None
        Index of the next item to administer, or None if complete.
    """

    theta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    items_administered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    n_items: int = 0
    is_complete: bool = False
    next_item: int | None = None

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        return len(self.theta)

    @property
    def trace_covariance(self) -> float:
        """Trace of the covariance matrix (sum of variances)."""
        return float(np.trace(self.covariance))

    @property
    def det_covariance(self) -> float:
        """Determinant of the covariance matrix."""
        return float(np.linalg.det(self.covariance))

    def __repr__(self) -> str:
        theta_str = ", ".join(f"{t:.3f}" for t in self.theta)
        se_str = ", ".join(f"{s:.3f}" for s in self.standard_error)
        return (
            f"MCATState(theta=[{theta_str}], "
            f"se=[{se_str}], "
            f"n_items={self.n_items}, "
            f"complete={self.is_complete})"
        )


@dataclass
class MCATResult:
    """Final result of a completed multidimensional CAT session.

    This class contains the complete record of an MCAT administration,
    including the final ability estimates across all dimensions, all items
    administered, responses given, and the history of ability estimates.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Final ability estimates, shape (n_factors,).
    covariance : NDArray[np.float64]
        Final posterior covariance matrix, shape (n_factors, n_factors).
    standard_error : NDArray[np.float64]
        Standard errors for each dimension.
    items_administered : list[int]
        Indices of all items administered in order.
    responses : NDArray[np.int_]
        Array of responses to administered items.
    n_items_administered : int
        Total number of items administered.
    stopping_reason : str
        Description of why the test stopped.
    theta_history : list[NDArray[np.float64]]
        History of ability estimates after each item.
    se_history : list[NDArray[np.float64]]
        History of standard errors after each item.
    covariance_history : list[NDArray[np.float64]]
        History of covariance matrices after each item.
    item_info_history : list[float]
        History of item information values for selected items.
    """

    theta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    items_administered: list[int]
    responses: NDArray[np.int_]
    n_items_administered: int
    stopping_reason: str
    theta_history: list[NDArray[np.float64]] = field(default_factory=list)
    se_history: list[NDArray[np.float64]] = field(default_factory=list)
    covariance_history: list[NDArray[np.float64]] = field(default_factory=list)
    item_info_history: list[float] = field(default_factory=list)

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        return len(self.theta)

    def summary(self) -> str:
        """Return a formatted summary of the MCAT result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "MCAT Result Summary",
            "=" * 50,
            f"Dimensions:            {self.n_factors}",
            f"Items administered:    {self.n_items_administered}",
            f"Stopping reason:       {self.stopping_reason}",
            "",
            "Final theta estimates:",
        ]
        for i, (t, se) in enumerate(zip(self.theta, self.standard_error)):
            lines.append(f"  Dimension {i + 1}: {t:7.4f} (SE: {se:.4f})")

        lines.extend(
            [
                "",
                f"Trace(Cov):            {np.trace(self.covariance):.4f}",
                f"Det(Cov):              {np.linalg.det(self.covariance):.6f}",
                "",
                "Response pattern:",
                f"  Correct: {np.sum(self.responses == 1)} / {self.n_items_administered}",
            ]
        )
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert MCAT history to a DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns for step, item, response, and theta/se per dimension.
        """
        from mirt.utils.dataframe import create_dataframe

        n = len(self.items_administered)
        data: dict[str, Any] = {
            "step": list(range(1, n + 1)),
            "item": self.items_administered,
            "response": list(self.responses),
        }

        for d in range(self.n_factors):
            theta_key = f"theta_{d + 1}"
            se_key = f"se_{d + 1}"
            if self.theta_history:
                data[theta_key] = [th[d] for th in self.theta_history[:n]]
            else:
                data[theta_key] = [np.nan] * n
            if self.se_history:
                data[se_key] = [se[d] for se in self.se_history[:n]]
            else:
                data[se_key] = [np.nan] * n

        if self.item_info_history:
            data["info"] = self.item_info_history[:n]

        return create_dataframe(data)

    def plot_convergence(self) -> Any:
        """Plot theta and SE convergence for each dimension.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with subplots showing theta and SE history for each dimension.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        n_factors = self.n_factors
        fig, axes = plt.subplots(
            n_factors, 2, figsize=(10, 3 * n_factors), squeeze=False
        )

        steps = list(range(1, len(self.theta_history) + 1))
        colors = plt.cm.tab10(np.linspace(0, 1, n_factors))

        for d in range(n_factors):
            theta_d = [th[d] for th in self.theta_history]
            se_d = [se[d] for se in self.se_history]

            axes[d, 0].plot(steps, theta_d, "-o", markersize=4, color=colors[d])
            axes[d, 0].axhline(y=self.theta[d], color="r", linestyle="--", alpha=0.7)
            axes[d, 0].set_ylabel(f"Theta {d + 1}")
            axes[d, 0].grid(True, alpha=0.3)

            axes[d, 1].plot(steps, se_d, "-o", markersize=4, color=colors[d])
            axes[d, 1].axhline(
                y=self.standard_error[d], color="r", linestyle="--", alpha=0.7
            )
            axes[d, 1].set_ylabel(f"SE {d + 1}")
            axes[d, 1].grid(True, alpha=0.3)

        axes[0, 0].set_title("Theta Convergence")
        axes[0, 1].set_title("SE Convergence")
        axes[-1, 0].set_xlabel("Items Administered")
        axes[-1, 1].set_xlabel("Items Administered")

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        theta_str = ", ".join(f"{t:.3f}" for t in self.theta)
        return (
            f"MCATResult(theta=[{theta_str}], "
            f"n_items={self.n_items_administered}, "
            f"reason='{self.stopping_reason}')"
        )


@dataclass
class CATState:
    """Current state during CAT administration.

    This class tracks the evolving state of a CAT session, including the
    current ability estimate, items administered, and whether the test
    has reached a stopping condition.

    Attributes
    ----------
    theta : float
        Current ability estimate.
    standard_error : float
        Standard error of the current ability estimate.
    items_administered : list[int]
        Indices of items that have been administered.
    responses : list[int]
        Responses to administered items (0/1 for dichotomous, 0..k for polytomous).
    n_items : int
        Number of items administered so far.
    is_complete : bool
        Whether the CAT session has reached a stopping condition.
    next_item : int | None
        Index of the next item to administer, or None if complete.
    """

    theta: float
    standard_error: float
    items_administered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    n_items: int = 0
    is_complete: bool = False
    next_item: int | None = None

    def __repr__(self) -> str:
        return (
            f"CATState(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items}, "
            f"complete={self.is_complete})"
        )


@dataclass
class CATResult:
    """Final result of a completed CAT session.

    This class contains the complete record of a CAT administration,
    including the final ability estimate, all items administered,
    responses given, and the history of ability estimates.

    Attributes
    ----------
    theta : float
        Final ability estimate.
    standard_error : float
        Standard error of the final ability estimate.
    items_administered : list[int]
        Indices of all items administered in order.
    responses : NDArray[np.int_]
        Array of responses to administered items.
    n_items_administered : int
        Total number of items administered.
    stopping_reason : str
        Description of why the test stopped (e.g., "SE threshold reached").
    theta_history : list[float]
        History of ability estimates after each item.
    se_history : list[float]
        History of standard errors after each item.
    item_info_history : list[float]
        History of item information values for selected items.
    """

    theta: float
    standard_error: float
    items_administered: list[int]
    responses: NDArray[np.int_]
    n_items_administered: int
    stopping_reason: str
    theta_history: list[float] = field(default_factory=list)
    se_history: list[float] = field(default_factory=list)
    item_info_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary of the CAT result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "CAT Result Summary",
            "=" * 40,
            f"Final theta estimate:  {self.theta:.4f}",
            f"Standard error:        {self.standard_error:.4f}",
            f"Items administered:    {self.n_items_administered}",
            f"Stopping reason:       {self.stopping_reason}",
            "",
            "Response pattern:",
            f"  Correct: {np.sum(self.responses == 1)} / {self.n_items_administered}",
            f"  Items:   {self.items_administered}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert CAT history to a DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns: item, response, theta, se, info.
        """
        from mirt.utils.dataframe import create_dataframe

        n = len(self.items_administered)
        data: dict[str, Any] = {
            "step": list(range(1, n + 1)),
            "item": self.items_administered,
            "response": list(self.responses),
            "theta": self.theta_history[:n] if self.theta_history else [np.nan] * n,
            "se": self.se_history[:n] if self.se_history else [np.nan] * n,
        }

        if self.item_info_history:
            data["info"] = self.item_info_history[:n]

        return create_dataframe(data)

    def to_array(self) -> NDArray[np.float64]:
        """Convert result to numpy array.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_items, 4) containing [item, response, theta, se].
        """
        n = len(self.items_administered)
        arr = np.zeros((n, 4), dtype=np.float64)
        arr[:, 0] = self.items_administered
        arr[:, 1] = self.responses
        if self.theta_history:
            arr[:, 2] = self.theta_history[:n]
        if self.se_history:
            arr[:, 3] = self.se_history[:n]
        return arr

    def plot_convergence(self) -> Any:
        """Plot theta and SE convergence over items.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with two subplots showing theta and SE history.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        steps = list(range(1, len(self.theta_history) + 1))

        ax1.plot(steps, self.theta_history, "b-o", markersize=4)
        ax1.axhline(y=self.theta, color="r", linestyle="--", alpha=0.7)
        ax1.set_ylabel("Theta Estimate")
        ax1.set_title("CAT Convergence")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, self.se_history, "g-o", markersize=4)
        ax2.axhline(y=self.standard_error, color="r", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Items Administered")
        ax2.set_ylabel("Standard Error")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"CATResult(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items_administered}, "
            f"reason='{self.stopping_reason}')"
        )
